"""Repair the pictures around hard drops with I_PCM patches.

When a reference twin D is dropped, the decoder's default reference lists
change for the pictures that follow it until D would have left them: an
index that pointed at D now points at the kept twin K, and with more than one
active reference later indices can shift as well (e.g. [D, K] -> [K, older]).

Which pictures are affected is found from header-only ref lists of the
original and the deduplicated stream (`stream.new_lists`). Only those
pictures, and B pictures whose co-located picture is one of them, are parsed.
For each, motion is derived twice from the same macroblock records: in the
original context and in the new one. Every 4x4 block whose reference picture
or motion vector differs is a direct reader of changed data; its macroblock
is replaced by an I_PCM copy of the original decoded pixels (`patch.py`).

Work is split by IDR period and run in parallel.
"""
import mmap
import multiprocessing as mp
import time

from .decode import decode_pocs, frame_size
from .h264mv import derive_motion
from .patch import patch_pictures
from .roundtrip_test import decode_picture
from .stream import new_lists
from .system import REPAIR_PEAK, lower_priority, peak_rss, pool_size


def find_candidates(spss, pics, drop):
    """Original decode indices of kept pictures whose reference context
    changes. Returns (candidates, new lists by orig dec)."""
    nl = new_lists(spss, pics, drop)
    cand = {d for d, lists in nl.items() if lists != pics[d]['lists']}
    for d in nl:
        p = pics[d]
        if p['type'] == 'B' and p['lists'][1] and p['lists'][1][0] in cand:
            cand.add(d)
    return sorted(cand), nl


def _slim(p):
    keep = ('slices', 'idr', 'ref_idc', 'dec', 'poc', 'lists', 'type', 'au_start', 'au_end', 'disp', 'period')
    return {k: p[k] for k in keep if k in p}


def build_tasks(pics, new_pics, new_of, cands):
    """One task per IDR period that holds candidates."""
    by_period = {}
    for d in cands:
        by_period.setdefault(pics[d]['period'], []).append(d)
    starts = {}
    for p in pics:
        starts.setdefault(p['period'], []).append(p['dec'])
    nstarts = {}
    for q in new_pics:
        nstarts.setdefault(q['period'], []).append(q['dec'])
    tasks = []
    for g, ds in sorted(by_period.items()):
        tasks.append({
            'period': g, 'cands': ds,
            'orig': {d: _slim(pics[d]) for d in starts[g]},
            'new': {d: _slim(new_pics[d]) for d in nstarts[g]},
            'new_of': {d: new_of[d] for d in starts[g] if d in new_of},
        })
    return tasks


_G = {}


def _init(orig_es, new_es, spss, ppss):
    lower_priority()
    _G.update(orig_es=orig_es, new_es=new_es, spss=spss, ppss=ppss)


def changed_blocks(m_orig, m_new, orig_of_new, W, H):
    """4x4 blocks whose reference picture or motion vector differs between
    the original and the new reference context."""
    out = []
    for gy in range(H * 4):
        for gx in range(W * 4):
            for l in range(2):
                ro = m_orig.refpic[l][gy][gx]
                rn = m_new.refpic[l][gy][gx]
                rn = None if rn is None else orig_of_new[rn]
                if ro != rn or (ro is not None and m_orig.mv[l][gy][gx] != m_new.mv[l][gy][gx]):
                    out.append((gx, gy))
                    break
    return out


def _mb_rect(gx, gy, mv, W, H):
    """Macroblocks a 4x4 block's luma and chroma prediction can read
    (6-tap / bilinear margins, generously rounded)."""
    x = gx * 4 + (mv[0] >> 2)
    y = gy * 4 + (mv[1] >> 2)
    # out-of-picture reads replicate the edge samples: clamp both ends
    cx = lambda v: min(max(v, 0), W * 16 - 1)
    cy = lambda v: min(max(v, 0), H * 16 - 1)
    x0, x1 = cx(x - 4), cx(x + 3 + 5)
    y0, y1 = cy(y - 4), cy(y + 3 + 5)
    return y0 // 16, y1 // 16 + 1, x0 // 16, x1 // 16 + 1


class PixelOracle:
    """Answers whether a changed block's prediction is provably unchanged,
    from the original decoded pixels of the reference pictures."""

    def __init__(self, frames, poc_of, W, H):
        self.frames, self.poc_of, self.W, self.H = frames, poc_of, W, H
        self.eq = {}

    def same(self, a, b, rect):
        if a == b:
            return True
        key = (min(a, b), max(a, b))
        if key not in self.eq:
            from .verify import mb_maxdiff
            m, _ = mb_maxdiff(self.frames[self.poc_of[a]], self.frames[self.poc_of[b]], self.W, self.H)
            self.eq[key] = m == 0
        y0, y1, x0, x1 = rect
        sel = self.eq[key][y0:y1, x0:x1]
        assert sel.size, rect
        return bool(sel.all())

    def harmless(self, gx, gy, mo, mn, orig_of_new):
        W, H = self.W, self.H
        used = []
        changed = False
        for l in range(2):
            ro = mo.refpic[l][gy][gx]
            rn = mn.refpic[l][gy][gx]
            rn = None if rn is None else orig_of_new[rn]
            if (ro is None) != (rn is None):
                return False
            if ro is None:
                continue
            mv = mo.mv[l][gy][gx]
            if mv != mn.mv[l][gy][gx]:
                return False
            if ro != rn:
                changed = True
                if not self.same(ro, rn, _mb_rect(gx, gy, mv, W, H)):
                    return False
            used.append((ro, mv))
        if changed and len(used) == 2:
            # implicit bi-prediction weights follow the reference POCs, so the
            # result is only unchanged if both predictions are identical
            (r0, mv0), (r1, mv1) = used
            if mv0 != mv1 or not self.same(r0, r1, _mb_rect(gx, gy, mv0, W, H)):
                return False
        return True


def run_task(task):
    t0 = time.time()
    spss, ppss = _G['spss'], _G['ppss']
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    orig, new, new_of = task['orig'], task['new'], task['new_of']
    orig_of_new = {v: k for k, v in new_of.items()}
    f = open(_G['new_es'], 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    need = set(task['cands'])
    for d in task['cands']:
        if orig[d]['type'] == 'B':
            need.add(orig[d]['lists'][1][0])
    parsed, m_orig, blocks = {}, {}, {}
    patches, sniff = {}, []
    log_lines = []
    for d in sorted(need):
        po, nd = orig[d], new_of[d]
        pn = new[nd]
        assert pn['poc'] == po['poc']
        dd = decode_picture(mm, spss, ppss, pn)
        t = po['type']
        if t == 'B':
            col_o = po['lists'][1][0]
            assert orig_of_new[pn['lists'][1][0]] == col_o, 'co-located picture changed'
            mo = derive_motion(dict(po), dd['mbs'], W, H, orig, m_orig[col_o])
            mn = derive_motion(dict(pn), dd['mbs'], W, H, new, parsed[new_of[col_o]][1])
        else:
            mo = derive_motion(dict(po), dd['mbs'], W, H, orig)
            mn = derive_motion(dict(pn), dd['mbs'], W, H, new)
        m_orig[d] = mo
        parsed[nd] = (dd, mn)
        if d in task['cands'] and t != 'I':
            blocks[d] = changed_blocks(mo, mn, orig_of_new, W, H)
    repl, stats = {}, {}
    todo = {d: b for d, b in blocks.items() if b}
    if todo:
        # original pixels of the affected pictures and of every picture their
        # changed blocks read, old and new context
        refs = set(todo)
        for d, bl in todo.items():
            mo, mn = m_orig[d], parsed[new_of[d]][1]
            for gx, gy in bl:
                for l in range(2):
                    if mo.refpic[l][gy][gx] is not None:
                        refs.add(mo.refpic[l][gy][gx])
                    if mn.refpic[l][gy][gx] is not None:
                        refs.add(orig_of_new[mn.refpic[l][gy][gx]])
        first = min(orig)
        poc_of = {d: orig[d]['poc'] for d in orig}
        frames = decode_pocs(_G['orig_es'], orig, first, max(refs) + 1, {poc_of[d] for d in refs},
                             frame_size(sps))
        oracle = PixelOracle(frames, poc_of, W, H)
        for d, bl in sorted(todo.items()):
            mo, mn = m_orig[d], parsed[new_of[d]][1]
            mbs = {(gx // 4, gy // 4) for gx, gy in bl if not oracle.harmless(gx, gy, mo, mn, orig_of_new)}
            sniff.append({'dec': d, 'display': orig[d]['disp'], 'type': orig[d]['type'],
                          'changed_blocks': len(bl), 'mbs': len(mbs)})
            if mbs:
                patches[new_of[d]] = sorted(mbs)
    if patches:
        repl = patch_pictures(mm, spss, ppss, new, patches, lambda nd: frames[new[nd]['poc']],
                              log=log_lines.append, parsed=parsed, stats=stats)
    mm.close()
    f.close()
    return {'period': task['period'], 'repl': repl, 'stats': stats, 'sniff': sniff,
            'patches': {nd: len(v) for nd, v in patches.items()}, 'log': log_lines,
            'seconds': round(time.time() - t0, 1), 'peak_rss': peak_rss()}


def repair(orig_es, new_es, spss, ppss, pics, new_pics, drop, workers, log=print):
    """Returns ({new dec: NAL}, summary dict)."""
    new_of = {p['dec']: i for i, p in enumerate(p for p in pics if p['dec'] not in set(drop))}
    cands, _ = find_candidates(spss, pics, drop)
    tasks = build_tasks(pics, new_pics, new_of, cands)
    log(f'repair: {len(cands)} pictures with a changed reference context in {len(tasks)} IDR periods')
    results = []
    nproc = pool_size(workers, REPAIR_PEAK, len(tasks)) if tasks else 0
    if tasks:
        log(f'repair: {nproc} workers')
        # workers are recycled every 2 tasks so their memory peak does not creep
        with mp.get_context('spawn').Pool(nproc, initializer=_init, maxtasksperchild=2,
                                          initargs=(orig_es, new_es, spss, ppss)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
                results.append(r)
                n_pcm = sum(r['patches'].values())
                log(f'  [{i + 1}/{len(tasks)}] IDR period {r["period"]}: {len(r["repl"])} pictures rewritten, '
                    f'{n_pcm} PCM MBs ({r["seconds"]} s)')
    repl = {}
    summary = {'candidate_pictures': len(cands), 'periods': len(tasks), 'pictures_rewritten': 0,
               'pcm_mbs': 0, 'bytes_added': 0, 'skip_to_explicit': 0, 'direct_to_explicit': 0,
               'qp_carriers': 0, 'workers': nproc,
               'worker_peak_rss_mb': round(max((r['peak_rss'] for r in results), default=0) / 2**20),
               'sniff': [], 'pictures': []}
    for r in sorted(results, key=lambda r: r['period']):
        repl.update(r['repl'])
        summary['sniff'] += r['sniff']
        for nd, s in sorted(r['stats'].items()):
            summary['pictures_rewritten'] += 1
            summary['pcm_mbs'] += s['pcm']
            summary['bytes_added'] += s['bytes_after'] - s['bytes_before']
            summary['skip_to_explicit'] += s['skip_explicit']
            summary['direct_to_explicit'] += s['direct_explicit']
            summary['qp_carriers'] += s['qp_carrier']
            summary['pictures'].append({'out_dec': nd, 'display': new_pics[nd]['disp'], **s})
    return repl, summary

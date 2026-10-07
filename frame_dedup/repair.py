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
import collections
import hashlib
import mmap
import multiprocessing as mp
import os
import time

import numpy as np

from .decode import decode_pocs, frame_size, iter_frames
from .h264hdr import unescape
from .h264mv import derive_motion
from .patch import patch_pictures, pcm_record
from .roundtrip_test import decode_picture, to_bits
from .stream import new_lists
from .system import REPAIR_PEAK, REPAIR_PEAK_EXPLICIT, lower_priority, peak_rss, pool_size


def find_candidates(spss, pics, drop, sigma=None):
    """Original decode indices of kept pictures whose reference context
    changes. Returns (candidates, new lists by orig dec). With `sigma`
    (explicit mode) the new lists are the original ones with every dropped
    reference replaced by its substitute."""
    if sigma is not None:
        dropset = set(drop)
        nl = {p['dec']: [[sigma.get(x, x) for x in l] for l in p['lists']] for p in pics if p['dec'] not in dropset}
    else:
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


def _init(orig_es, new_es, spss, ppss, sigma=None, work=None):
    lower_priority()
    _G.update(orig_es=orig_es, new_es=new_es, spss=spss, ppss=ppss, sigma=sigma, work=work)


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


# Explicit mode (see explicit.py) repairs by measurement: a dropped twin D
# and its substitute K are two encodes of the same frame, so reads of K
# instead of D shift predictions by encode noise. Macroblocks whose decoded
# result moves from the original by DRIFT_THRESHOLD or more (max over Y, U,
# V, visible area) become I_PCM copies of the original, earliest reference
# picture first, until nothing in the IDR period crosses the threshold, with
# at most DRIFT_MAX_FRACTION of any picture patched.
DRIFT_THRESHOLD = 16
DRIFT_MAX_FRACTION = 0.02
DRIFT_MAX_ROUNDS = 12
DRIFT_SEQUENTIAL_ROUNDS = 6   # after these, each round patches every picture still over
PARSED_CAP = 12               # parsed pictures kept in memory per worker


def visible_maxdiff(a, b, sps):
    """Per-MB max absolute difference over Y, U and V, counting only the
    pixels inside the SPS cropping window."""
    W, H = sps['mb_w'], sps['mb_h']
    w, h = W * 16, H * 16
    cl, cr, ct, cb = [2 * c for c in sps.get('crop') or (0, 0, 0, 0)]
    x = np.frombuffer(a, np.uint8)
    y = np.frombuffer(b, np.uint8)
    d = np.maximum(x, y) - np.minimum(x, y)
    ly = d[:w * h].reshape(h, w)
    u = d[w * h:w * h * 5 // 4].reshape(h // 2, w // 2)
    v = d[w * h * 5 // 4:].reshape(h // 2, w // 2)
    for pl, s in ((ly, 1), (u, 2), (v, 2)):
        if cb:
            pl[pl.shape[0] - cb // s:] = 0
        if ct:
            pl[:ct // s] = 0
        if cl:
            pl[:, :cl // s] = 0
        if cr:
            pl[:, pl.shape[1] - cr // s:] = 0
    return np.maximum(ly.reshape(H, 16, W, 16).max(axis=(1, 3)),
                      np.maximum(u.reshape(H, 8, W, 8).max(axis=(1, 3)), v.reshape(H, 8, W, 8).max(axis=(1, 3))))


def _write_segment(mm, seq, repl, path):
    """Write the pictures `seq` (slim pictures in decode order, starting at an
    IDR) with NALs in `repl` substituted."""
    with open(path, 'wb') as o:
        pos = None
        for p in seq:
            sl = p['slices'][0]
            s0 = sl['offset'] - 3
            if mm[s0 - 1] == 0:
                s0 -= 1
            if pos is None:
                pos = p['au_start']
            o.write(mm[pos:s0])
            if p['dec'] in repl:
                o.write(b'\x00\x00\x00\x01' + repl[p['dec']])
            else:
                o.write(mm[s0:sl['offset'] + sl['size']])
            pos = sl['offset'] + sl['size']


class _ParsedLRU(collections.OrderedDict):
    """Parsed pictures for patch_pictures, at most `cap` of them in memory
    (each holds ~25-45 MB of records and motion). Pictures in `reload` (B
    pictures that keep their original motion) are recomputed by reload(nd)
    when evicted; anything else evicted is simply parsed again by
    patch_pictures."""

    def __init__(self, cap, reload_keys, reload):
        super().__init__()
        self.cap, self.reload_keys, self.reload = cap, set(reload_keys), reload

    def __contains__(self, k):
        return super().__contains__(k) or k in self.reload_keys

    def __getitem__(self, k):
        if super().__contains__(k):
            self.move_to_end(k)
            return super().__getitem__(k)
        if k in self.reload_keys:
            v = self.reload(k)
            self[k] = v
            return v
        raise KeyError(k)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.move_to_end(k)
        while len(self) > self.cap:
            self.popitem(last=False)


def _orig_frames(path, orig, periods, fs):
    """(original dec, frame) for the original pictures of `periods`, in
    (period, POC) order, decoding one period at a time."""
    for g in periods:
        sub = sorted((p for p in orig.values() if p['period'] == g), key=lambda p: p['dec'])
        by_poc = sorted(sub, key=lambda p: p['poc'])
        frames = iter_frames(path, [(sub[0]['au_start'], sub[-1]['au_end'])], fs)
        for p, fr in zip(by_poc, frames):
            yield p['dec'], fr


def run_task_explicit(task):
    """Explicit mode: kept pictures keep their reference indices and mvs; only
    reads of a dropped picture D now read its substitute K. B pictures whose
    co-located picture was D keep their original motion: direct blocks whose
    derived motion would change are made explicit. Then the period is decoded
    and compared with the original, and drifting MBs are patched (see
    DRIFT_THRESHOLD).

    Memory stays bounded however many IDR periods anchor rebuilds merged:
    the original is decoded again in lockstep every round, only the target
    frames of patched pictures are kept, and parsed pictures live in an LRU."""
    t0 = time.time()
    spss, ppss, sigma, work = _G['spss'], _G['ppss'], _G['sigma'], _G['work']
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    fs = frame_size(sps)
    orig, new, orig_of_new = task['orig'], task['new'], task['orig_of_new']
    new_of = {v: k for k, v in orig_of_new.items()}
    fo = open(_G['orig_es'], 'rb')
    mo_mm = mmap.mmap(fo.fileno(), 0, access=mmap.ACCESS_READ)
    fn = open(_G['new_es'], 'rb')
    mm = mmap.mmap(fn.fileno(), 0, access=mmap.ACCESS_READ)

    def orig_motion(d, memo):
        if d not in memo:
            po = orig[d]
            dd = decode_picture(mo_mm, spss, ppss, po)
            col = orig_motion(po['lists'][1][0], memo)[1] if po['type'] == 'B' else None
            memo[d] = (dd, derive_motion(dict(po), dd['mbs'], W, H, orig, col))
        return memo[d]

    def keep_original_motion(nd):
        """Records of B picture nd under its new header, with the motion it
        had in the original (its co-located picture there was D)."""
        dd, mo = orig_motion(orig_of_new[nd], {})
        ns = new[nd]['slices'][0]
        nal = bytes(mm[ns['offset']:ns['offset'] + ns['size']])
        bits = to_bits(unescape(nal[1:ns['data_bit'] // 8 + 64]))[:ns['data_bit']]
        return dict(dd, nal=nal, rbsp=None, bits=bits, sl=ns), mo

    force = {new_of[d] for d in new_of
             if orig[d]['type'] == 'B' and sigma.get(orig[d]['lists'][1][0]) is not None}
    parsed = _ParsedLRU(PARSED_CAP, force, keep_original_motion)

    seq = [new[nd] for nd in sorted(new)]
    order = [q['dec'] for q in sorted(seq, key=lambda q: q['poc'])]
    budget = max(1, int(DRIFT_MAX_FRACTION * W * H))
    tmp = os.path.join(work, f'drift_{task["period"]}.h264')
    patches, log_lines, stats, cache, seen, target = {}, [], {}, {}, {}, {}
    repl, capped, rounds, worst, viol = {}, set(), 0, {}, {}
    try:
        while True:
            stats = {}
            repl = patch_pictures(mm, spss, ppss, new, patches, target.__getitem__,
                                  log=log_lines.append, parsed=parsed, stats=stats, force=force, cache=cache)
            _write_segment(mm, seq, repl, tmp)
            viol, keep = {}, {}
            ogen = _orig_frames(_G['orig_es'], orig, task['orig_periods'], fs)
            n_out = 0
            for nd, fr in zip(order, iter_frames(tmp, [(0, os.path.getsize(tmp))], fs)):
                n_out += 1
                want = orig_of_new[nd]
                for d, ofr in ogen:          # the mapping is monotone: skip dropped originals
                    if d == want:
                        break
                else:
                    raise RuntimeError(f'original picture {want} not found in decode order')
                # a frame decoded exactly as in the previous round needs no new diff
                key = hashlib.md5(fr).digest()
                if seen.get(nd, (None,))[0] != key:
                    seen[nd] = (key, visible_maxdiff(fr, ofr, sps))
                m = seen[nd][1]
                done = set(patches.get(nd, ()))
                bad = [(int(m[y, x]), int(x), int(y)) for y, x in zip(*(m >= DRIFT_THRESHOLD).nonzero())
                       if (int(x), int(y)) not in done]
                worst[nd] = int(m.max())
                if bad:
                    viol[nd] = sorted(bad, reverse=True)
                    # keep the original samples of the MBs that may be patched
                    # (not whole frames: a merged period can hold hundreds)
                    room = budget - len(patches.get(nd, ()))
                    keep[nd] = {(x, y): pcm_record(ofr, W, x, y) for _, x, y in viol[nd][:max(room, 0)]}
            for _ in ogen:   # let the original decoder finish cleanly
                pass
            if n_out != len(order):
                raise RuntimeError(f'period {task["period"]}: decoded {n_out} of {len(order)} pictures')
            rounds += 1
            todo = [nd for nd in sorted(viol) if nd not in capped]
            if not todo or rounds > DRIFT_MAX_ROUNDS:
                break
            # early rounds patch in decode order up to the first reference
            # picture (what it fixes downstream is measured again next round);
            # later rounds patch everything still over the threshold
            for nd in todo:
                room = budget - len(patches.get(nd, ()))
                if len(viol[nd]) > room:
                    capped.add(nd)
                mbs = [(x, y) for _, x, y in viol[nd][:max(room, 0)]]
                if mbs:
                    patches[nd] = sorted(set(patches.get(nd, [])) | set(mbs))
                    target.setdefault(nd, {}).update({k: keep[nd][k] for k in mbs})
                if new[nd]['ref_idc'] and rounds <= DRIFT_SEQUENTIAL_ROUNDS:
                    break
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    sniff = []
    for nd in sorted(set(patches) | set(viol) | force):
        d = orig_of_new[nd]
        sniff.append({'dec': d, 'display': orig[d]['disp'], 'type': orig[d]['type'],
                      'mbs': len(patches.get(nd, ())), 'worst_after': worst.get(nd, 0),
                      'over_threshold_after': len(viol.get(nd, ())), 'capped': nd in capped})
    mm.close()
    fn.close()
    mo_mm.close()
    fo.close()
    return {'period': task['period'], 'repl': repl, 'stats': stats, 'sniff': sniff, 'rounds': rounds,
            'patches': {nd: len(v) for nd, v in patches.items()}, 'log': log_lines,
            'seconds': round(time.time() - t0, 1), 'peak_rss': peak_rss()}


def run_task(task):
    if _G.get('sigma') is not None:
        return run_task_explicit(task)
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


def build_tasks_explicit(pics, new_pics, orig_of_new, sigma):
    """Explicit mode: one task per IDR period of the new stream that holds a
    picture reading a substituted picture (anchor rebuilds merge periods, so
    a task can span several original periods)."""
    nper = {}
    for q in new_pics:
        nper.setdefault(q['period'], []).append(q['dec'])
    operiods = {}
    for p in pics:
        operiods.setdefault(p['period'], []).append(p)
    subst = {d for d, k in sigma.items() if k is not None}
    tasks, cands = [], []
    for g, nds in sorted(nper.items()):
        readers = [nd for nd in nds if any(x in subst for l in pics[orig_of_new[nd]]['lists'] for x in l)]
        if not readers:
            continue
        cands += [orig_of_new[nd] for nd in readers]
        og = sorted({pics[orig_of_new[nd]]['period'] for nd in nds})
        tasks.append({'period': g, 'new': {nd: _slim(new_pics[nd]) for nd in nds},
                      'orig_of_new': {nd: orig_of_new[nd] for nd in nds},
                      'orig': {p['dec']: _slim(p) for o in og for p in operiods[o]}, 'orig_periods': og})
    return tasks, cands


def repair(orig_es, new_es, spss, ppss, pics, new_pics, drop, workers, log=print, sigma=None, work=None,
           orig_of_out=None):
    """Returns ({new dec: NAL}, summary dict). Explicit mode (see explicit.py):
    sigma = {dropped reference: substitute or None}, orig_of_out = original
    decode index of every picture of the new stream."""
    if sigma is not None:
        orig_of_new = dict(enumerate(orig_of_out))
        tasks, cands = build_tasks_explicit(pics, new_pics, orig_of_new, sigma)
        log(f'repair: {len(cands)} pictures read a substituted twin, in {len(tasks)} IDR periods')
        peak = REPAIR_PEAK_EXPLICIT
    else:
        new_of = {p['dec']: i for i, p in enumerate(p for p in pics if p['dec'] not in set(drop))}
        orig_of_new = {v: k for k, v in new_of.items()}
        cands, _ = find_candidates(spss, pics, drop)
        tasks = build_tasks(pics, new_pics, new_of, cands)
        log(f'repair: {len(cands)} pictures with a changed reference context in {len(tasks)} IDR periods')
        peak = REPAIR_PEAK
    results = []
    nproc = pool_size(workers, peak, len(tasks)) if tasks else 0
    if tasks:
        log(f'repair: {nproc} workers')
        # workers are recycled every 2 tasks so their memory peak does not creep
        with mp.get_context('spawn').Pool(nproc, initializer=_init, maxtasksperchild=2,
                                          initargs=(orig_es, new_es, spss, ppss, sigma, work)) as pool:
            for i, r in enumerate(pool.imap_unordered(run_task, tasks)):
                results.append(r)
                n_pcm = sum(r['patches'].values())
                rounds = f", {r['rounds']} rounds" if 'rounds' in r else ''
                log(f'  [{i + 1}/{len(tasks)}] IDR period {r["period"]}: {len(r["repl"])} pictures rewritten, '
                    f'{n_pcm} PCM MBs ({r["seconds"]} s{rounds})')
    repl = {}
    summary = {'candidate_pictures': len(cands), 'periods': len(tasks), 'pictures_rewritten': 0,
               'pcm_mbs': 0, 'bytes_added': 0, 'skip_to_explicit': 0, 'direct_to_explicit': 0,
               'qp_carriers': 0, 'workers': nproc,
               'worker_peak_rss_mb': round(max((r['peak_rss'] for r in results), default=0) / 2**20),
               'sniff': [], 'pictures': []}
    if sigma is not None:
        summary.update(drift_threshold=DRIFT_THRESHOLD, drift_max_per_picture=int(
            DRIFT_MAX_FRACTION * spss[0]['mb_w'] * spss[0]['mb_h']),
            seconds=round(sum(r['seconds'] for r in results), 1))
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
            summary['pictures'].append({'out_dec': nd, 'out_display': new_pics[nd]['disp'],
                                        'in_display': pics[orig_of_new[nd]]['disp'], **s})
    if sigma is not None:
        sn = summary['sniff']
        summary.update(pictures_over_threshold_after=sum(1 for x in sn if x['over_threshold_after']),
                       pictures_capped=sum(1 for x in sn if x['capped']),
                       worst_after=max((x['worst_after'] for x in sn), default=0))
    return repl, summary

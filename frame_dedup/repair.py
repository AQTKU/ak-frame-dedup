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


def changed_mbs(m_orig, m_new, orig_of_new, W, H):
    out = set()
    for gy in range(H * 4):
        for gx in range(W * 4):
            for l in range(2):
                ro = m_orig.refpic[l][gy][gx]
                rn = m_new.refpic[l][gy][gx]
                rn = None if rn is None else orig_of_new[rn]
                if ro != rn or (ro is not None and m_orig.mv[l][gy][gx] != m_new.mv[l][gy][gx]):
                    out.add((gx // 4, gy // 4))
                    break
    return out


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
    parsed, m_orig = {}, {}
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
            ch = changed_mbs(mo, mn, orig_of_new, W, H)
            sniff.append({'dec': d, 'display': po['disp'], 'type': t, 'mbs': len(ch)})
            if ch:
                patches[nd] = sorted(ch)
    repl, stats = {}, {}
    if patches:
        # original pixels of the patched pictures, by POC
        last = max(orig_of_new[nd] for nd in patches)
        first = min(orig)
        want = {new[nd]['poc'] for nd in patches}
        frames = decode_pocs(_G['orig_es'], orig, first, last + 1, want, frame_size(sps))
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

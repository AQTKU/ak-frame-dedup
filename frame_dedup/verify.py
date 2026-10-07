"""Decode the output and the original and compare every kept frame.

Frames are matched by (IDR period, POC): dropping pictures never touches POC
and never drops an IDR, so each output picture has exactly one original.
The work is split into groups of IDR periods decoded in parallel.
"""
import multiprocessing as mp

import numpy as np

from .decode import frame_size, iter_frames


def mb_maxdiff(a, b, W, H):
    """Per-macroblock max absolute difference over Y, U and V."""
    x = np.frombuffer(a, np.uint8).astype(np.int16)
    y = np.frombuffer(b, np.uint8).astype(np.int16)
    d = np.abs(x - y).astype(np.uint8)
    w, h = W * 16, H * 16
    ly = d[:w * h].reshape(H, 16, W, 16).max(axis=(1, 3))
    u = d[w * h:w * h * 5 // 4].reshape(H, 8, W, 8).max(axis=(1, 3))
    v = d[w * h * 5 // 4:].reshape(H, 8, W, 8).max(axis=(1, 3))
    return np.maximum(ly, np.maximum(u, v))


def _job(args):
    orig_es, out_es, orange, nrange, okeys, nkeys, W, H, fs, disp_of = args
    nset = set(nkeys)
    res = {'compared': 0, 'exact': 0, 'differing': [], 'missing': 0, 'errors': []}
    try:
        it_o = iter_frames(orig_es, [orange], fs)
        it_n = iter_frames(out_es, [nrange], fs)
        nk = iter(nkeys)
        for key in okeys:
            fo = next(it_o)
            if key not in nset:
                continue
            kn = next(nk)
            fn = next(it_n)
            assert kn == key, (kn, key)
            res['compared'] += 1
            if fo == fn:
                res['exact'] += 1
                continue
            m = mb_maxdiff(fo, fn, W, H)
            res['differing'].append({'display': disp_of[key], 'period': key[0], 'poc': key[1],
                                     'worst': int(m.max()), 'mbs_ge3': int((m >= 3).sum()),
                                     'mbs_diff': int((m > 0).sum())})
        for _ in it_n:
            res['errors'].append('output has extra frames')
            break
        for _ in it_o:
            pass
    except (RuntimeError, StopIteration, AssertionError) as e:
        res['errors'].append(f'{type(e).__name__}: {e}')
    return res


def verify(orig_es, out_es, spss, orig_pics, out_pics, workers, log=print, group=600):
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    fs = frame_size(sps)

    def periods(pics):
        out = {}
        for p in pics:
            out.setdefault(p['period'], []).append(p)
        return out
    po, pn = periods(orig_pics), periods(out_pics)
    if sorted(po) != sorted(pn):
        raise RuntimeError('IDR periods differ between original and output')
    jobs, cur = [], []

    def flush():
        if not cur:
            return
        a, b = cur[0], cur[-1]
        okeys = sorted((p['period'], p['poc']) for g in cur for p in po[g])
        nkeys = sorted((p['period'], p['poc']) for g in cur for p in pn[g])
        disp_of = {(p['period'], p['poc']): p['disp'] for g in cur for p in po[g]}
        jobs.append((orig_es, out_es, (po[a][0]['au_start'], po[b][-1]['au_end']),
                     (pn[a][0]['au_start'], pn[b][-1]['au_end']), okeys, nkeys, W, H, fs, disp_of))
        cur.clear()
    n = 0
    for g in sorted(po):
        cur.append(g)
        n += len(po[g])
        if n >= group:
            flush()
            n = 0
    flush()
    tot = {'compared': 0, 'exact': 0, 'differing': [], 'errors': []}
    with mp.get_context('spawn').Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            for k in ('compared', 'exact'):
                tot[k] += r[k]
            tot['differing'] += r['differing']
            tot['errors'] += r['errors']
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                log(f'  verify: {i + 1}/{len(jobs)} groups, {tot["compared"]} frames compared, '
                    f'{len(tot["differing"])} differ')
    tot['differing'].sort(key=lambda d: d['display'])
    diff = tot['differing']
    return {
        'frames_output': len(out_pics), 'frames_compared': tot['compared'], 'bit_exact': tot['exact'],
        'differing': len(diff), 'worst_difference': max((d['worst'] for d in diff), default=0),
        'mbs_ge3': sum(d['mbs_ge3'] for d in diff), 'mbs_differing': sum(d['mbs_diff'] for d in diff),
        'errors': tot['errors'], 'differing_frames': diff,
    }

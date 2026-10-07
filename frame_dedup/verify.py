"""Decode the output and the original and compare every kept frame.

Frames are matched by (IDR period, POC) of the original: dropping pictures
never touches POC and never drops an IDR, so each output picture has exactly
one original. Anchor rebuilds merge IDR periods and shift POCs; for them the
caller passes the original of every output picture. The work is split into
groups of output IDR periods (each covering whole original periods) decoded
in parallel.
"""
import hashlib
import multiprocessing as mp

import numpy as np

from .decode import frame_size, iter_frames
from .system import VERIFY_PEAK, peak_rss, pool_size, worker_init


def mb_maxdiff(a, b, W, H):
    """Per-macroblock max absolute difference over Y, U and V."""
    x = np.frombuffer(a, np.uint8).astype(np.int16)
    y = np.frombuffer(b, np.uint8).astype(np.int16)
    d = np.abs(x - y).astype(np.uint8)
    w, h = W * 16, H * 16
    ly = d[:w * h].reshape(H, 16, W, 16).max(axis=(1, 3))
    u = d[w * h:w * h * 5 // 4].reshape(H, 8, W, 8).max(axis=(1, 3))
    v = d[w * h * 5 // 4:].reshape(H, 8, W, 8).max(axis=(1, 3))
    return np.maximum(ly, np.maximum(u, v)), ly


def _job(args):
    """Compare one group of IDR periods. Only one ffmpeg decoder runs at a
    time: the original is decoded and hashed, then the output is decoded and
    compared by hash; frames that differ are decoded from the original again
    (kept only for those POCs) to measure per-macroblock differences."""
    orig_es, out_es, orange, nrange, okeys, nkeys, W, H, fs, disp_of = args
    nset = set(nkeys)
    res = {'compared': 0, 'exact': 0, 'differing': [], 'errors': []}
    try:
        ohash = {}
        n = 0
        for n, fo in enumerate(iter_frames(orig_es, [orange], fs), 1):
            if n > len(okeys):
                raise RuntimeError('original group decoded too many frames')
            if okeys[n - 1] in nset:
                ohash[okeys[n - 1]] = hashlib.blake2b(fo, digest_size=16).digest()
        if n != len(okeys):
            raise RuntimeError(f'original group decoded {n} frames, expected {len(okeys)}')
        bad = {}
        n = 0
        for n, fn in enumerate(iter_frames(out_es, [nrange], fs), 1):
            if n > len(nkeys):
                raise RuntimeError('output group decoded too many frames')
            key = nkeys[n - 1]
            res['compared'] += 1
            if hashlib.blake2b(fn, digest_size=16).digest() == ohash[key]:
                res['exact'] += 1
            else:
                bad[key] = fn
        if n != len(nkeys):
            raise RuntimeError(f'output group decoded {n} frames, expected {len(nkeys)}')
        if bad:
            for key, fo in zip(okeys, iter_frames(orig_es, [orange], fs)):
                if not bad:
                    break
                if key in bad:
                    m, ly = mb_maxdiff(fo, bad.pop(key), W, H)
                    res['differing'].append({'display': disp_of[key], 'period': key[0], 'poc': key[1],
                                             'worst': int(m.max()), 'mbs_ge3': int((m >= 3).sum()),
                                             'mbs_diff': int((m > 0).sum()), 'worst_luma': int(ly.max()),
                                             'mbs_ge3_luma': int((ly >= 3).sum())})
    except (RuntimeError, KeyError) as e:
        res['errors'].append(f'{type(e).__name__}: {e}')
    res['peak_rss'] = peak_rss()
    return res


def verify(orig_es, out_es, spss, orig_pics, out_pics, workers, log=print, group=300, orig_of_out=None):
    """orig_of_out: original decode index of every output picture (output
    decode order). Without it, output pictures are matched to originals by
    (IDR period, POC), which holds when pictures were only dropped. Anchor
    rebuilds merge IDR periods and shift POCs, so the pipeline passes it."""
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    fs = frame_size(sps)

    def periods(pics):
        out = {}
        for p in pics:
            out.setdefault(p['period'], []).append(p)
        return out
    po, pn = periods(orig_pics), periods(out_pics)
    if orig_of_out is None:
        key_of = {q['dec']: (q['period'], q['poc']) for q in out_pics}
    else:
        assert len(orig_of_out) == len(out_pics)
        key_of = {q['dec']: (orig_pics[o]['period'], orig_pics[o]['poc']) for q, o in zip(out_pics, orig_of_out)}
    # each output IDR period covers whole original IDR periods
    cover = {g: sorted({key_of[q['dec']][0] for q in pn[g]}) for g in pn}
    flat = [x for g in sorted(pn) for x in cover[g]]
    if flat != sorted(po) or any(c != list(range(c[0], c[-1] + 1)) for c in cover.values()):
        raise RuntimeError('output IDR periods do not map onto whole original IDR periods')
    jobs, cur = [], []

    def flush():
        if not cur:
            return
        og = [x for g in cur for x in cover[g]]
        a, b = og[0], og[-1]
        okeys = sorted((p['period'], p['poc']) for g in og for p in po[g])
        nkeys = [key_of[q['dec']] for g in cur for q in sorted(pn[g], key=lambda q: q['poc'])]
        disp_of = {(p['period'], p['poc']): p['disp'] for g in og for p in po[g]}
        jobs.append((orig_es, out_es, (po[a][0]['au_start'], po[b][-1]['au_end']),
                     (pn[cur[0]][0]['au_start'], pn[cur[-1]][-1]['au_end']), okeys, nkeys, W, H, fs, disp_of))
        cur.clear()
    n = 0
    for g in sorted(pn):
        cur.append(g)
        n += len(pn[g])
        if n >= group:
            flush()
            n = 0
    flush()
    tot = {'compared': 0, 'exact': 0, 'differing': [], 'errors': []}
    # each verify worker keeps a 2-thread ffmpeg decoder busy, so use half
    # the worker budget to stay near the same CPU load as the other stages
    nproc = pool_size(max(1, workers // 2), VERIFY_PEAK, len(jobs))
    log(f'  verify: {len(jobs)} groups on {nproc} workers (one decoder each)')
    peak = 0
    with mp.get_context('spawn').Pool(nproc, initializer=worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs)):
            for k in ('compared', 'exact'):
                tot[k] += r[k]
            tot['differing'] += r['differing']
            tot['errors'] += r['errors']
            peak = max(peak, r['peak_rss'])
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                log(f'  verify: {i + 1}/{len(jobs)} groups, {tot["compared"]} frames compared, '
                    f'{len(tot["differing"])} differ')
    tot['differing'].sort(key=lambda d: d['display'])
    diff = tot['differing']
    return {
        'frames_output': len(out_pics), 'frames_compared': tot['compared'], 'bit_exact': tot['exact'],
        'differing': len(diff), 'worst_difference': max((d['worst'] for d in diff), default=0),
        'mbs_ge3': sum(d['mbs_ge3'] for d in diff), 'mbs_differing': sum(d['mbs_diff'] for d in diff),
        'worst_luma': max((d['worst_luma'] for d in diff), default=0),
        'mbs_ge3_luma': sum(d['mbs_ge3_luma'] for d in diff),
        'errors': tot['errors'], 'workers': nproc, 'worker_peak_rss_mb': round(peak / 2**20),
        'differing_frames': diff,
    }

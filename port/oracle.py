"""Python-side dumpers for checking the Rust port against frame_dedup v1.2.0.

Usage: python -m port.oracle <command> [args...]

Each command prints canonical JSON (port/canon.py), usually one object per
line, matching the Rust `frame-dedup dev <command>` of the same name.
"""
import json
import os
import sys

COMMANDS = {}

ORACLE_DIR = os.environ.get('PORT_ORACLE_DIR', 'C:/Users/aquat/Documents/GitHub/ak-frame-dedup/work/port/oracle')


def command(fn):
    COMMANDS[fn.__name__.replace('_', '-')] = fn
    return fn


@command
def headers(path):
    """SPSs and PPSs on the first line, then one line per picture with its first
    slice header (stream.load: dpb_walk, display order, AU ranges)."""
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = load(path)
    ppss = {k: dict(v) for k, v in ppss.items()}
    for p in ppss.values():
        p.setdefault('scaling_matrix', 0)  # the Rust struct cannot tell absent from 0
    print(text({'spss': {str(k): v for k, v in spss.items()}, 'ppss': {str(k): v for k, v in ppss.items()}}))
    for p in pics:
        print(text({'dec': p['dec'], 'poc': p['poc'], 'frame_num': p['frame_num'], 'lists': p['lists'],
                    'dpb_after': p['dpb_after'], 'period': p['period'], 'disp': p['disp'],
                    'au_start': p['au_start'], 'au_end': p['au_end'], 'idr': p['idr'],
                    'ref_idc': p['ref_idc'], 'type': p['type'], 'sl': p['slices'][0]}))


@command
def escape(path, n):
    """For the first n slice NALs: unescape / escape results as [length, crc32]."""
    import zlib
    from frame_dedup.h264hdr import escape as esc, iter_nals, unescape as unesc
    from .canon import text
    count = 0
    for s, e, mm in iter_nals(path):
        if mm[s] & 0x1f not in (1, 5):
            continue
        if count >= int(n):
            break
        count += 1
        body = bytes(mm[s + 1:e])
        rbsp = unesc(body)
        enc = esc(rbsp)
        raw_esc = esc(body)
        raw_unesc = unesc(enc)
        print(text({'offset': s, 'unesc': [len(rbsp), zlib.crc32(rbsp)], 'esc': [len(enc), zlib.crc32(enc)],
                    'same': enc == body, 'esc_raw': [len(raw_esc), zlib.crc32(raw_esc)],
                    'unesc_raw': [len(raw_unesc), zlib.crc32(raw_unesc)]}))


@command
def patch_sps(path):
    """Every SPS NAL, patched to 1001/48000 and to 1/48 (hex)."""
    from frame_dedup.h264hdr import iter_nals, patch_sps_timing
    from .canon import text
    for s, e, mm in iter_nals(path):
        if mm[s] & 0x1f != 7:
            continue
        nal = bytes(mm[s:e])
        print(text({'offset': s, 'nal': nal.hex(), 'p1001': patch_sps_timing(nal, 1001, 48000).hex(),
                    'p1_48': patch_sps_timing(nal, 1, 48).hex()}))


@command
def features(path, threads='2'):
    """features <file.h264|features.npz> [threads]: the four proxy-feature arrays
    (mad, bmax, mean, max). An .npz is read; a stream is decoded with ffmpeg."""
    import numpy as np
    from frame_dedup.evidence import proxy_features
    from . import canon
    if path.endswith('.npz'):
        f = {k: np.load(path)[k] for k in ('mad', 'bmax', 'mean', 'max')}
    else:
        f = proxy_features(path, threads=int(threads))
    print(canon.text({k: v.tolist() for k, v in f.items()}))


@command
def features_raw(path):
    """features-raw <file.gray>: proxy_features on raw 480x270 gray frames instead of an ffmpeg pipe."""
    import io
    from unittest import mock
    from frame_dedup import evidence
    from . import canon

    class FakeProc:
        returncode = 0

        def __init__(self, *a, **k):
            self.stdout = open(path, 'rb')
            self.stderr = io.BytesIO()

        def wait(self):
            pass
    with mock.patch.object(evidence.subprocess, 'Popen', FakeProc):
        f = evidence.proxy_features('unused')
    print(canon.text({k: v.tolist() for k, v in f.items()}))


@command
def features_npz(path, out, threads='4'):
    """features-npz <file.h264> <out.npz> [threads]: decode once and cache (as the pipeline's features.npz)."""
    import numpy as np
    from frame_dedup.evidence import proxy_features
    f = proxy_features(path, threads=int(threads))
    np.savez(out, es_size=os.path.getsize(path), **f)
    print(f'{len(f["mad"])} frames -> {out}')


def _load_features(path):
    import numpy as np
    if path.endswith('.npz'):
        z = np.load(path)
        return {k: np.array(z[k], dtype=np.float64) for k in ('mad', 'bmax', 'mean', 'max')}
    with open(path) as fh:
        return {k: np.array(v, dtype=np.float64) for k, v in json.load(fh).items()}


def _fr(s):
    from fractions import Fraction
    return Fraction(s)


@command
def detect(path, fps):
    """detect <features.json|.npz> <fps num/den>: what Run.analyse computes (evidence, static
    mask, black runs, cuts, segments, candidates, predicted dups, score) and its report sections."""
    import numpy as np
    from frame_dedup.cadence import cadence_score, detect as detect_, hard_cuts, predicted_dups, segments_from
    from frame_dedup.evidence import black_runs, evidence, load_model, static_mask
    from . import canon
    feat = _load_features(path)
    fps_in = _fr(fps)
    N = len(feat['mad'])
    model, meta = load_model()
    e = evidence(feat, model)
    static = static_mask(feat['bmax'])
    black = black_runs(feat, float(fps_in))
    cuts = hard_cuts(feat, static)
    segs = segments_from(N, black, cuts)
    cands = detect_(e, segs, fps_in)
    best = cands[0]
    dup = predicted_dups(N, segs, best)
    score, n_ev = cadence_score(e, static, dup)
    # --- the report sections, as pipeline.Run.analyse builds them
    det = {'model': meta.get('trained_on'), 'black_runs': [
        {'start': s, 'end': x, 'start_s': round(s / float(fps_in), 3), 'frames': x - s} for s, x in black],
        'hard_cuts': len(cuts), 'segments': len(segs), 'static_frames': int(static.sum()),
        'candidates': [], 'cadence_score': round(score, 3), 'fail_threshold': 2.0,
        'duplicates_with_evidence': n_ev}
    for c in cands:
        cd = predicted_dups(N, segs, c)
        det['candidates'].append({'source': c['source'], 'ratio': str(c['r']), 'total_llr': round(c['total'], 1),
                                  'cadence_score': round(cadence_score(e, static, cd)[0], 3),
                                  'phase_changes': sum(1 for i in range(1, len(segs))
                                                       if c['phases'][i] != c['phases'][i - 1])})
    r, fps_out = best['r'], best['rate']
    chunks, changes = [], []
    for i, ((s, x, kind), m) in enumerate(zip(segs, best['phases'])):
        if kind in ('start', 'black') or not chunks:
            chunks.append({'start': s, 'end': x, 'start_s': round(s / float(fps_in), 3), 'phases': [m]})
        else:
            chunks[-1]['end'] = x
            if m != chunks[-1]['phases'][-1]:
                chunks[-1]['phases'].append(m)
        if i and m != best['phases'][i - 1]:
            changes.append({'display': s, 'time_s': round(s / float(fps_in), 3), 'at': kind,
                            'from_phase': best['phases'][i - 1], 'to_phase': m, 'q': best['q']})
    for ch in chunks:
        ch['frames'] = ch['end'] - ch['start']
        ch['phi'] = [round(m / best['q'], 6) for m in ch['phases']]
    cad = {'source_rate': best['source'], 'output_fps': str(fps_out), 'ratio': str(r),
           'phase_bins': best['q'], 'chunks': chunks, 'phase_changes': changes}
    print(canon.text({
        'evidence': e.tolist(),
        'static': [int(i) for i in np.nonzero(static)[0]],
        'black': [list(b) for b in black],
        'cuts': cuts,
        'segments': [list(s) for s in segs],
        'candidates': [{'source': c['source'], 'rate': str(c['rate']), 'r': str(c['r']), 'p': c['p'], 'q': c['q'],
                        'total': c['total'], 'phases': c['phases'], 'segment_scores': c['segment_scores']}
                       for c in cands],
        'predicted_dups': [int(i) for i in np.nonzero(dup)[0]],
        'cadence_score': [score, n_ev],
        'detection': det,
        'cadence': cad,
    }))


def _jsonable(v):
    """numpy scalars / arrays and sets to plain JSON types."""
    import numpy as np
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    raise TypeError(type(v))


def _canon_dump(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), default=_jsonable)


@command
def plan_inputs(name, h264, features_npz, out, *flags):
    """plan-inputs <name> <file.h264> <features.npz> <out.json> [--no-rebuild] [--keep-idr-pairs]:
    run the real Run.analyse + Run.plan (stream load cached as <name>.load.pkl, features read, never
    decoded) while recording every free(dec) / rebuildable(K, D) answer. Writes the inputs file for
    `plan` to <out.json> and the report sections Python produced to <out>.expected.json
    (same shape as `frame-dedup dev plan` prints)."""
    import pickle
    import shutil
    import types
    import numpy as np
    from frame_dedup import explicit as X, pipeline, stream
    work = os.path.join(ORACLE_DIR, f'{name}_work')
    os.makedirs(work, exist_ok=True)
    es = os.path.join(work, 'in.h264')
    if not os.path.exists(es):
        os.link(h264, es)
    ft = os.path.join(work, 'features.npz')
    if not os.path.exists(ft):
        shutil.copy(features_npz, ft)
    pkl = os.path.join(ORACLE_DIR, f'{name}.load.pkl')
    orig_load = stream.load

    def cached_load(path):
        if os.path.exists(pkl):
            with open(pkl, 'rb') as fh:
                return pickle.load(fh)
        r = orig_load(path)
        with open(pkl, 'wb') as fh:
            pickle.dump(r, fh, protocol=5)
        return r

    def no_decode(*a, **k):
        raise RuntimeError('features.npz does not match the stream: refusing to decode')
    rec_free, rec_rb = {}, {}

    def rec_free_checker(orig):
        def make(*a):
            f = orig(*a)

            def g(d):
                v = f(d)
                rec_free[d] = bool(v)
                return v
            return g
        return make

    def rec_candidate(orig):
        def g(spss, pics, periods, K, D):
            rb, why = orig(spss, pics, periods, K, D)
            rec_rb[f'{K},{D}'] = [list(rb['B']) if rb else None, why]
            return rb, why
        return g
    pipeline.extract_annexb = lambda *a, **k: None
    pipeline.load = cached_load
    pipeline.proxy_features = no_decode
    pipeline.free_drop_checker = rec_free_checker(pipeline.free_drop_checker)
    X.free_checker = rec_free_checker(X.free_checker)
    pipeline.candidate = rec_candidate(pipeline.candidate)
    X.candidate = rec_candidate(X.candidate)
    args = types.SimpleNamespace(work=work, input=h264, output=None, workers=2,
                                 no_rebuild='--no-rebuild' in flags, keep_idr_pairs='--keep-idr-pairs' in flags)
    R = pipeline.Run(args)
    R.analyse()
    R.plan()
    rep = R.report
    ch = rep['cadence']['chunks']
    inputs = {
        'disp': [{'dec': p['dec'], 'idr': bool(p['idr']), 'ref_idc': p['ref_idc'], 'type': p['type']}
                 for p in R.disp],
        'dup': [int(i) for i in np.nonzero(R.dup)[0]],
        'free': {str(k): v for k, v in rec_free.items()},
        'has_free': True,
        'rebuildable': None if args.no_rebuild else rec_rb,
        'mad': R.feat['mad'].tolist(), 'bmax': R.feat['bmax'].tolist(),
        'visible_subs': not args.keep_idr_pairs, 'explicit': bool(R.explicit),
        'black': [list(b) for b in R.black],
        'r': [R.best['r'].numerator, R.best['r'].denominator],
        'fps_out': [R.fps_out.numerator, R.fps_out.denominator],
        'cadence': {'chunks': [[c['start'], c['end']] for c in ch],
                    'phase_changes': [[c['display'], c['at']] for c in rep['cadence']['phase_changes']]},
    }
    expected = {'drop': sorted(R.drop),
                'report': {k: rep[k] for k in ('drops', 'unreferenced_reference_drops', 'substitute_drops',
                                               'hard_pairs', 'anchor_rebuild_fallbacks', 'unsupported_pairs',
                                               'sync')},
                'hard': R.hard, 'rebuilds': R.rebuilds, 'freed': R.freed}
    with open(out, 'w') as fh:
        fh.write(_canon_dump(inputs))
    with open(out + '.expected.json', 'w') as fh:
        fh.write(_canon_dump(expected))
    with open(out + '.reportparts.json', 'w') as fh:   # detection / cadence as the pipeline reported them
        fh.write(_canon_dump({'detection': rep['detection'], 'cadence': rep['cadence']}))
    print(f'{len(R.disp)} pictures, {len(R.drop)} drops, {len(rec_free)} free() and {len(rec_rb)} '
          f'rebuildable() answers recorded -> {out}')


@command
def plan(path):
    """plan <inputs.json>: the Python plan functions on a recorded inputs file (replaying the free /
    rebuildable answers), printing the same JSON as `frame-dedup dev plan`."""
    import numpy as np
    from fractions import Fraction
    from frame_dedup.plan import choose_drops, offsets, sync_correct, sync_stats
    with open(path) as fh:
        inp = json.load(fh)
    disp = [dict(p) for p in inp['disp']]
    for p in disp:
        p['idr'] = int(p['idr'])
    N = len(disp)
    dup = np.zeros(N, bool)
    dup[inp['dup']] = True
    ftab = {int(k): v for k, v in inp['free'].items()}
    rtab = inp['rebuildable']
    rebuildable = None
    if rtab is not None:
        def rebuildable(K, D):
            B, why = rtab[f'{K},{D}']
            return ({'B': B} if B is not None else None), why
    mad, bmax = np.array(inp['mad']), np.array(inp['bmax'])
    r, fps_out = Fraction(*inp['r']), Fraction(*inp['fps_out'])
    black = [tuple(b) for b in inp['black']]
    drop, hard, unsupported, freed, subs, rebuilds = choose_drops(
        disp, dup, ftab.__getitem__, mad, bmax, visible_subs=inp['visible_subs'], rebuildable=rebuildable,
        explicit=inp['explicit'])
    fallbacks = [x for x in rebuilds if 'fallback' in x]
    rebuilds = [x for x in rebuilds if 'fallback' not in x]
    n_planned = len(drop)
    n_nonref = len(drop) - len(hard) - len(freed) - len(subs) - len(rebuilds)
    events = [d for d, at in inp['cadence']['phase_changes'] if at == 'cut']
    events += [u['display'] for u in unsupported]
    drop, corrections = sync_correct(disp, drop, r, black, events)
    hard_f = [h for h in hard if h['dropped_display'] in drop]
    rebuilds_f = [x for x in rebuilds if x['dropped_display'] in drop and x['kept_display'] not in drop]
    freed_f = [f for f in freed if f['display'] in drop]
    st = sync_stats(N, drop, r, float(fps_out))
    off = offsets(N, drop, r)
    chunk_off = []
    for s, e in inp['cadence']['chunks']:
        seg = off[s:e]
        if np.isfinite(seg).any():
            chunk_off.append({'start': s, 'mean_offset_frames': round(float(np.nanmean(seg)), 3),
                              'max_abs_offset_frames': round(float(np.nanmax(np.abs(seg))), 3)})
    st['chunks'] = chunk_off
    report = {'drops': {'total': len(drop), 'duplicates': int(dup.sum()), 'non_reference': n_nonref,
                        'unreferenced_reference': len(freed),
                        'substitute_invisible': sum(1 for x in subs if not x['visible']),
                        'substitute_visible': sum(1 for x in subs if x['visible']),
                        'hard': len(hard_f), 'anchor_rebuild': len(rebuilds_f),
                        'anchor_rebuild_fallbacks': len(fallbacks), 'sync_net_change': len(drop) - n_planned},
              'unreferenced_reference_drops': freed, 'substitute_drops': subs, 'hard_pairs': hard_f,
              'anchor_rebuild_fallbacks': fallbacks, 'unsupported_pairs': unsupported,
              'sync': {'corrections': corrections, **st}}
    print(_canon_dump({'drop': sorted(drop), 'report': report, 'hard': hard_f, 'rebuilds': rebuilds_f,
                       'freed': freed_f}))


@command
def train_llr(path, repeats):
    """train-llr <features.json|.npz> <repeats.json>: evidence.train_llr on both features."""
    import numpy as np
    from frame_dedup.evidence import rel_diff, train_llr as train
    from . import canon
    feat = _load_features(path)
    with open(repeats) as fh:
        reps = json.load(fh)
    out = {}
    for k in ('mad', 'bmax'):
        bins, llr = train(rel_diff(feat[k]), reps)
        out[k] = [bins.tolist(), llr.tolist()]
    print(canon.text(out))


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f'usage: python -m port.oracle <{"|".join(sorted(COMMANDS))}> [args...]')
    COMMANDS[argv[0]](*argv[1:])


@command
def framemd5(path, *rest):
    """framemd5 <file.h264> [start end] [--take N] [--size WxH]: MD5 of every decoded frame."""
    import hashlib
    import os
    from frame_dedup.decode import iter_frames
    pos, take, size, it = [], None, (1920, 1088), iter(rest)
    for a in it:
        if a == '--take':
            take = int(next(it))
        elif a == '--size':
            size = tuple(int(x) for x in next(it).split('x'))
        else:
            pos.append(a)
    rng = (int(pos[0]), int(pos[1])) if len(pos) >= 2 else (0, os.path.getsize(path))
    fs = size[0] * size[1] * 3 // 2
    for n, fr in enumerate(iter_frames(path, [rng], fs)):
        if take is not None and n >= take:
            break
        print(n, hashlib.md5(fr).hexdigest())


@command
def mux_cmd(in_mkv, es, out_mkv, num, den):
    """mux-cmd <in.mkv> <es> <out.mkv> <num> <den>: the mkvmerge command line (not run)."""
    import subprocess
    from fractions import Fraction
    from frame_dedup import mux as m
    real_run, seen = subprocess.run, []

    def fake(cmd, *a, **k):
        if cmd[:2] == ['mkvmerge', '-J']:
            return real_run(cmd, *a, **k)
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '', '')
    m.subprocess.run = fake
    m.mux(in_mkv, es, out_mkv, Fraction(int(num), int(den)), log=lambda s: None)
    print(json.dumps(seen[0], separators=(',', ':')))


@command
def mux(in_mkv, es, out_mkv, num, den):
    """mux <in.mkv> <es> <out.mkv> <num> <den>: really mux; prints the log line and the messages."""
    from fractions import Fraction
    from frame_dedup import mux as m
    msgs = m.mux(in_mkv, es, out_mkv, Fraction(int(num), int(den)), log=print)
    print(json.dumps(msgs, separators=(',', ':')))


@command
def duration_check(in_mkv, out_mkv, n_out, num, den):
    """duration-check <in.mkv> <out.mkv> <n_out> <num> <den>."""
    from fractions import Fraction
    from frame_dedup.mux import duration_check as dc
    from .canon import text
    print(text(dc(in_mkv, out_mkv, int(n_out), Fraction(int(num), int(den)))))


@command
def maxdiff(path, mb_w, mb_h, cl, cr, ct, cb, i, j):
    """maxdiff <file.h264> <mb_w> <mb_h> <cl> <cr> <ct> <cb> <i> <j>: mb_maxdiff / visible_maxdiff maps."""
    import os
    from frame_dedup.decode import iter_frames
    from frame_dedup.repair import visible_maxdiff
    from frame_dedup.verify import mb_maxdiff
    from .canon import text
    W, H, i, j = int(mb_w), int(mb_h), int(i), int(j)
    frames = []
    for k, f in enumerate(iter_frames(path, [(0, os.path.getsize(path))], W * H * 384)):
        frames.append(f)
        if k == max(i, j):
            break
    m, ly = mb_maxdiff(frames[i], frames[j], W, H)
    vis = visible_maxdiff(frames[i], frames[j], {'mb_w': W, 'mb_h': H, 'crop': tuple(int(x) for x in (cl, cr, ct, cb))})
    print(text({'mb': m.ravel().tolist(), 'luma': ly.ravel().tolist(), 'visible': vis.ravel().tolist()}))


def _picture_args(rest):
    """[--from N] [--count N] [--full] after the file name."""
    opts, it = {'from': 0, 'count': None, 'full': False}, iter(rest)
    pos = []
    for a in it:
        if a == '--from':
            opts['from'] = int(next(it))
        elif a == '--count':
            opts['count'] = int(next(it))
        elif a == '--full':
            opts['full'] = True
        elif a == '--threads':
            next(it)  # accepted for symmetry with the Rust command
        else:
            pos.append(a)
    return opts, pos


def _open_pictures(path, opts):
    import mmap
    from frame_dedup.stream import load
    spss, ppss, pics = load(path)
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    a = min(opts['from'], len(pics))
    b = len(pics) if opts['count'] is None else min(a + opts['count'], len(pics))
    return mm, spss, ppss, pics, range(a, b)


@command
def records(path, *rest):
    """records <file.h264> [--from N] [--count N] [--full]: per picture, the digest of the
    canonical macroblock records (CABAC parse)."""
    import sys
    import time
    from frame_dedup.roundtrip_test import decode_picture
    from .canon import canon_mbs, digest, text
    opts, _ = _picture_args(rest)
    mm, spss, ppss, pics, rng = _open_pictures(path, opts)
    t0 = time.time()
    for i in rng:
        p = pics[i]
        d = decode_picture(mm, spss, ppss, p)
        c = canon_mbs(d['mbs'])
        line = {'dec': p['dec'], 'type': p['type'], 'n_mbs': len(c), 'digest': digest(c)}
        if opts['full']:
            line['mbs'] = c
        print(text(line), flush=True)
    print(f'records: {len(rng)} pictures in {time.time() - t0:.2f}s', file=sys.stderr)


@command
def roundtrip(path, *rest):
    """roundtrip <file.h264> [--from N] [--count N]: roundtrip_test.check for each picture."""
    import sys
    import time
    from frame_dedup.roundtrip_test import decode_picture, encode_picture
    from .canon import text
    opts, _ = _picture_args(rest)
    mm, spss, ppss, pics, rng = _open_pictures(path, opts)
    ok_n, bad, dec_s, enc_s = 0, [], 0.0, 0.0
    for i in rng:
        p = pics[i]
        t0 = time.time()
        d = decode_picture(mm, spss, ppss, p)
        t1 = time.time()
        sl = d['sl']
        tail = d['bits'][d['end']:]
        stop_ok = d['bits'][d['end'] - 1] == 1 and not any(tail)
        late_stop = (not stop_ok) and d['end'] < len(d['bits']) and d['bits'][d['end']] == 1 and not any(d['bits'][d['end'] + 1:])
        nal2, enc_bits = encode_picture(d)
        t2 = time.time()
        same_bits = enc_bits == d['bits'][sl['data_bit']:d['end']]
        same_nal = nal2 == d['nal']
        dec_s += t1 - t0
        enc_s += t2 - t1
        if stop_ok and same_bits and same_nal:
            ok_n += 1
        else:
            bad.append(p['dec'])
        print(text({'dec': p['dec'], 'type': p['type'], 'nal_len': len(d['nal']), 'stop_bit_ok': stop_ok,
                    'tail_bits': len(tail), 'bits_equal': same_bits, 'nal_equal': same_nal,
                    'late_stop_bit': late_stop}), flush=True)
    print(text({'summary': True, 'pictures': len(rng), 'ok': ok_n, 'failed': len(bad), 'failed_dec': bad}))
    n = max(len(rng), 1)
    print(f'roundtrip: mean decode {dec_s / n * 1e3:.1f} ms, mean encode {enc_s / n * 1e3:.1f} ms', file=sys.stderr)


class SplitMix64:
    M = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self.M

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & self.M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.M
        return z ^ (z >> 31)


def apply_edit(mbs, t, w, seed):
    """The deterministic edit spec of `encode-edited` (same as picture.rs `apply_edit`)."""
    from frame_dedup.patch import to_explicit
    rng = SplitMix64(seed)
    period = 6 + seed % 29
    counts = [0, 0, 0, 0]
    for i, mb in enumerate(mbs):
        r = rng.next()
        x, y = i % w, i // w
        if r % period == 0:
            mbs[i] = {'skip': 0, 'kind': 'PCM', 'samples': [(x * 31 + y * 17 + j * 7 + seed) & 255 for j in range(384)],
                      'intra': 1, 'pcm': 1, 'i16': 0, 'direct16': 0, 'cbp': 0x2f, 't8': 0, 'cmode': 0, 'qpd': 0}
            counts[0] += 1
        elif t == 'P' and mb['skip'] and (r >> 20) % 6 == 0:
            mbs[i] = to_explicit(mb, 'P', None)
            counts[1] += 1
        elif t == 'B' and mb['skip'] and (r >> 20) % 6 == 0:
            mb.update(skip=0, kind='B', btype=0, part='D', direct16=1, cbp=0, t8=0, qpd=0, coef={})
            counts[1] += 1
        elif not mb['skip'] and mb['cbp'] == 0 and mb['kind'] != 'I16' and (r >> 32) % 4 == 0:
            mb['cbp'] |= 0x10
            mb.setdefault('coef', {})
            mb['qpd'] = ((r >> 40) % 5) - 2
            counts[2] += 1
        elif not mb['skip'] and (mb['cbp'] != 0 or mb['kind'] == 'I16') and (r >> 48) % 9 == 0:
            mb['qpd'] = ((r >> 52) % 7) - 3
            counts[3] += 1
    return counts


@command
def encode_edited(path, *pairs):
    """encode-edited <file.h264> <dec> <seed> [<dec> <seed> ...]: decode picture dec, apply the
    deterministic edit, encode; one line per (dec, seed)."""
    import hashlib
    from frame_dedup.roundtrip_test import decode_picture, encode_picture
    from .canon import text
    mm, spss, ppss, pics, _ = _open_pictures(path, {'from': 0, 'count': None})
    for k in range(0, len(pairs), 2):
        dec, seed = int(pairs[k]), int(pairs[k + 1])
        p = pics[dec]
        d = decode_picture(mm, spss, ppss, p)
        mbs = d['mbs']
        counts = apply_edit(mbs, d['sl']['slice_type'], d['sps']['mb_w'], seed)
        nal, _ = encode_picture(d, mbs)
        print(text({'dec': dec, 'seed': seed, 'type': p['type'], 'len': len(nal),
                    'sha256': hashlib.sha256(nal).hexdigest(), 'pcm': counts[0], 'explicit': counts[1],
                    'empty_dc': counts[2], 'qpd': counts[3]}), flush=True)



def motion_digests(m):
    """Canonical digests of a Python Motion (same packing as Rust `Motion::canonical_digests`):
    little-endian arrays row-major by 4x4 block; refpic None = -1."""
    import hashlib

    import numpy as np
    bufs = [np.array(m.intra, np.uint8).tobytes(), np.array(m.direct, np.uint8).tobytes(),
            b''.join(np.array(m.refidx[l], np.int8).tobytes() for l in range(2)),
            b''.join(np.array(m.mv[l], '<i2').tobytes() for l in range(2)),
            b''.join(np.array([[-1 if r is None else r for r in row] for row in m.refpic[l]], '<i4').tobytes()
                     for l in range(2))]
    h = lambda b: hashlib.sha256(b).hexdigest()
    return h(b''.join(bufs)), [h(b) for b in bufs]


def _motion_task(args):
    """Motion (and with `target` the re-expression) digests of pictures p0..p1 of an IDR period."""
    import hashlib
    import mmap
    from frame_dedup.h264mv import derive_motion
    from frame_dedup.roundtrip_test import decode_picture, encode_picture
    from frame_dedup.stream import load
    from frame_dedup.verify_motion import reexpress
    from .canon import canon_mbs, digest
    path, p0, p1, a, target = args
    spss, ppss, pics = load(path)
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    W, H = spss[0]['mb_w'], spss[0]['mb_h']
    motions, out = {}, []
    for d in range(p0, p1):
        p = pics[d]
        t = p['type']
        dd = decode_picture(mm, spss, ppss, p)
        col = motions.get(p['lists'][1][0]) if t == 'B' else None
        m = derive_motion(p, dd['mbs'], W, H, pics, col)
        if d >= a:
            whole, fields = motion_digests(m)
            out.append({'dec': d, 'type': t, 'digest': whole, 'intra': fields[0], 'direct': fields[1],
                        'refidx': fields[2], 'mv': fields[3], 'refpic': fields[4]})
            if target and t != 'I':
                new = [reexpress(mb, t, m) for mb in dd['mbs']]
                m2 = derive_motion(p, new, W, H, pics, col, target=m)
                bad = sum(1 for l in range(2) for y in range(H * 4) for x in range(W * 4)
                          if m.refidx[l][y][x] >= 0 and (m.mv[l][y][x] != m2.mv[l][y][x]
                                                         or m.refidx[l][y][x] != m2.refidx[l][y][x]))
                rec = digest(canon_mbs(new))
                nal, _ = encode_picture(dd, new)
                out.append({'dec': d, 'target': True, 'type': t, 'records': rec, 'bad': bad, 'nal_len': len(nal),
                            'nal_sha256': hashlib.sha256(nal).hexdigest(), 'motion2': motion_digests(m2)[0]})
        motions[d] = m
        if p['ref_idc'] or p['idr']:
            keep = set(p['dpb_after'])
            motions = {k: v for k, v in motions.items() if k in keep}
    return out


@command
def motion(path, *rest):
    """motion <file.h264> [--from N] [--count N] [--target] [--workers N]: per picture, the digests
    of the canonical motion field; with --target also the re-expression (all inter MBs explicit 8x8,
    target-mode mvd solve, re-encode) digests. Derivation starts at the IDR of the period holding
    --from; one process per IDR period."""
    import multiprocessing as mp
    import time
    from frame_dedup.stream import load
    from .canon import text
    opts, it, pos = {'from': 0, 'count': None, 'target': False, 'workers': 1}, iter(rest), []
    for x in it:
        if x == '--from':
            opts['from'] = int(next(it))
        elif x == '--count':
            opts['count'] = int(next(it))
        elif x == '--target':
            opts['target'] = True
        elif x == '--workers':
            opts['workers'] = int(next(it))
        elif x == '--threads':
            next(it)
        else:
            pos.append(x)
    _, _, pics = load(path)
    a = min(opts['from'], len(pics))
    b = len(pics) if opts['count'] is None else min(a + opts['count'], len(pics))
    starts = [p['dec'] for p in pics if p['idr']]
    tasks = []
    for k, s0 in enumerate(starts):
        e0 = starts[k + 1] if k + 1 < len(starts) else len(pics)
        if e0 > a and s0 < b:
            tasks.append((path, s0, min(e0, b), a, opts['target']))
    t0 = time.time()
    with mp.get_context('spawn').Pool(opts['workers']) as pool:
        for lines in pool.imap(_motion_task, tasks):
            for l in lines:
                print(text(l), flush=True)
    print(f'motion: pictures {a}..{b} in {time.time() - t0:.2f}s', file=sys.stderr)


# ---------------------------------------------------------------------------
# patch_pictures corpus: real calls recorded from Python's repair tasks
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


def _scenario(work, mkv, pkl):
    """Run the current pipeline (analyse, plan, write) in-process up to the repair call, with the
    cached proxy features in `work` (in.h264 and features.npz must be there), and capture the
    arguments `pipeline` passes to `repair`. Cached in `pkl`."""
    import pickle
    import types
    if os.path.exists(pkl):
        with open(pkl, 'rb') as f:
            scn = pickle.load(f)
        if isinstance(scn, tuple):
            # work/mmco/full_tasks.pkl (written by work/mmco/t_task.py): the repair inputs of the full Amazon run
            spss, ppss, pics, npics, orig_of_out, sigma = scn
            w = os.path.join(os.path.dirname(pkl), 'full_work')
            scn = {'explicit': True, 'orig_es': os.path.join(w, 'in.h264'), 'new_es': os.path.join(w, 'dedup.h264'),
                   'spss': spss, 'ppss': ppss, 'pics': pics, 'new_pics': npics, 'drop': None, 'sigma': sigma,
                   'work': ORACLE_DIR, 'orig_of_out': orig_of_out}
        return scn
    from frame_dedup import pipeline as P
    args = types.SimpleNamespace(input=mkv, output=None, work=work, workers=4, report=None, verify=False,
                                 keep_work=True, dry_run=False, no_rebuild=False, keep_idr_pairs=False)
    cap = {}

    def fake_repair(*a, **k):
        cap['args'], cap['kwargs'] = a, dict(k)
        raise _Captured()
    P.extract_annexb = lambda *a, **k: None
    P.repair = fake_repair
    run = P.Run(args, log=lambda s: print(s, file=sys.stderr, flush=True))
    run.analyse()
    run.plan()
    try:
        run.write()
    except _Captured:
        pass
    a, k = cap['args'], cap['kwargs']
    k.pop('log', None)
    scn = {'explicit': k.get('sigma') is not None, 'orig_es': a[0], 'new_es': a[1], 'spss': a[2], 'ppss': a[3],
           'pics': a[4], 'new_pics': a[5], 'drop': a[6], 'sigma': k.get('sigma'), 'work': k.get('work'),
           'orig_of_out': k.get('orig_of_out')}
    with open(pkl, 'wb') as f:
        pickle.dump(scn, f)
    return scn


def _scenario_tasks(scn):
    from frame_dedup import repair as R
    pics, npics = scn['pics'], scn['new_pics']
    if scn['explicit']:
        tasks, cands = R.build_tasks_explicit(pics, npics, dict(enumerate(scn['orig_of_out'])), scn['sigma'])
    else:
        new_of = {p['dec']: i for i, p in enumerate(p for p in pics if p['dec'] not in set(scn['drop']))}
        cands, _ = R.find_candidates(scn['spss'], pics, scn['drop'])
        tasks = R.build_tasks(pics, npics, new_of, cands)
    return tasks, cands


def _task_force(scn, task):
    if not scn['explicit']:
        return []
    orig, sigma = task['orig'], scn['sigma']
    new_of = {v: k for k, v in task['orig_of_new'].items()}
    return sorted(new_of[d] for d in new_of if orig[d]['type'] == 'B' and sigma.get(orig[d]['lists'][1][0]) is not None)


def _check_tails(scn, task):
    """Every picture of the task must carry the slice data of the original picture it is mapped to
    (explicit mode rewrites headers only), read from the stream files as they are now."""
    import mmap

    def tail(path, sl):
        with open(path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                return bytes(mm[sl['offset'] + sl['size'] - 48:sl['offset'] + sl['size']])
            finally:
                mm.close()
    if not scn['explicit']:
        return
    for nd, od in sorted(task['orig_of_new'].items()):
        a = tail(scn['new_es'], task['new'][nd]['slices'][0])
        b = tail(scn['orig_es'], task['orig'][od]['slices'][0])
        assert a == b, f'period {task["period"]}: picture {nd} does not match original {od} in {scn["new_es"]}'


@command
def patch_plan(work, mkv, pkl):
    """patch-plan <work dir> <input.mkv> <scenario.pkl>: run analyse / plan / write in-process (needs
    in.h264 and features.npz in the work dir), cache the repair inputs in the pickle, and list the
    repair tasks: period, pictures, candidate pictures, forced pictures."""
    from .canon import text
    scn = _scenario(work, mkv, pkl)
    tasks, cands = _scenario_tasks(scn)
    print(text({'explicit': scn['explicit'], 'orig_es': scn['orig_es'], 'new_es': scn['new_es'],
                'pictures': len(scn['pics']), 'new_pictures': len(scn['new_pics']), 'candidates': len(cands),
                'tasks': len(tasks)}))
    cset = set(cands)
    for t in tasks:
        n_cand = len(t['cands']) if 'cands' in t else sum(1 for d in t['orig_of_new'].values() if d in cset)
        print(text({'period': t['period'], 'new_pictures': len(t['new']), 'candidates': n_cand,
                    'orig_periods': t.get('orig_periods'), 'force': len(_task_force(scn, t)),
                    'range': [min(t['new']), max(t['new'])]}))


def _recorder(real, calls, W):
    """Wrap `patch_pictures` so that every call appends a record to `calls`: patch sets, the target PCM
    samples, force, the digests of the NALs returned, the stats and the log lines."""
    import hashlib
    import time
    from frame_dedup.patch import pcm_record

    def wrapped(mm, spss_, ppss_, pics, patches, yuv_of, log=print, parsed=None, stats=None, force=(), cache=None):
        rec = {'patches': {str(k): [list(v) for v in vs] for k, vs in sorted(patches.items())},
               'force': sorted(force), 'targets': {}}
        for d, vs in sorted(patches.items()):
            if not vs:
                continue
            y = yuv_of(d)
            rec['targets'][str(d)] = {
                f'{x},{yy}': list((y[(x, yy)] if isinstance(y, dict) else pcm_record(y, W, x, yy))['samples'])
                for x, yy in sorted(set(map(tuple, vs)))}
        lines = []

        def lg(s):
            lines.append(s)
            log(s)
        t0 = time.time()
        repl = real(mm, spss_, ppss_, pics, patches, yuv_of, lg, parsed, stats, force, cache)
        rec['seconds'] = round(time.time() - t0, 2)
        rec['repl'] = {str(d): {'len': len(n), 'sha256': hashlib.sha256(n).hexdigest()} for d, n in sorted(repl.items())}
        rec['stats'] = {str(d): dict(s) for d, s in sorted((stats or {}).items())}
        rec['log'] = lines
        calls.append(rec)
        return repl
    return wrapped


@command
def patch_record(pkl, out, *periods):
    """patch-record <scenario.pkl> <corpus.json> <period>...: run the given repair tasks in-process
    (as work/mmco/t_task.py does) with patch_pictures wrapped to record every call: patch sets, the
    target PCM samples, force, and the digests of the returned NALs, the stats and the log lines."""
    import hashlib
    import time
    from frame_dedup import repair as R
    from frame_dedup.patch import pcm_record
    from .canon import text
    scn = _scenario(None, None, pkl)
    spss = scn['spss']
    W = spss[0]['mb_w']
    tasks, _ = _scenario_tasks(scn)
    for g in map(int, periods):   # the scenario's streams must match its task data: slice data tails agree
        _check_tails(scn, [t for t in tasks if t['period'] == g][0])
    R._init(scn['orig_es'], scn['new_es'], spss, scn['ppss'], scn['sigma'], scn['work'])
    calls = []
    wrapped = _recorder(R.patch_pictures, calls, W)
    R.patch_pictures = wrapped
    result = {'explicit': scn['explicit'], 'orig_es': scn['orig_es'], 'new_es': scn['new_es'], 'tasks': []}
    for g in map(int, periods):
        task = [t for t in tasks if t['period'] == g][0]
        calls.clear()
        t0 = time.time()
        res = R.run_task(task)
        print(f'period {g}: {len(task["new"])} pictures, {len(calls)} patch_pictures calls, {res.get("rounds")} rounds, '
              f'{sum(res["patches"].values())} PCM MBs, {time.time() - t0:.0f} s', file=sys.stderr, flush=True)
        ent = {'period': g, 'range': [min(task['new']), max(task['new'])], 'force': _task_force(scn, task),
               'calls': list(calls), 'seconds': res['seconds'], 'rounds': res.get('rounds'),
               'final_repl': {str(d): hashlib.sha256(n).hexdigest() for d, n in sorted(res['repl'].items())}}
        if scn['explicit']:
            ent['orig_of_new'] = {str(k): v for k, v in sorted(task['orig_of_new'].items())}
        result['tasks'].append(ent)
        with open(out, 'w') as f:
            f.write(text(result))


@command
def patch_synth(pkl, out, seed, rounds, density, *periods):
    """patch-synth <scenario.pkl> <corpus.json> <seed> <rounds> <density> <period>...: synthetic patch_pictures
    calls on the pictures of exact-mode tasks, for paths the real repairs rarely take (temporal direct made
    explicit, skip made explicit, QP carriers). Each round adds pseudo-random macroblocks (about `density`
    per picture over all rounds) with pseudo-random target samples to about a third of the P and B pictures,
    and calls patch_pictures on the growing patch set with the same `parsed` dict and `cache`, as the measured
    repair does. The Python results are the oracle."""
    import hashlib
    import mmap
    from frame_dedup import repair as R
    from frame_dedup.patch import patch_pictures
    from .canon import text
    scn = _scenario(None, None, pkl)
    assert not scn['explicit'], 'patch-synth is for exact-mode scenarios'
    spss, ppss = scn['spss'], scn['ppss']
    W, H = spss[0]['mb_w'], spss[0]['mb_h']
    tasks, _ = _scenario_tasks(scn)
    with open(scn['new_es'], 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        result = {'explicit': False, 'orig_es': scn['orig_es'], 'new_es': scn['new_es'], 'tasks': []}
        rng = SplitMix64(int(seed))
        rounds, density = int(rounds), int(density)
        for g in map(int, periods):
            task = [t for t in tasks if t['period'] == g][0]
            new = task['new']
            ds = sorted(d for d in new if new[d]['slices'][0]['slice_type'] != 'I')
            calls, cache, parsed = [], {}, {}
            patches, targets = {}, {}
            wrapped = _recorder(patch_pictures, calls, W)
            for r in range(rounds):
                for d in ds:
                    if rng.next() % 3:
                        continue
                    for _ in range(max(1, density // rounds)):
                        x, y = rng.next() % W, rng.next() % H
                        if (x, y) in targets.get(d, {}):
                            continue
                        rec = {'skip': 0, 'kind': 'PCM', 'samples': [rng.next() & 255 for _ in range(384)], 'intra': 1,
                               'pcm': 1, 'i16': 0, 'direct16': 0, 'cbp': 0x2f, 't8': 0, 'cmode': 0, 'qpd': 0}
                        targets.setdefault(d, {})[(x, y)] = rec
                        patches.setdefault(d, []).append((x, y))
                stats, lines = {}, []
                wrapped(mm, spss, ppss, new, {k: sorted(v) for k, v in patches.items()}, targets.__getitem__,
                        log=lines.append, parsed=parsed, stats=stats, force=(), cache=cache)
            print(f'period {g}: {len(calls)} calls, {sum(len(v) for v in patches.values())} PCM MBs', file=sys.stderr, flush=True)
            result['tasks'].append({'period': g, 'range': [min(new), max(new)], 'force': [], 'calls': calls})
        mm.close()
    with open(out, 'w') as f:
        f.write(text(result))


if __name__ == '__main__':
    main(sys.argv[1:])

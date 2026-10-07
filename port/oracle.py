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
# ---------------------------------------------------------------------------
# header planning, stream writing, anchor rebuild headers
# (Rust: explicit.rs, stream.rs, rebuild.rs)
# ---------------------------------------------------------------------------

def _load(path):
    """stream.load, or (spss, ppss, pics) from a pickle whose first three items are those (work/mmco/full_tasks.pkl:
    the full Amazon file's parse, so it need not be parsed again; the stream path is then still needed by the writers)."""
    import os
    import pickle
    if os.environ.get('ORACLE_PARSED'):     # a cached parse of this very file (see pickle-load)
        return pickle.load(open(os.environ['ORACLE_PARSED'], 'rb'))[:3]
    if path.endswith('.pkl'):
        return pickle.load(open(path, 'rb'))[:3]
    from frame_dedup.stream import load
    return load(path)


@command
def pickle_load(path, out):
    """pickle-load <file.h264> <out.pkl>: cache stream.load's result for the commands below (set ORACLE_PARSED)."""
    import pickle
    from frame_dedup.stream import load
    pickle.dump(load(path), open(out, 'wb'))


def _err(e):
    """An exception as a dump string; assertion failures carry no comparable message."""
    return 'AssertionError' if isinstance(e, AssertionError) else str(e)


def _scenario(path):
    """drop set, sigma, rebuilds [(K, D)], timing, repl {dec: src dec} from a scenario JSON."""
    j = json.load(open(path))
    drop = set(j.get('drop', []))
    sigma = {int(k): v for k, v in j.get('sigma', {}).items()}
    rebuilds = [tuple(x) for x in j.get('rebuilds', [])]
    timing = tuple(j['timing']) if j.get('timing') else None
    repl = {int(k): v for k, v in j.get('repl', {}).items()}
    return drop, sigma, rebuilds, timing, repl


def _nal_bytes(mm, p):
    sl = p['slices'][0]
    return bytes(mm[sl['offset']:sl['offset'] + sl['size']])


def _plan_lines(entries, hdrs):
    from .canon import text
    print(text({'entries': len(entries)}))
    for i, (e, h) in enumerate(zip(entries, hdrs)):
        print(text({'i': i,
                    'e': {'pic': e['pic']['dec'], 'idr': bool(e['idr']), 'poc': e['poc'],
                          'demote': bool(e.get('demote')), 'as': e['as']['dec'] if e.get('as') else None},
                    'h': {'frame_num': h['frame_num'], 'poc_lsb': h['poc_lsb'], 'rplm': h['rplm'],
                          'mmco': h['mmco'], 'demote': bool(h['demote']), 'lists': h['lists'],
                          'same': bool(h['same'])}}))


@command
def plan_stream(path, scenario):
    """plan-stream <file.h264> <scenario.json>: explicit.plan_stream as JSONL (or the PlanError)."""
    from frame_dedup import explicit as X
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = _load(path)
    drop, sigma, rebuilds, timing, repl = _scenario(scenario)
    try:
        entries, hdrs = X.plan_stream(spss, pics, drop, sigma, rebuilds)
    except X.PlanError as e:
        print(text({'plan_error': str(e)}))
        return
    _plan_lines(entries, hdrs)


@command
def write_explicit(path, scenario, out, parsed=None):
    """write-explicit <file.h264> <scenario.json> <out.h264> [parsed.pkl]: plan_stream + write_explicit (parsed.pkl: a
    stream.load result to use instead of parsing the file again)."""
    import os
    import time
    from frame_dedup import explicit as X
    from frame_dedup.stream import load, write_explicit as we
    from .canon import text
    spss, ppss, pics = _load(parsed or path)
    drop, sigma, rebuilds, timing, repl = _scenario(scenario)
    t0 = time.time()
    entries, hdrs = X.plan_stream(spss, pics, drop, sigma, rebuilds)
    t1 = time.time()
    n_rw, n_sps = we(path, pics, spss, ppss, out, entries, hdrs, timing)
    t2 = time.time()
    print(text({'n_rw': n_rw, 'n_sps': n_sps, 'size': os.path.getsize(out), 'entries': len(entries)}))
    print(f'plan {t1 - t0:.1f} s, write {t2 - t1:.1f} s', file=sys.stderr)


@command
def write_stream(path, scenario, out, *flags):
    """write-stream <file.h264> <scenario.json> <out.h264> [--hdrs]: stream.write_stream with the scenario's
    drops, timing and replacement NALs (the NAL bytes of another picture)."""
    import mmap
    import os
    from frame_dedup import explicit as X
    from frame_dedup.stream import load, write_stream as ws
    from .canon import text
    spss, ppss, pics = _load(path)
    drop, sigma, rebuilds, timing, repl_src = _scenario(scenario)
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        repl = {d: _nal_bytes(mm, pics[s]) for d, s in repl_src.items()}
        mm.close()
    hdrs = None
    if '--hdrs' in flags:
        entries, hl = X.plan_stream(spss, pics, drop, sigma, [])
        hdrs = {e['pic']['dec']: h for e, h in zip(entries, hl)}
    n_renum, n_sps = ws(path, pics, spss, out, drop=drop, repl=repl or None, timing=timing, hdrs=hdrs, ppss=ppss)
    print(text({'n_renum': n_renum, 'n_sps': n_sps, 'size': os.path.getsize(out)}))


@command
def new_lists(path, scenario=None):
    """new-lists <file.h264> [scenario.json]: free_drop_checker verdict and a digest of new_lists for a single
    drop of every picture (or the error), then new_lists for the scenario's drops in full."""
    import zlib
    from frame_dedup.stream import free_drop_checker, load, new_lists as nl
    from .canon import text
    spss, ppss, pics = _load(path)
    free = free_drop_checker(spss, pics)
    for p in pics:
        try:
            f = bool(free(p['dec']))
        except Exception as e:
            f = {'error': _err(e)}
        try:
            m = nl(spss, pics, {p['dec']})
            t = text([[d, m[d]] for d in sorted(m)])
            d = [len(t), zlib.crc32(t.encode())]
        except Exception as e:
            d = {'error': _err(e)}
        print(text({'dec': p['dec'], 'free': f, 'nl': d}))
    if scenario:
        drop = _scenario(scenario)[0]
        try:
            m = nl(spss, pics, drop)
            for d in sorted(m):
                print(text({'dec': d, 'lists': m[d]}))
        except Exception as e:
            print(text({'error': _err(e)}))


def _pair_candidates(pics, n):
    """(K, D): the last n non-IDR reference pictures of each IDR period with the next period's IDR."""
    periods = {}
    for p in pics:
        periods.setdefault(p['period'], []).append(p)
    out = []
    for g in range(len(periods) - 1):
        D = periods[g + 1][0]['dec']
        ks = [p['dec'] for p in periods[g] if p['ref_idc'] and not p['idr']]
        out += [(k, D) for k in ks[-n:]]
    return out, periods


@command
def explicit_candidates(path, n='3'):
    """explicit-candidates <file.h264> [n]: explicit.candidate for the last n reference pictures of every period
    with the next IDR, then free_checker's verdict of every picture."""
    import time
    from frame_dedup import explicit as X
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = _load(path)
    pairs, periods = _pair_candidates(pics, int(n))
    t0 = time.time()
    for k, d in pairs:
        r, why = X.candidate(spss, pics, periods, k, d)
        print(text({'K': k, 'D': d, 'r': r, 'reason': why}))
    print(f'candidates {time.time() - t0:.1f} s', file=sys.stderr)
    free = X.free_checker(pics)
    print(text({'free': [d for d in range(len(pics)) if free(d)]}))


@command
def rebuild_candidates(path):
    """rebuild-candidates <file.h264>: rebuild.candidate for (last reference picture of period g, IDR of period
    g+1), then check_lists over the whole stream with all accepted candidates."""
    import time
    from frame_dedup import rebuild as R
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = _load(path)
    pairs, periods = _pair_candidates(pics, 1)
    t0 = time.time()
    rebuilds = []
    for k, d in pairs:
        r, why = R.candidate(spss, pics, periods, k, d)
        print(text({'K': k, 'D': d, 'r': r, 'reason': why}))
        if r:
            rebuilds.append(r)
    print(f'candidates {time.time() - t0:.1f} s', file=sys.stderr)
    fails, shadow = R.check_lists(spss, pics, rebuilds)
    print(text({'fails': fails, 'n': len(shadow)}))
    for q in shadow:
        print(text({'src': q['src'], 'fn': q['slices'][0]['frame_num'], 'poc': q['poc'], 'idr': bool(q['idr']),
                    'lists': q['lists']}))


@command
def write_rebuilt(path, spec, prefix):
    """write-rebuilt <file.h264> <spec.json> <outprefix>: rebuild.write_rebuilt on segments of IDR periods."""
    import mmap
    import os
    import time
    from frame_dedup import rebuild as R
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = _load(path)
    pairs, periods = _pair_candidates(pics, 1)
    acc = []
    for k, d in pairs:
        r, why = R.candidate(spss, pics, periods, k, d)
        if r:
            acc.append(r)
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        spec = json.load(open(spec))
        for i, seg in enumerate(spec['segments']):
            p0, p1 = seg['p0'], seg['p1']
            run = [p for g in range(p0, p1 + 1) for p in periods[g]]
            lo, hi = run[0]['dec'], run[-1]['dec']
            rebuilds = [r for r in acc if lo <= r['K'] <= hi and lo <= r['D'] <= hi]
            repl = {int(k): _nal_bytes(mm, pics[v]) for k, v in seg.get('repl', {}).items()}
            out = f'{prefix}_{i}.h264'
            t0 = time.time()
            entries = R.write_rebuilt(path, run, spss, ppss, out, rebuilds, repl)
            print(f'segment {i}: {time.time() - t0:.1f} s', file=sys.stderr)
            print(text({'i': i, 'rebuilds': [[r['K'], r['D'], r['B']] for r in rebuilds],
                        'size': os.path.getsize(out), 'entries': len(entries)}))
            for p, i2, fn, poc in entries:
                print(text([p['dec'], i2, fn, poc]))
        mm.close()


def _opts(args):
    return {a.split('=', 1)[0]: a.split('=', 1)[1] for a in args if '=' in a}


@command
def make_scenario(kind, path, out, *args):
    """make-scenario <explicit|old|rebuilt> <file.h264> <out.json> [step=5] [offset=0] [refdrops=0] [repl=0]
    [timing=1001,48000]: a drop scenario for the writers.
      explicit: plan.choose_drops with a dup at every step-th display frame (explicit mode, the pipeline's sigma);
      old: the same without explicit mode (free_drop_checker), plus `refdrops` extra reference drops and `repl`
           replacement NALs (each a kept non-reference picture getting the NAL of another one of its type);
      rebuilt: a spec for write-rebuilt: up to three segments of consecutive rebuild pairs (runs of 1, 2, 3+),
           each with `repl` replacement NALs."""
    import numpy as np
    from frame_dedup import explicit as X
    from frame_dedup import rebuild as R
    from frame_dedup.plan import choose_drops
    from frame_dedup.stream import free_drop_checker, load
    o = _opts(args)
    step, offset = int(o.get('step', 5)), int(o.get('offset', 0))
    spss, ppss, pics = _load(path)
    N = len(pics)
    timing = [int(x) for x in o.get('timing', '1001,48000').split(',')] if o.get('timing') != 'none' else None
    if kind == 'rebuilt':
        pairs, periods = _pair_candidates(pics, 1)
        ok = set()
        for k, d in pairs:
            r, why = R.candidate(spss, pics, periods, k, d)
            if r:
                ok.add(pics[k]['period'])
        runs, g = [], 0
        while g < len(periods):
            if g in ok:
                h = g
                while h + 1 in ok:
                    h += 1
                runs.append((g, h - g + 1))
                g = h + 1
            else:
                g += 1
        # segments of 1, 2, 3 and 6 consecutive rebuild pairs (inside the runs of accepted pairs), a few places each
        picked = []
        for n in (1, 2, 3, 6):
            c = [r for r in runs if r[1] >= n]
            if c:
                r = max(c, key=lambda x: x[1])
                for j in (1, 2):
                    picked.append((r[0] + (r[1] - n) * j // 3, n))
        segs = []
        nrepl = int(o.get('repl', 3))
        for g0, n in picked:
            run = [p for g in range(g0, g0 + n + 1) for p in periods[g]]
            ks = {pics[k]['dec'] for k, d in pairs if pics[k]['period'] in range(g0, g0 + n)}
            nonref = [p['dec'] for p in run if not p['ref_idc'] and p['type'] == 'B']
            repl = {}
            for j in range(min(nrepl, len(nonref) // 2)):
                repl[str(nonref[2 * j])] = nonref[2 * j + 1]
            segs.append({'p0': g0, 'p1': g0 + n, 'pairs': n, 'repl': repl})
        json.dump({'segments': segs}, open(out, 'w'))
        print(f'{len(runs)} runs of rebuild pairs; segments: {[(s["p0"], s["p1"]) for s in segs]}')
        return
    if kind == 'random':
        # random drops with random substitutes: exercises the PlanError paths
        import random
        rnd = random.Random(int(o.get('seed', 1)))
        n = int(o.get('n', 6))
        drop, sigma = set(), {}
        for _ in range(n):
            d = rnd.randrange(1, N)
            if pics[d]['idr'] or d in drop:
                continue
            drop.add(d)
            if pics[d]['ref_idc']:
                c = rnd.random()
                sigma[d] = None if c < 0.2 else max(0, min(N - 1, d + rnd.choice([-3, -2, -1, 1, 2])))
        json.dump({'drop': sorted(drop), 'sigma': {str(k): v for k, v in sorted(sigma.items())}, 'rebuilds': [],
                   'timing': timing}, open(out, 'w'))
        print(f'{len(drop)} drops, {len(sigma)} sigma')
        return
    explicit = kind == 'explicit'
    disp = [None] * N
    for p in pics:
        disp[p['disp']] = p
    dup = np.zeros(N, bool)
    dup[offset + step::step] = True
    periods = {}
    for p in pics:
        periods.setdefault(p['period'], []).append(p)
    if explicit:
        free = X.free_checker(pics)
        rebuildable = lambda K, D: X.candidate(spss, pics, periods, K, D)
    else:
        free = free_drop_checker(spss, pics)
        rebuildable = None
    drop, hard, unsupported, freed, subs, rebuilds = choose_drops(
        disp, dup, free, None, None, visible_subs=False, rebuildable=rebuildable, explicit=explicit)
    rebuilds = [r for r in rebuilds if 'fallback' not in r]
    drop_dec = {disp[i]['dec'] for i in drop}
    sc = {'timing': timing}
    if explicit:
        rbs = [(r['K_dec'], r['D_dec']) for r in rebuilds]
        rb_K = {K for K, _ in rbs}
        sigma = {}
        for f in freed:
            sigma[f['dec']] = f['sigma']
        for h in hard:
            sigma[h['dropped_dec']] = h['kept_dec']
        for d in drop_dec - rb_K:
            if pics[d]['ref_idc'] and d not in sigma:
                sigma[d] = None
        sc.update(drop=sorted(drop_dec - rb_K), sigma={str(k): v for k, v in sorted(sigma.items())},
                  rebuilds=[list(x) for x in rbs])
    else:
        nref = int(o.get('refdrops', 0))
        cands = [p['dec'] for p in pics if p['ref_idc'] and not p['idr'] and p['dec'] not in drop_dec]
        for j in range(nref):
            drop_dec.add(cands[(2 * j + 1) * len(cands) // (2 * nref)])
        sc['drop'] = sorted(drop_dec)
        nonref = [p['dec'] for p in pics if not p['ref_idc'] and p['type'] == 'B' and p['dec'] not in drop_dec]
        repl = {}
        for j in range(int(o.get('repl', 0))):
            a = nonref[(2 * j + 1) * len(nonref) // (2 * int(o['repl']))]
            b = nonref[(2 * j) * len(nonref) // (2 * int(o['repl']))]
            repl[str(a)] = b
        sc['repl'] = repl
    json.dump(sc, open(out, 'w'))
    print(f'{len(sc["drop"])} drops, {len(sc.get("sigma", {}))} sigma, {len(sc.get("rebuilds", []))} rebuilds, '
          f'{len(sc.get("repl", {}))} repl; hard {len(hard)}, freed {len(freed)}, unsupported {len(unsupported)}')


@command
def full_scenario(out):
    """full-scenario <out.json>: the real full-length explicit run's scenario, recovered like work/mmco/t_task.py:
    the original's pictures (full_tasks.pkl's parse of in.h264) matched with dedup.h264's by slice tails (so the
    drop set is every original decode index not in the output), sigma from full_report.json (unreferenced drops,
    hard pairs, anchor rebuilds' K -> D; every other dropped reference -> None). The pickle's own orig_of_out is
    not used: it belongs to an older run than the current dedup.h264."""
    import mmap
    import os
    import pickle
    from frame_dedup.stream import load
    root = os.environ.get('DEDUP_ROOT', r'C:\Users\aquat\Documents\GitHub\ak-frame-dedup')
    W = os.path.join(root, 'work/mmco/full_work')
    spss, ppss, pics = pickle.load(open(os.path.join(root, 'work/mmco/full_tasks.pkl'), 'rb'))[:3]
    _, _, npics = load(os.path.join(W, 'dedup.h264'))

    import hashlib
    from frame_dedup.h264hdr import unescape

    def keys(path, ps):
        # the slice data (after the header, whose bytes change), exactly
        f = open(path, 'rb')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        res = []
        for p in ps:
            sl = p['slices'][0]
            rbsp = unescape(bytes(mm[sl['offset'] + 1:sl['offset'] + sl['size']]))
            res.append(hashlib.md5(rbsp[sl['data_bit'] // 8:]).digest())
        mm.close()
        f.close()
        return res
    rep = json.load(open(os.path.join(root, 'work/mmco/full_report.json')))
    Ds = {x['D_dec'] for x in rep['anchor_rebuilds']['pairs_detail']}
    in_keys = keys(os.path.join(W, 'in.h264'), pics)
    where = {}
    for i, k in enumerate(in_keys):
        where.setdefault(k, []).append(i)
    # the output is the input minus the drops, in order, except that a demoted D sits in K's slot: match each
    # output picture to the first unused input picture with the same slice data after the last in-order match
    # (identical slice data, e.g. black pictures, is ambiguous: prefer the candidate whose POC differs from the
    # output picture's by the offset of the previous match)
    orig_of_out, last, used, off = [], -1, set(), 0
    for j, k in enumerate(keys(os.path.join(W, 'dedup.h264'), npics)):
        cands = [i for i in where[k] if i > last and i not in used]
        good = [i for i in cands if pics[i]['poc'] + off == npics[j]['poc']]
        m = (good or cands)[0]
        used.add(m)
        if m in Ds:
            Ds.discard(m)       # an I2: its input position is later than its output position
        else:
            last = m
            off = npics[j]['poc'] - pics[m]['poc']
        orig_of_out.append(m)
    assert not Ds, 'rebuilt IDRs not found'
    rep = json.load(open(os.path.join(root, 'work/mmco/full_report.json')))
    sigma = {}
    for f in rep['unreferenced_reference_drops']:
        sigma[f['dec']] = f['sigma']
    for h in rep['hard_pairs']:
        sigma[h['dropped_dec']] = h['kept_dec']
    rbs = [[x['K_dec'], x['D_dec']] for x in rep['anchor_rebuilds']['pairs_detail']]
    for K, D in rbs:
        sigma[K] = D
    kept = set(orig_of_out)
    for p in pics:
        if p['ref_idc'] and p['dec'] not in kept and p['dec'] not in sigma:
            sigma[p['dec']] = None
    json.dump({'drop': [d for d in range(len(pics)) if d not in kept],
               'sigma': {str(k): v for k, v in sorted(sigma.items())},
               'rebuilds': rbs, 'timing': [1001, 48000]}, open(out, 'w'))
    json.dump(orig_of_out, open(out + '.orig_of_out.json', 'w'))
    print(f'{len(pics)} pictures, {len(pics) - len(kept)} dropped, {len(sigma)} sigma, {len(rbs)} rebuilds')


if __name__ == '__main__':
    main(sys.argv[1:])

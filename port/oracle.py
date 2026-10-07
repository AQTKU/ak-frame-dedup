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


if __name__ == '__main__':
    main(sys.argv[1:])

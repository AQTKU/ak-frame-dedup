"""Python-side dumpers for checking the Rust port against frame_dedup v1.2.0.

Usage: python -m port.oracle <command> [args...]

Each command prints canonical JSON (port/canon.py), usually one object per
line, matching the Rust `frame-dedup dev <command>` of the same name.
"""
import json
import sys

COMMANDS = {}


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

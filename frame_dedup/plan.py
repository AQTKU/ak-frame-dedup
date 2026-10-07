"""Turn a detected cadence into a drop list, with A/V-sync corrections.

Drop rules for a duplicate pair (n-1, n) in display order:
  * the later twin if it is a non-reference picture (bit-exact for the rest),
  * else the earlier twin if that is a non-reference picture,
  * else (both reference pictures) the later twin -- a "hard" drop that is
    repaired afterwards -- unless it is an IDR: then the pair is kept
    ("unsupported") and the frame count is corrected in the next black run.

A/V sync: kept frame n becomes output frame k, shown at k / fps_out where it
was shown at n / fps_in. Its offset in output frames is k - n * r. Inside a
stretch of constant phase the offsets cycle around a constant mean. After a
phase change or a kept pair that mean moves; at each black run the number of
black frames kept is adjusted (one extra dropped, or one duplicate kept) so
that the mean offset of the stretch that follows lies within +-0.5 frame.
"""
import numpy as np


def choose_drops(disp, dup):
    """disp: pictures in display order; dup: bool per display index.
    Returns (drop set of display indices, hard list, unsupported list)."""
    drop, hard, unsupported = set(), [], []
    for n in np.nonzero(dup)[0]:
        n = int(n)
        if n == 0:
            continue
        a, b = disp[n - 1], disp[n]
        if not b['ref_idc']:
            drop.add(n)
        elif not a['ref_idc']:
            drop.add(n - 1)
        elif b['idr']:
            unsupported.append({'display': n, 'reason': 'later twin is an IDR',
                                'types': a['type'] + b['type']})
        else:
            drop.add(n)
            hard.append({'kept_display': n - 1, 'dropped_display': n, 'kept_dec': a['dec'],
                         'dropped_dec': b['dec'], 'types': a['type'] + ('i' if a['idr'] else '') + b['type']})
    return drop, hard, unsupported


def offsets(N, drop, r):
    """Offset k - n*r (output frames) of every display index (NaN if dropped)."""
    keep = np.ones(N, bool)
    keep[list(drop)] = False
    k = np.cumsum(keep) - 1
    off = k - np.arange(N) * float(r)
    off[~keep] = np.nan
    return off


def sync_correct(disp, drop, r, black, events, max_off=0.5):
    """Adjust drops inside black runs so that the mean offset of the stretch
    after each run is within +-max_off. `events`: display indices where the
    offset can jump outside a black run (phase changes at hard cuts, kept
    pairs). Returns (drop, corrections)."""
    N = len(disp)
    drop = set(drop)
    corrections = []
    stops = sorted(set([s for s, _ in black] + list(events) + [N]))
    for s, e in black:
        nxt = min(x for x in stops if x >= e) if any(x >= e for x in stops) else N
        if nxt <= e:
            nxt = min([x for x in stops if x > e] + [N])
        if nxt <= e:
            continue
        for _ in range(4):
            off = offsets(N, drop, r)
            m = float(np.nanmean(off[e:nxt]))
            if abs(m) <= max_off:
                break
            if m > max_off:
                # drop one more black frame: a non-reference picture in the run
                cand = [i for i in range(e - 1, s - 1, -1)
                        if i not in drop and not disp[i]['ref_idc'] and not disp[i]['idr']]
                if not cand:
                    corrections.append({'black_run': [s, e], 'action': 'none possible', 'mean_offset': m})
                    break
                drop.add(cand[0])
                corrections.append({'black_run': [s, e], 'action': 'drop extra black frame',
                                    'display': cand[0], 'mean_offset_before': round(m, 3)})
            else:
                cand = [i for i in range(e - 1, s - 1, -1) if i in drop]
                if not cand:
                    corrections.append({'black_run': [s, e], 'action': 'none possible', 'mean_offset': m})
                    break
                drop.discard(cand[0])
                corrections.append({'black_run': [s, e], 'action': 'keep black duplicate',
                                    'display': cand[0], 'mean_offset_before': round(m, 3)})
    return drop, corrections


def sync_stats(N, drop, r, fps_out):
    off = offsets(N, drop, r)
    kept = N - len(drop)
    return {'kept_frames': kept, 'ideal_frames': round(N * float(r), 3),
            'final_offset_frames': round(float(off[~np.isnan(off)][-1]), 3),
            'max_abs_offset_frames': round(float(np.nanmax(np.abs(off))), 3),
            'mean_offset_frames': round(float(np.nanmean(off)), 3),
            'max_abs_offset_ms': round(float(np.nanmax(np.abs(off))) * 1000 / fps_out, 1)}

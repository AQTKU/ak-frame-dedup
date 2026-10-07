"""Turn a detected cadence into a drop list, with A/V-sync corrections.

Drop rules for a duplicate pair (n-1, n) in display order:
  * the later twin if it is a non-reference picture (bit-exact for the rest),
  * else the earlier twin if that is a non-reference picture,
  * else (both reference pictures) a twin whose removal changes no other
    picture's reference lists ("free": nothing reads it, e.g. a P picture
    right before an IDR), later twin preferred -- also bit-exact,
  * else a "substitute": a cheap (non-reference or free) picture within 3
    frames that is a visually lossless repeat of its predecessor (black or
    static), keeping the frame count without any repair,
  * else the later twin -- a "hard" drop that is repaired afterwards --
    unless it is an IDR: then the earlier twin K is dropped by an "anchor
    rebuild" (the IDR is demoted and takes K's place, see rebuild.py) when
    `rebuildable` allows it; otherwise the cheap picture within 3 frames
    with the least motion is dropped instead (a visible substitute: one real
    frame is lost, but A/V sync holds); only if there is none is the pair
    kept ("unsupported") and the count corrected in the next black run.

A/V sync: kept frame n becomes output frame k, shown at k / fps_out where it
was shown at n / fps_in. Its offset in output frames is k - n * r. Inside a
stretch of constant phase the offsets cycle around a constant mean. After a
phase change or a kept pair that mean moves; at each black run the number of
black frames kept is adjusted (one extra dropped, or one duplicate kept) so
that the mean offset of the stretch that follows lies within +-0.5 frame.
"""
import numpy as np


SUB_WINDOW = 3            # frames either side of a pair searched for a substitute drop
INVISIBLE_MAD = 0.25      # proxy mean / block-max difference below which a frame is
INVISIBLE_BMAX = 1.0      # a visually lossless repeat of its predecessor


def choose_drops(disp, dup, free=None, mad=None, bmax=None, visible_subs=True, rebuildable=None):
    """disp: pictures in display order; dup: bool per display index;
    free(dec) -> bool tells whether dropping a reference picture leaves all
    other reference lists unchanged; mad / bmax: proxy differences of each
    frame to its predecessor (for substitute drops); rebuildable(K dec,
    D dec) -> (rebuild dict or None, reason) for pairs whose later twin is an
    IDR.
    Returns (drop set of display indices, hard list, unsupported list,
    free-reference drops list, substitutes list, rebuilds list). A rebuild
    entry records the pair and why it fell back if it did ('fallback')."""
    N = len(disp)
    drop, hard, unsupported, freed, subs, rebuilds = set(), [], [], [], [], []
    twins = set()
    for n in np.nonzero(dup)[0]:
        twins.update((int(n) - 1, int(n)))

    def cheap(i):
        p = disp[i]
        return not p['idr'] and (not p['ref_idc'] or (free is not None and free(p['dec'])))

    def substitute(n, invisible):
        if mad is None:
            return None
        best = None
        for i in range(max(1, n - 1 - SUB_WINDOW), min(N, n + SUB_WINDOW + 1)):
            if i in twins or i in drop or not cheap(i):
                continue
            if invisible and not (mad[i] < INVISIBLE_MAD and bmax[i] < INVISIBLE_BMAX):
                continue
            key = (round(float(mad[i]), 3), abs(i - n))
            if best is None or key < best[0]:
                best = (key, i)
        return None if best is None else best[1]

    for n in np.nonzero(dup)[0]:
        n = int(n)
        if n == 0:
            continue
        a, b = disp[n - 1], disp[n]
        types = a['type'] + ('i' if a['idr'] else '') + b['type'] + ('i' if b['idr'] else '')
        if not b['ref_idc']:
            drop.add(n)
        elif not a['ref_idc']:
            drop.add(n - 1)
        elif free and free(b['dec']):
            drop.add(n)
            freed.append({'display': n, 'twin': 'later', 'types': types})
        elif free and free(a['dec']):
            drop.add(n - 1)
            freed.append({'display': n - 1, 'twin': 'earlier', 'types': types})
        else:
            s = substitute(n, invisible=True)
            if s is not None:
                drop.add(s)
                subs.append({'pair': n, 'display': s, 'visible': False, 'mad': round(float(mad[s]), 3),
                             'types': types})
            elif not b['idr']:
                drop.add(n)
                hard.append({'kept_display': n - 1, 'dropped_display': n, 'kept_dec': a['dec'],
                             'dropped_dec': b['dec'], 'types': types})
            else:
                rb, why = rebuildable(a['dec'], b['dec']) if rebuildable else (None, None)
                if rb is not None:
                    drop.add(n - 1)
                    rebuilds.append({'pair': n, 'dropped_display': n - 1, 'kept_display': n, 'types': types,
                                     'K_dec': a['dec'], 'D_dec': b['dec'], 'B_dec': rb['B']})
                    continue
                if rebuildable:
                    rebuilds.append({'pair': n, 'types': types, 'fallback': why})
                s = substitute(n, invisible=False) if visible_subs else None
                if s is not None:
                    drop.add(s)
                    subs.append({'pair': n, 'display': s, 'visible': True, 'mad': round(float(mad[s]), 3),
                                 'types': types})
                else:
                    unsupported.append({'display': n, 'reason': 'later twin is an IDR and no substitute',
                                        'types': types})
    return drop, hard, unsupported, freed, subs, rebuilds


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

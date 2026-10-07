"""General duplicate-cadence model and detector.

Model: input frame n shows source frame floor(n*r + phi), r = S/T (source
rate over input rate). Frame n repeats frame n-1 iff frac((n-1)*r + phi) < 1 - r.

With r = p/q in lowest terms every arc endpoint is a multiple of 1/q, so the
score of a phase is piecewise constant on [m/q, (m+1)/q) and phi = m/q for
m in 0..q-1 covers every distinct duplicate pattern. Frame n is a duplicate
under phase m iff ((n-1)*p + m) mod q < q - p.

For a segment [a, b) of frames with evidence e, bin each frame's evidence by
c = ((n-1)*p) mod q; the score of phase m is the sum of the bins c with
(c + m) mod q < q - p, i.e. a circular window sum. That is the exact sweep
over the circle in O(L + q), no grid search.

Per candidate ratio the per-segment phases are chosen by dynamic
programming: continuing the previous segment's phase is free, changing it
costs `pen_black` at a black-run boundary and `pen_cut` at a hard cut inside
a chunk. The ratio with the best total wins.
"""
from fractions import Fraction

import numpy as np

SOURCE_RATES = {'23.976': Fraction(24000, 1001), '24': Fraction(24), '25': Fraction(25)}
INPUT_RATES = {'29.97': Fraction(30000, 1001), '30': Fraction(30)}

PEN_BLACK = 25.0     # nats to change phase at a black run of >= 0.5 s
PEN_CUT = 150.0      # nats to change phase at a hard cut inside a chunk
FAIL_SCORE = 2.0     # minimum (mean evidence of predicted dups) - (mean evidence)
MIN_EVIDENCE_DUPS = 200  # predicted duplicates with usable (non-static) evidence


def rate_name(fr, names):
    for k, v in names.items():
        if v == fr:
            return k
    return None


def phase_scores(e, a, b, p, q):
    """Exact score of every phase m (0..q-1) over frames [a, b)."""
    n = np.arange(a, b)
    c = ((n - 1) * p) % q
    h = np.bincount(c, weights=e[a:b], minlength=q)
    C = np.concatenate([[0.0], np.cumsum(np.concatenate([h, h]))])
    s = (-np.arange(q)) % q
    return C[s + (q - p)] - C[s]


def dup_mask(a, b, p, q, m):
    n = np.arange(a, b)
    return ((n - 1) * p + m) % q < q - p


def segments_from(N, black, cuts):
    """Split [0, N) into segments. A black run starts a new segment (the run
    belongs to the segment after it); a hard cut starts a new segment too.
    Returns list of (start, end, kind) with kind in {'start', 'black', 'cut'}."""
    bounds = {0: 'start'}
    for s, e in black:
        if s > 0:
            bounds[s] = 'black'
    for c in cuts:
        if c not in bounds and 0 < c < N:
            # a cut inside (or right after) a black run is not a separate boundary
            if any(s <= c <= e for s, e in black):
                continue
            bounds[c] = 'cut'
    keys = sorted(bounds)
    return [(k, keys[i + 1] if i + 1 < len(keys) else N, bounds[k]) for i, k in enumerate(keys)]


def hard_cuts(feat, e_static, rel_thresh=4.0, abs_thresh=12.0):
    """Frames that start a new shot: a large difference both absolutely and
    relative to the neighbourhood."""
    from .evidence import rel_diff
    mad = feat['mad']
    r = rel_diff(mad)
    idx = np.nonzero((r > rel_thresh) & (mad > abs_thresh))[0]
    return [int(i) for i in idx if i > 0]


def dp_phases(scores, kinds, pen_black=PEN_BLACK, pen_cut=PEN_CUT):
    """scores: list of per-segment phase-score vectors (length q each).
    Returns (total, phases per segment)."""
    V = scores[0].copy()
    back = []
    for s, kind in zip(scores[1:], kinds[1:]):
        pen = pen_black if kind == 'black' else pen_cut
        best = int(np.argmax(V))
        switch = V[best] - pen
        stay = V >= switch
        back.append(np.where(stay, np.arange(len(V)), best))
        V = np.where(stay, V, switch) + s
    m = int(np.argmax(V))
    total = float(V[m])
    phases = [m]
    for b in reversed(back):
        m = int(b[m])
        phases.append(m)
    return total, phases[::-1]


def detect(e, segments, in_rate, pen_black=PEN_BLACK, pen_cut=PEN_CUT):
    """Score every candidate source rate. Returns a list of candidate dicts,
    best first."""
    out = []
    kinds = [k for _, _, k in segments]
    for name, src in SOURCE_RATES.items():
        r = src / in_rate
        if not (Fraction(1, 2) < r < 1):
            continue
        p, q = r.numerator, r.denominator
        sc = [phase_scores(e, a, b, p, q) for a, b, _ in segments]
        total, phases = dp_phases(sc, kinds, pen_black, pen_cut)
        out.append({'source': name, 'rate': src, 'r': r, 'p': p, 'q': q, 'total': total,
                    'phases': phases, 'segment_scores': [float(s[m]) for s, m in zip(sc, phases)]})
    out.sort(key=lambda c: -c['total'])
    return out


def predicted_dups(N, segments, cand):
    dup = np.zeros(N, bool)
    for (a, b, _), m in zip(segments, cand['phases']):
        dup[a:b] = dup_mask(a, b, cand['p'], cand['q'], m)
    dup[0] = False
    return dup


def cadence_score(e, static, dup):
    """Mean evidence of the predicted duplicates minus what a random phase
    would score (the mean evidence over all frames), both over frames whose
    evidence is not masked as static."""
    use = ~static
    use[0] = False
    nd = int((dup & use).sum())
    if nd == 0 or use.sum() == 0:
        return 0.0, nd
    return float(e[dup & use].mean() - e[use].mean()), nd

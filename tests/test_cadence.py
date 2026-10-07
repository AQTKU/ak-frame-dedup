"""Synthetic cadences: every conversion must be recovered exactly."""
from fractions import Fraction

import numpy as np
import pytest

from frame_dedup.cadence import (FAIL_SCORE, INPUT_RATES, SOURCE_RATES, cadence_score, detect, dup_mask,
                                 hard_cuts, phase_scores, predicted_dups, segments_from)
from frame_dedup.evidence import black_runs, evidence, static_mask
from frame_dedup.plan import choose_drops, offsets, sync_correct

CONVERSIONS = [('23.976', '29.97'), ('24', '30'), ('25', '30'),
               ('25', '29.97'), ('24', '29.97'), ('23.976', '30')]


def synth(N, r, phi, rng, black=(), phase_jump=None):
    """Feature arrays for an input whose frame n shows source floor(n*r + phi).
    `black`: display ranges rendered as black. `phase_jump`: (frame, new phi)."""
    src = np.floor(np.arange(N) * float(r) + phi).astype(np.int64)
    if phase_jump:
        f, phi2 = phase_jump
        src[f:] = np.floor(np.arange(f, N) * float(r) + phi2).astype(np.int64) + 10 ** 6
    motion = rng.uniform(1.5, 9.0, src.max() + 2)       # difference to the previous source frame
    mad = np.where(np.diff(src, prepend=src[0] - 1) == 0, rng.uniform(0.0, 0.08, N), motion[src])
    bmax = mad * rng.uniform(4, 8, N)
    mean, mx = np.full(N, 110.0), np.full(N, 230.0)
    for a, b in black:
        mad[a:b] = 0.0
        bmax[a:b] = 0.0
        mean[a:b] = 0.0
        mx[a:b] = 2.0
        if b < N:
            mad[b], bmax[b] = 60.0, 200.0
    mad[0] = bmax[0] = 0.0
    truth = np.zeros(N, bool)
    truth[1:] = src[1:] == src[:-1]
    return {'mad': mad, 'bmax': bmax, 'mean': mean, 'max': mx}, truth


def run_detect(feat, fps_in):
    e = evidence(feat)
    st = static_mask(feat['bmax'])
    black = black_runs(feat, float(fps_in))
    segs = segments_from(len(e), black, hard_cuts(feat, st))
    cands = detect(e, segs, fps_in)
    dup = predicted_dups(len(e), segs, cands[0])
    return cands, dup, cadence_score(e, st, dup)[0], segs, black


@pytest.mark.parametrize('src,inp', CONVERSIONS)
def test_conversion_recovered(src, inp):
    rng = np.random.default_rng(CONVERSIONS.index((src, inp)) + 100)
    fps_in = INPUT_RATES[inp]
    r = SOURCE_RATES[src] / fps_in
    phi = rng.uniform()
    feat, truth = synth(12000, r, phi, rng)
    cands, dup, score, segs, _ = run_detect(feat, fps_in)
    assert cands[0]['source'] == src
    assert cands[0]['r'] == r
    changes = sum(1 for i in range(1, len(segs)) if cands[0]['phases'][i] != cands[0]['phases'][i - 1])
    assert changes == 0
    # phase: the detected bin is the one containing phi
    assert cands[0]['phases'][0] == int(phi * cands[0]['q'])
    assert np.array_equal(dup, truth)
    assert score > FAIL_SCORE


def test_phase_scores_exact():
    rng = np.random.default_rng(3)
    e = rng.normal(size=700)
    for r in (Fraction(4, 5), Fraction(5, 6), Fraction(1001, 1200), Fraction(1001, 1250), Fraction(800, 1001)):
        p, q = r.numerator, r.denominator
        s = phase_scores(e, 13, 650, p, q)
        bf = np.array([e[13:650][dup_mask(13, 650, p, q, m)].sum() for m in range(q)])
        assert np.allclose(s, bf)


def test_phase_change_at_black_and_sync():
    rng = np.random.default_rng(7)
    fps_in = INPUT_RATES['29.97']
    r = Fraction(4, 5)
    N = 20000
    feat, truth = synth(N, r, 0.1, rng, black=[(9000, 9040)], phase_jump=(9040, 0.5))
    cands, dup, score, segs, black = run_detect(feat, fps_in)
    best = cands[0]
    assert best['source'] == '23.976'
    kinds = [segs[i][2] for i in range(1, len(segs)) if best['phases'][i] != best['phases'][i - 1]]
    assert kinds == ['black']
    content = np.ones(N, bool)
    content[9000:9040] = False
    assert np.array_equal(dup[content], truth[content])
    # A/V sync: after corrections the mean offset of every stretch is within half a frame
    disp = [{'dec': i, 'ref_idc': 0, 'idr': i == 0, 'type': 'B'} for i in range(N)]
    disp[0]['ref_idc'] = 1
    drop, hard, uns = choose_drops(disp, dup)
    drop, corr = sync_correct(disp, drop, r, black, [])
    off = offsets(N, drop, r)
    assert abs(np.nanmean(off[9040:])) <= 0.5
    assert abs(np.nanmean(off[:9000])) <= 0.5
    assert abs(off[~np.isnan(off)][-1]) < 1.0


def test_no_cadence_fails():
    """Native-rate content: every frame is new, nothing should pass."""
    rng = np.random.default_rng(11)
    N = 15000
    mad = rng.uniform(0.3, 9.0, N)
    feat = {'mad': mad, 'bmax': mad * rng.uniform(4, 8, N), 'mean': np.full(N, 100.0), 'max': np.full(N, 230.0)}
    feat['mad'][0] = feat['bmax'][0] = 0
    for inp in INPUT_RATES.values():
        cands, dup, score, _, _ = run_detect(feat, inp)
        assert score < FAIL_SCORE

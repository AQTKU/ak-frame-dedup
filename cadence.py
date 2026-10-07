"""Find inserted duplicate frames with a cadence-aware Viterbi decoder.

Evidence per frame i: r_i = (difference between frame i and i-1) / (local
median difference). Repeats have r well below 1. The hidden state is the
cadence phase p (frame i repeats frame i-1 iff i % 5 == p) or "no cadence".
Staying in a state is free; switching costs `switch_cost` nats. Strong
evidence in clear stretches carries the phase through stretches where the
pictures alone cannot decide (smoke, darkness, static shots).
"""
import numpy as np

NO_CADENCE = 5


def rel_diff(mad, half=6):
    N = len(mad)
    r = np.ones(N)
    for i in range(1, N):
        nb = mad[max(1, i - half):i + half + 1]
        r[i] = mad[i] / (np.median(nb) + 0.05)
    return r


def train_llr(r, repeats, bins=np.linspace(-4, 2, 61)):
    """Log-likelihood ratio log P(r|repeat)/P(r|not), binned on log(r)."""
    lr = np.log(np.maximum(r, 1e-3))
    rep = np.zeros(len(r), bool)
    rep[list(repeats)] = True
    rep[0] = False
    h1, _ = np.histogram(lr[rep], bins)
    h0, _ = np.histogram(lr[~rep][1:], bins)
    p1 = (h1 + 1) / (h1.sum() + len(h1))
    p0 = (h0 + 1) / (h0.sum() + len(h0))
    llr = np.log(p1 / p0)
    # smooth and force monotone (smaller r => more likely a repeat)
    llr = np.convolve(np.pad(llr, 2, mode='edge'), np.ones(5) / 5, 'valid')
    llr = np.maximum.accumulate(llr[::-1])[::-1]
    return bins, np.clip(llr, -8, 8)


def emission(r, model):
    bins, llr = model
    idx = np.clip(np.digitize(np.log(np.maximum(r, 1e-3)), bins) - 1, 0, len(llr) - 1)
    e = llr[idx]
    e[0] = 0
    return e


def mask_static(e, blockmax, floor=1.0, half=6):
    """Zero the evidence where nothing in the picture moves (local median of
    the per-frame max 16x16-block difference below `floor`): there the
    differences are compression noise and say nothing about the cadence."""
    e = e.copy()
    N = len(e)
    for i in range(N):
        if np.median(blockmax[max(1, i - half):i + half + 1]) < floor:
            e[i] = 0.0
    return e


def viterbi(e, switch_cost=12.0):
    """e: per-frame repeat log-likelihood ratio. Returns state per frame."""
    N = len(e)
    S = 6
    score = np.zeros(S)
    back = np.zeros((N, S), np.int8)
    idx = np.arange(N)
    for i in range(N):
        gain = np.zeros(S)
        gain[i % 5] = e[i]
        best_prev = int(np.argmax(score))
        stay = score
        switch = score[best_prev] - switch_cost
        choose_switch = switch > stay
        back[i] = np.where(choose_switch, best_prev, np.arange(S))
        score = np.where(choose_switch, switch, stay) + gain
    states = np.zeros(N, np.int8)
    s = int(np.argmax(score))
    for i in range(N - 1, -1, -1):
        states[i] = s
        s = back[i, s]
    return states


def repeats_from_states(states):
    i = np.arange(len(states))
    return set(np.nonzero((states < 5) & (i % 5 == states))[0].tolist())

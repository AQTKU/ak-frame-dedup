"""Per-frame duplicate evidence from a low-resolution luma proxy.

A raw Annex-B decode comes out in display order (IDR period, then POC), so
row i of every array here is display frame i.

Features per transition i (frame i against frame i-1):
  * mad   : mean absolute difference of the 480x270 proxies,
  * bmax  : max over 16x15 blocks of the block-mean absolute difference
            (catches small local motion that the global mean hides).
Per frame: mean and max luma of the proxy (for black detection).
"""
import json
import os
import subprocess

import numpy as np

from .system import popen_kwargs

PW, PH = 480, 270
BW, BH = 16, 15  # proxy block size (30 x 18 blocks)

_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cadence_model.json')


def proxy_features(es_path, threads=8, log=print, batch=256):
    """Decode the Annex-B stream to the gray proxy and return a dict of arrays."""
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-f', 'h264', '-i', es_path,
           '-fps_mode', 'passthrough', '-vf', f'scale={PW}:{PH}:flags=area,format=gray',
           '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=PW * PH * 32,
                         **popen_kwargs())
    fs = PW * PH
    mad, bmax, mean, mx = [], [], [], []
    prev = None
    while True:
        buf = p.stdout.read(fs * batch)
        n = len(buf) // fs
        if n == 0:
            break
        f = np.frombuffer(buf[:n * fs], np.uint8).reshape(n, PH, PW)
        mean.append(f.mean(axis=(1, 2)))
        mx.append(f.max(axis=(1, 2)))
        cur = f.astype(np.int16)
        if prev is None:
            stack = cur
            first = [0.0]
        else:
            stack = np.concatenate([prev[None], cur])
            first = []
        d = np.abs(np.diff(stack, axis=0)).astype(np.float32)
        mad.append(np.array(first + list(d.mean(axis=(1, 2)))))
        bm = d.reshape(len(d), PH // BH, BH, PW // BW, BW).mean(axis=(2, 4)).max(axis=(1, 2))
        bmax.append(np.array(first + list(bm)))
        prev = cur[-1]
        if len(buf) < fs * batch:
            break
    err = p.stderr.read().decode(errors='replace')
    p.wait()
    if p.returncode:
        raise RuntimeError(f'proxy decode failed: {err[:500]}')
    out = {k: np.concatenate(v).astype(np.float64) for k, v in
           (('mad', mad), ('bmax', bmax), ('mean', mean), ('max', mx))}
    return out


# --------------------------------------------------------------------------
# relative differences and log-likelihood ratios (from the prototype)

def _local_median(x, half):
    """Median of x[max(1, i-half) : i+half+1] for every i (index 0 excluded,
    since transition 0 has no predecessor)."""
    N = len(x)
    pad = np.full(N + 2 * half, np.nan)
    pad[half:half + N] = x
    pad[half] = np.nan  # exclude transition 0
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    with np.errstate(all='ignore'):
        med = np.nanmedian(win, axis=1)
    return np.nan_to_num(med, nan=0.0)


def rel_diff(x, half=6, eps=0.05):
    r = x / (_local_median(x, half) + eps)
    r[0] = 1.0
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
    e = np.asarray(llr)[idx]
    e[0] = 0
    return e


def static_mask(bmax, floor=1.0, half=6):
    """True where nothing in the picture moves (local median of the per-frame
    max block difference below `floor`): there the differences are
    compression noise that follows the encoder's GOP rhythm."""
    return _local_median(bmax, half) < floor


def load_model(path=_MODEL):
    m = json.load(open(path))
    return {k: (np.array(v['bins']), np.array(v['llr'])) for k, v in m['features'].items()}, m


def evidence(feat, model=None):
    """Per-frame duplicate log-likelihood ratio (positive: frame i repeats i-1)."""
    if model is None:
        model, _ = load_model()
    e = emission(rel_diff(feat['mad']), model['mad']) + emission(rel_diff(feat['bmax']), model['bmax'])
    e[static_mask(feat['bmax'])] = 0.0
    e[0] = 0.0
    return e


def black_runs(feat, fps, min_seconds=0.5, mean_max=20.0, max_max=40.0):
    """[(start, end)) display-index runs of black frames lasting >= min_seconds."""
    blk = (feat['mean'] < mean_max) & (feat['max'] < max_max)
    runs, i, N = [], 0, len(blk)
    min_len = int(np.ceil(min_seconds * fps))
    while i < N:
        if blk[i]:
            j = i
            while j < N and blk[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs

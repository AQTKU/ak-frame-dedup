"""Dev tool: train the duplicate-evidence LLR tables shipped in
frame_dedup/cadence_model.json.

Labels come from sample.mkv, whose duplicates were retimed 1 ms after their
twin: in PTS order, frame i is a repeat when pts[i] - pts[i-1] <= 5 ms.
(PTS order equals POC display order for that file; this is checked.)

Usage: python tools/train_model.py sample.mkv [--work DIR]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from frame_dedup.evidence import _MODEL, proxy_features, rel_diff, train_llr  # noqa: E402
from frame_dedup.stream import extract_annexb  # noqa: E402


def pts_labels(mkv):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                          'packet=pts_time', '-of', 'csv=p=0', mkv], capture_output=True, text=True).stdout
    pts = np.sort(np.array([float(x) for x in out.split() if x.strip() not in ('', 'N/A')]))
    return set((np.nonzero(np.diff(pts) <= 0.005)[0] + 1).tolist()), len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mkv')
    ap.add_argument('--work', default='work/train')
    ap.add_argument('--out', default=_MODEL)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    es = os.path.join(a.work, 'train.h264')
    feat_path = os.path.join(a.work, 'train_feat.npz')
    if not os.path.exists(feat_path):
        extract_annexb(a.mkv, es)
        np.savez(feat_path, **proxy_features(es))
    feat = dict(np.load(feat_path))
    labels, n = pts_labels(a.mkv)
    assert n == len(feat['mad']), (n, len(feat['mad']))
    model = {'trained_on': os.path.basename(a.mkv), 'n_frames': n, 'n_repeats': len(labels),
             'features': {}}
    for k in ('mad', 'bmax'):
        bins, llr = train_llr(rel_diff(feat[k]), labels)
        model['features'][k] = {'bins': [round(float(x), 6) for x in bins],
                                'llr': [round(float(x), 6) for x in llr]}
    json.dump(model, open(a.out, 'w'), indent=1)
    print(f'{n} frames, {len(labels)} labelled repeats -> {a.out}')


if __name__ == '__main__':
    main()

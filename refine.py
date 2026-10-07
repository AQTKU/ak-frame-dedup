"""Grow I_PCM patches until the patched pictures decode exactly as the original.

Start from the macroblocks that read a dropped reference twin (from the
sniffer). Each round: patch, decode the affected GOPs of both streams, and add
every macroblock of a patched picture that still differs from the original.
The patch border settles where the original decoder did no deblocking across
it and no intra prediction crossed it, so the PCM boundary is lossless there.

Usage: python refine.py <orig.h264> <dedup.h264> <patches.json> <out.h264> [rounds]
patches.json uses decode indices of dedup.h264.
"""
import json
import subprocess
import sys

import numpy as np

from h264hdr import parse_stream
from h264mv import dpb_walk
from patch import patch_stream

W, H = 1920, 1088
FS = W * H * 3 // 2


def au_offsets(pics):
    prev = 0
    for p in pics:
        p['au_start'] = prev
        prev = p['slices'][-1]['offset'] + p['slices'][-1]['size']
        p['au_end'] = prev


def decode_segment(path, pics, a, b):
    """Decode pictures [a, b) (a must be an IDR); return {poc: yuv bytes}."""
    with open(path, 'rb') as f:
        f.seek(pics[a]['au_start'])
        data = f.read(pics[b - 1]['au_end'] - pics[a]['au_start'])
    r = subprocess.run(['ffmpeg', '-v', 'error', '-apply_cropping', 'none', '-f', 'h264', '-i', '-',
                        '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-'],
                       input=data, capture_output=True)
    frames = [r.stdout[i:i + FS] for i in range(0, len(r.stdout), FS)]
    pocs = sorted(p['poc'] for p in pics[a:b])
    assert len(frames) == len(pocs), (path, a, b, len(frames), len(pocs), r.stderr[:300])
    return dict(zip(pocs, frames))


def mb_diff(x, y):
    a = np.frombuffer(x, np.uint8).astype(np.int16)
    b = np.frombuffer(y, np.uint8).astype(np.int16)
    d = np.abs(a - b)
    ly = d[:W * H].reshape(H // 16, 16, W // 16, 16).max(axis=(1, 3))
    u = d[W * H:W * H * 5 // 4].reshape(H // 16, 8, W // 16, 8).max(axis=(1, 3))
    v = d[W * H * 5 // 4:].reshape(H // 16, 8, W // 16, 8).max(axis=(1, 3))
    return np.maximum(ly, np.maximum(u, v))


def main(orig_es, dedup_es, patch_json, out_path, rounds=6):
    rounds = int(rounds)
    patches = {int(k): set(map(tuple, v)) for k, v in json.load(open(patch_json)).items()}
    so, _, po = parse_stream(orig_es)
    sd, _, pd = parse_stream(dedup_es)
    dpb_walk(so, po)
    dpb_walk(sd, pd)
    au_offsets(po)
    au_offsets(pd)
    idr_d = [p['dec'] for p in pd if p['idr']] + [len(pd)]
    idr_o = [p['dec'] for p in po if p['idr']] + [len(po)]
    # GOPs (IDR periods) of the dedup stream that contain patched pictures
    gops = sorted({max(i for i in idr_d if i <= d) for d in patches})
    seg = {}
    for g in gops:
        g_end = min(i for i in idr_d if i > g)
        k = idr_d.index(g)
        seg[g] = (g, g_end, idr_o[k], idr_o[k + 1])
    print(f'{len(patches)} pictures, {sum(map(len, patches.values()))} MBs to start; {len(gops)} GOPs', flush=True)
    orig_frames = {g: decode_segment(orig_es, po, s[2], s[3]) for g, s in seg.items()}

    def yuv_of(d):
        g = max(i for i in idr_d if i <= d)
        return orig_frames[g][pd[d]['poc']]

    for rnd in range(rounds):
        patch_stream(dedup_es, {d: sorted(v) for d, v in patches.items()}, yuv_of, out_path, log=lambda s: None)
        sn, _, pn = parse_stream(out_path)
        dpb_walk(sn, pn)
        au_offsets(pn)
        added = 0
        report = []
        for g, (a, b, _, _) in seg.items():
            new = decode_segment(out_path, pn, a, b)
            ref = orig_frames[g]
            nd, worst = 0, 0
            for p in pn[a:b]:
                m = mb_diff(new[p['poc']], ref[p['poc']])
                if m.any():
                    nd += 1
                    worst = max(worst, int(m.max()))
                if p['dec'] in patches:
                    cur = patches[p['dec']]
                    for y, x in zip(*np.nonzero(m)):
                        x, y = int(x), int(y)
                        border = any((x + dx, y + dy) in cur for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                        if (x, y) not in cur and border:
                            cur.add((x, y))
                            added += 1
            report.append((g, nd, worst))
        total = sum(map(len, patches.values()))
        print(f'round {rnd}: frames still differing {sum(r[1] for r in report)}, worst {max(r[2] for r in report)}; '
              f'added {added} MBs (patch now {total} MBs)', flush=True)
        if not added:
            break
    json.dump({str(k): sorted(v) for k, v in patches.items()}, open(out_path + '.patches.json', 'w'))
    for g, nd, worst in report:
        print(f'  GOP at dedup dec {g}: {nd} frames differ, worst {worst}')


if __name__ == '__main__':
    main(*sys.argv[1:])

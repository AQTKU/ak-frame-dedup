"""Experiment 1: drop the non-reference twin of every duplicate pair.

Pairs are taken from the existing 1ms-retimed MKV (second frame of a pair is
<=5ms after the first). For each pair, if either twin has nal_ref_idc == 0 it
is removed from the Annex-B stream untouched; pairs where both twins are
reference pictures are left alone (reported separately).

Usage: python drop_test.py <in.h264> <packets.csv> <out.h264> <dropped.txt>
packets.csv is `ffprobe -show_entries packet=pts,dts,duration,...` in decode order.
"""
import csv
import mmap
import re
import sys

from h264hdr import parse_stream


def au_ranges(path, pics):
    """Byte range of every access unit (SPS/PPS/SEI + slices), decode order."""
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(mm)
    # an AU starts right after the previous AU's last slice, so any SPS/PPS
    # in front of a slice belongs to it
    ranges = []
    prev_end = 0
    for p in pics:
        last = p['slices'][-1]
        end = last['offset'] + last['size']
        ranges.append((prev_end, end))
        prev_end = end
    ranges[-1] = (ranges[-1][0], size)
    return ranges


def main(es, pk_csv, out, dropped_txt):
    spss, ppss, pics = parse_stream(es)
    pts = [int(r[0]) for r in csv.reader(open(pk_csv))]
    assert len(pts) == len(pics)
    for d, (p, t) in enumerate(zip(pics, pts)):
        p['pts'] = t
        p['dec'] = d
    disp = sorted(pics, key=lambda p: p['pts'])
    for i, p in enumerate(disp):
        p['idx'] = i

    drop, hard = set(), []
    for i in range(len(disp) - 1):
        a, b = disp[i], disp[i + 1]
        if b['pts'] - a['pts'] > 5:
            continue
        if not b['ref_idc']:
            drop.add(b['dec'])
        elif not a['ref_idc']:
            drop.add(a['dec'])
        else:
            hard.append(i)

    ranges = au_ranges(es, pics)
    with open(es, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out, 'wb') as o:
            for d, (s, e) in enumerate(ranges):
                if d in drop:
                    # keep any parameter sets that ride in front of the slice
                    sl = pics[d]['slices'][0]['offset'] - 4
                    if mm[sl] != 0:
                        sl += 1
                    o.write(mm[s:sl])
                    continue
                o.write(mm[s:e])
    with open(dropped_txt, 'w') as f:
        for d in sorted(drop):
            f.write(f"{pics[d]['idx']}\n")
    print(f'pictures {len(pics)}  dropped {len(drop)}  hard pairs left {len(hard)}')


if __name__ == '__main__':
    main(*sys.argv[1:5])

"""Validate motion reconstruction by re-expressing macroblocks.

Every inter MB in a window of pictures is rewritten as an explicit 8x8
partitioning: skips become coded MBs, direct blocks become explicit B_Bi_8x8
blocks, and every mvd is re-solved against the new neighbourhood. If motion
derivation (mvp, P_Skip, temporal direct) is right, the decoded pictures are
bit-identical to the original.

Usage: python verify_motion.py <in.h264> <first dec idx (a P picture)> <count>
"""
import copy
import mmap
import subprocess
import sys

from h264cabac import B_SUB, P_SUB
from h264hdr import parse_stream
from h264mv import derive_motion, dpb_walk
from roundtrip_test import decode_picture, encode_picture

B_SUB_FOR = {'L0': 1, 'L1': 2, 'Bi': 3}


def reexpress(mb, t, motion):
    """Return an equivalent MB record with explicit 8x8 partitions."""
    if mb['intra']:
        return copy.deepcopy(mb)
    n = copy.deepcopy(mb)
    x0, y0 = mb['x'] * 4, mb['y'] * 4
    quad_pred, quad_sub = [], []
    for i in range(4):
        gx, gy = x0 + (i % 2) * 2, y0 + (i // 2) * 2
        use = [motion.refidx[l][gy][gx] >= 0 for l in range(2)]
        pred = 'Bi' if all(use) else ('L0' if use[0] else 'L1')
        quad_pred.append(pred)
        if mb.get('part') == '8x8' and not motion.direct[gy][gx]:
            quad_sub.append(mb['sub'][i])
        else:
            quad_sub.append(0 if t == 'P' else B_SUB_FOR[pred])
    n.update(skip=0, kind=t, part='8x8', sub=quad_sub, direct16=0)
    if t == 'B':
        n['btype'] = 22
    n['refs'] = [[motion.refidx[l][y0 + (i // 2) * 2][x0 + (i % 2) * 2] for i in range(4)] for l in range(2)]
    def nparts(st):
        shape = P_SUB[st] if t == 'P' else B_SUB[st][0]
        return {'8x8': 1, '8x4': 2, '4x8': 2, '4x4': 4}[shape]
    n['mvds'] = [[[(0, 0)] * nparts(st) for st in quad_sub] for _ in range(2)]
    if mb['skip']:
        n.update(cbp=0, t8=0, qpd=0, coef={})
    return n


def segment(mm, pics, start, end, repl):
    out = bytearray()
    pos = None
    for p in pics[start:end]:
        sl = p['slices'][0]
        s0 = sl['offset'] - 3
        if mm[s0 - 1] == 0:
            s0 -= 1
        if pos is None:
            pos = p['au_start']  # includes the SPS/PPS in front of the IDR
        out += mm[pos:s0]
        if p['dec'] in repl:
            out += b'\x00\x00\x00\x01' + repl[p['dec']]
        else:
            out += mm[s0:sl['offset'] + sl['size']]
        pos = sl['offset'] + sl['size']
    return bytes(out)


def frame_hashes(data):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'h264', '-i', '-', '-fps_mode', 'passthrough',
                        '-f', 'framemd5', '-'], input=data, capture_output=True)
    return [l.split(',')[-1].strip() for l in r.stdout.decode().splitlines() if not l.startswith('#')], r.stderr.decode()


def main(es, first, count):
    first, count = int(first), int(count)
    spss, ppss, pics = parse_stream(es)
    dpb_walk(spss, pics)
    prev_end = 0
    for p in pics:
        p['au_start'] = prev_end
        prev_end = p['slices'][-1]['offset'] + p['slices'][-1]['size']
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    f = open(es, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    repl, motions, stats = {}, {}, {}
    for d in range(first, first + count):
        p = pics[d]
        t = p['slices'][0]['slice_type']
        dd = decode_picture(mm, spss, ppss, p)
        col = motions.get(p['lists'][1][0]) if t == 'B' else None
        if t == 'B' and col is None:
            raise SystemExit(f'colocated picture {p["lists"][1][0]} for {d} not in window')
        m = derive_motion(p, dd['mbs'], W, H, pics, col)
        motions[d] = m
        if t == 'I':
            continue
        new = [reexpress(mb, t, m) for mb in dd['mbs']]
        m2 = derive_motion(p, new, W, H, pics, col, target=m)
        p['motion'] = m
        bad = sum(1 for l in range(2) for y in range(H * 4) for x in range(W * 4)
                  if m.refidx[l][y][x] >= 0 and (m.mv[l][y][x] != m2.mv[l][y][x] or m.refidx[l][y][x] != m2.refidx[l][y][x]))
        nal, _ = encode_picture(dd, new)
        repl[d] = nal
        stats[d] = (t, len(dd['nal']), len(nal), bad)
        print(f'dec {d} {t}: {len(dd["nal"])}B -> {len(nal)}B re-expressed, motion mismatches {bad}', flush=True)
    start = max(k for k in range(first + 1) if pics[k]['idr'])
    end = first + count
    a, ea = frame_hashes(segment(mm, pics, start, end, {}))
    b, eb = frame_hashes(segment(mm, pics, start, end, repl))
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    print(f'segment {start}..{end}: frames orig {len(a)} new {len(b)}  differing {len(diff)} {diff[:20]}')
    if ea or eb:
        print('decoder messages:', ea[:300], eb[:300])


if __name__ == '__main__':
    main(*sys.argv[1:4])

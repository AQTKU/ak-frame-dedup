"""Sniff who depends on a hard-pair twin, and predict where damage lands.

For a hard pair (kept twin K, dropped twin D, both reference pictures) this
walks the original stream from K to the next IDR and records, per picture:
  * blocks whose prediction reads D (forward users) and the area they read,
  * blocks whose prediction reads K (backward users before the cut, and D),
  * a predicted taint map: starting from macroblocks where K and D actually
    differ, taint follows motion footprints (and intra neighbours) forward.

The taint map is what dropping D (so D-readers read K instead) can disturb.
"""
import numpy as np

from .h264mv import derive_motion
from .roundtrip_test import decode_picture


def footprint(gx, gy, mv, W, H):
    """Macroblock rectangle a 4x4 luma block reads, incl. 6-tap margins."""
    x = gx * 4 + (mv[0] >> 2)
    y = gy * 4 + (mv[1] >> 2)
    fx, fy = mv[0] & 3, mv[1] & 3
    x0, x1 = x - (2 if fx else 0), x + 3 + (3 if fx else 0)
    y0, y1 = y - (2 if fy else 0), y + 3 + (3 if fy else 0)
    clamp = lambda v, hi: max(0, min(hi, v))
    return (clamp(x0, W * 16 - 1) // 16, clamp(x1, W * 16 - 1) // 16,
            clamp(y0, H * 16 - 1) // 16, clamp(y1, H * 16 - 1) // 16)


def analyse_window(mm, spss, ppss, pics, kept_dec, dropped_dec, end_dec, twin_delta, log=print):
    """twin_delta: (H, W) bool array of macroblocks where K and D differ.
    Returns per-picture dicts keyed by decode index."""
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    motions, out = {}, {}
    taint = {dropped_dec: twin_delta.copy()}  # D's content differs from K's here
    for d in range(kept_dec, end_dec):
        p = pics[d]
        t = p['slices'][0]['slice_type']
        dd = decode_picture(mm, spss, ppss, p)
        col = motions.get(p['lists'][1][0]) if t == 'B' else None
        if t == 'B' and col is None:
            log(f'  skip dec {d}: colocated not in window')
            continue
        m = derive_motion(p, dd['mbs'], W, H, pics, col)
        motions[d] = m
        if d == dropped_dec:
            continue  # D itself is removed; its own reads of K are 'backward'
        reads_D = np.zeros((H, W), bool)      # MBs of this picture reading D
        reads_K = np.zeros((H, W), bool)
        fp_D = np.zeros((H, W), bool)         # MBs of D being read
        fp_K = np.zeros((H, W), bool)
        tnt = np.zeros((H, W), bool)
        nD = nK = 0
        if t != 'I':
            for gy in range(H * 4):
                for gx in range(W * 4):
                    for l in range(2):
                        r = m.refpic[l][gy][gx]
                        if r is None:
                            continue
                        mv = m.mv[l][gy][gx]
                        bx0, bx1, by0, by1 = footprint(gx, gy, mv, W, H)
                        if r == dropped_dec:
                            nD += 1
                            reads_D[gy // 4, gx // 4] = True
                            fp_D[by0:by1 + 1, bx0:bx1 + 1] = True
                        elif r == kept_dec:
                            nK += 1
                            reads_K[gy // 4, gx // 4] = True
                            fp_K[by0:by1 + 1, bx0:bx1 + 1] = True
                        src = taint.get(r)
                        if src is not None and src[by0:by1 + 1, bx0:bx1 + 1].any():
                            tnt[gy // 4, gx // 4] = True
        # intra MBs read their decoded neighbours (left / top / top-left / top-right)
        for mb in dd['mbs']:
            if mb['intra'] and not mb['pcm']:
                x, y = mb['x'], mb['y']
                for dx, dy in ((-1, 0), (0, -1), (-1, -1), (1, -1)):
                    if 0 <= x + dx < W and 0 <= y + dy < H and tnt[y + dy, x + dx]:
                        tnt[y, x] = True
                        break
        if p['ref_idc'] and tnt.any():
            taint[d] = tnt
        out[d] = dict(type=t, disp=p.get('idx'), poc=p['poc'], n_read_D=nD, n_read_K=nK,
                      reads_D=reads_D, reads_K=reads_K, fp_D=fp_D, fp_K=fp_K, taint=tnt)
        log(f"  dec {d} {t} disp {p.get('idx')}: reads D {nD:6d} blk ({reads_D.sum():4d} MB)  "
            f"reads K {nK:6d} blk ({reads_K.sum():4d} MB)  predicted taint {tnt.sum():4d} MB")
    return out, motions

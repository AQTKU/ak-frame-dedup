//! Reconstruction of single inter macroblocks from decoded reference frames:
//! inter prediction (8.4.2: quarter-pel luma, eighth-pel chroma, default,
//! explicit and implicit weighted prediction) and the residual (8.5:
//! dequantisation with flat scaling lists, 4x4 / 8x8 inverse transforms,
//! chroma DC). No intra prediction and no deblocking: references are read
//! from frames a decoder already produced, which are exactly what motion
//! compensation reads.
//!
//! Scope: frame pictures (no fields / MBAFF), 4:2:0, 8-bit, CABAC, flat
//! scaling matrices, no long-term references (implicit weights would need
//! them). [`check_scope`] refuses anything else.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

use crate::decode::{frame_size, iter_frames_with};
use crate::h264hdr::{Picture, Pps, SliceType, Sps};
use crate::mv::{Motion, derive_motion};
use crate::picture::decode_picture;
use crate::records::{BLK_XY, Mb};
use crate::stream::period_ranges;
use crate::system::FFMPEG_THREADS;

/// One decoded yuv420p frame at the coded size (uncropped).
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub y: &'a [u8],
    pub u: &'a [u8],
    pub v: &'a [u8],
    pub w: usize,
    pub h: usize,
}

impl<'a> Frame<'a> {
    pub fn new(buf: &'a [u8], sps: &Sps) -> Frame<'a> {
        let (w, h) = (sps.mb_w * 16, sps.mb_h * 16);
        let (ys, cs) = (w * h, w * h / 4);
        Frame { y: &buf[..ys], u: &buf[ys..ys + cs], v: &buf[ys + cs..ys + 2 * cs], w, h }
    }

    /// (samples, width, height) of component `c` (0 Y, 1 Cb, 2 Cr).
    fn plane(&self, c: usize) -> (&'a [u8], usize, usize) {
        match c {
            0 => (self.y, self.w, self.h),
            1 => (self.u, self.w / 2, self.h / 2),
            _ => (self.v, self.w / 2, self.h / 2),
        }
    }

    /// The samples of macroblock (mbx, mby) in the I_PCM layout: 256 Y, 64
    /// Cb, 64 Cr, each row-major.
    pub fn mb(&self, mbx: usize, mby: usize) -> [u8; 384] {
        let mut out = [0u8; 384];
        for (c, (base, n)) in [(0usize, 16usize), (256, 8), (320, 8)].into_iter().enumerate() {
            let (p, w, _) = self.plane(c);
            for yy in 0..n {
                let row = (mby * n + yy) * w + mbx * n;
                out[base + yy * n..base + yy * n + n].copy_from_slice(&p[row..row + n]);
            }
        }
        out
    }
}

/// Refuse streams outside the supported scope (see the module docs).
pub fn check_scope(sps: &Sps, pps: &Pps, pics: &[Picture]) -> Result<()> {
    if !sps.frame_mbs_only || sps.mbaff {
        bail!("reconstruction supports frame pictures only");
    }
    if sps.chroma_format_idc != 1 || sps.bit_depth != [8, 8] {
        bail!("reconstruction supports 8-bit 4:2:0 only");
    }
    if sps.scaling_matrix || pps.scaling_matrix {
        bail!("reconstruction supports flat scaling lists only");
    }
    if !pps.cabac {
        bail!("reconstruction supports CABAC only");
    }
    let long_term = pics.iter().flat_map(|p| &p.slices).any(|s| s.long_term_ref == Some(true) || s.mmco.iter().any(|m| matches!(m.op, 3 | 6)));
    if long_term {
        bail!("reconstruction does not support long-term references");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// inter prediction (8.4.2.2, 8.4.2.3)
// ---------------------------------------------------------------------------

#[inline]
fn clip1(v: i32) -> i32 {
    v.clamp(0, 255)
}

#[inline]
fn px(p: &[u8], w: usize, h: usize, x: i32, y: i32) -> i32 {
    let x = x.clamp(0, w as i32 - 1) as usize;
    let y = y.clamp(0, h as i32 - 1) as usize;
    p[y * w + x] as i32
}

#[inline]
fn tap6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
    a - 5 * b + 20 * c + 20 * d - 5 * e + f
}

/// Luma sample at full-sample position (x, y) plus quarter-sample fraction
/// (fx, fy) (8.4.2.2.1, Table 8-12).
fn luma_sample(p: &[u8], w: usize, h: usize, x: i32, y: i32, fx: i32, fy: i32) -> i32 {
    let g = |dx: i32, dy: i32| px(p, w, h, x + dx, y + dy);
    if fx == 0 && fy == 0 {
        return g(0, 0);
    }
    // unclipped intermediates: horizontal at row y+dy, vertical at column x+dx
    let b1 = |dy: i32| tap6(g(-2, dy), g(-1, dy), g(0, dy), g(1, dy), g(2, dy), g(3, dy));
    let h1 = |dx: i32| tap6(g(dx, -2), g(dx, -1), g(dx, 0), g(dx, 1), g(dx, 2), g(dx, 3));
    let half = |v: i32| clip1((v + 16) >> 5);
    let j = || clip1((tap6(h1(-2), h1(-1), h1(0), h1(1), h1(2), h1(3)) + 512) >> 10);
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    match (fx, fy) {
        (0, 1) => avg(g(0, 0), half(h1(0))),
        (0, 2) => half(h1(0)),
        (0, 3) => avg(g(0, 1), half(h1(0))),
        (1, 0) => avg(g(0, 0), half(b1(0))),
        (2, 0) => half(b1(0)),
        (3, 0) => avg(g(1, 0), half(b1(0))),
        (1, 1) => avg(half(b1(0)), half(h1(0))),
        (3, 1) => avg(half(b1(0)), half(h1(1))),
        (1, 3) => avg(half(h1(0)), half(b1(1))),
        (3, 3) => avg(half(h1(1)), half(b1(1))),
        (2, 1) => avg(half(b1(0)), j()),
        (2, 3) => avg(j(), half(b1(1))),
        (1, 2) => avg(half(h1(0)), j()),
        (3, 2) => avg(j(), half(h1(1))),
        _ => j(),
    }
}

/// Chroma sample at full-sample position (x, y) plus eighth-sample fraction
/// (8.4.2.2.2).
fn chroma_sample(p: &[u8], w: usize, h: usize, x: i32, y: i32, fx: i32, fy: i32) -> i32 {
    let a = px(p, w, h, x, y);
    if fx == 0 && fy == 0 {
        return a;
    }
    let b = px(p, w, h, x + 1, y);
    let c = px(p, w, h, x, y + 1);
    let d = px(p, w, h, x + 1, y + 1);
    ((8 - fx) * (8 - fy) * a + fx * (8 - fy) * b + (8 - fx) * fy * c + fx * fy * d + 32) >> 6
}

/// How the two predictions of a block are combined, per component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Weights {
    Default,
    /// (logWD, w0, w1, o0, o1) per component; for one list only the
    /// matching weight / offset is used.
    Explicit([(i32, i32, i32, i32, i32); 3]),
    /// Implicit bi-prediction weights (w0, w1); logWD 5, no offsets.
    Implicit(i32, i32),
}

/// Implicit weights for current POC `cur` and reference POCs `p0`, `p1`
/// (8.4.2.3.1; no long-term references).
pub fn implicit_weights(cur: i32, p0: i32, p1: i32) -> (i32, i32) {
    let tb = (cur - p0).clamp(-128, 127);
    let td = (p1 - p0).clamp(-128, 127);
    if td == 0 {
        return (32, 32);
    }
    let tx = (16384 + (td / 2).abs()) / td;
    let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
    if (dsf >> 2) < -64 || (dsf >> 2) > 128 {
        return (32, 32);
    }
    (64 - (dsf >> 2), dsf >> 2)
}

fn block_weights(pps: &Pps, pic: &Picture, pics: &[Picture], ri: [i8; 2], rp: [i32; 2]) -> Result<Weights> {
    let sl = pic.sl();
    let t = sl.slice_type;
    let explicit = (pps.weighted_pred && t == SliceType::P) || (pps.weighted_bipred_idc == 1 && t == SliceType::B);
    if explicit {
        let pwt = sl.pwt.as_ref().ok_or_else(|| anyhow!("weighted slice without a pred_weight_table"))?;
        let entry = |l: usize| -> Result<[(i32, i32); 3]> {
            if ri[l] < 0 {
                return Ok([(0, 0); 3]);
            }
            pwt.lists[l].get(ri[l] as usize).copied().ok_or_else(|| anyhow!("no weights for list {l} index {}", ri[l]))
        };
        let (e0, e1) = (entry(0)?, entry(1)?);
        let mut out = [(0, 0, 0, 0, 0); 3];
        for c in 0..3 {
            let lwd = if c == 0 { pwt.luma_denom } else { pwt.chroma_denom } as i32;
            out[c] = (lwd, e0[c].0, e1[c].0, e0[c].1, e1[c].1);
        }
        return Ok(Weights::Explicit(out));
    }
    if pps.weighted_bipred_idc == 2 && t == SliceType::B && ri[0] >= 0 && ri[1] >= 0 {
        let (w0, w1) = implicit_weights(pic.poc, pics[rp[0] as usize].poc, pics[rp[1] as usize].poc);
        return Ok(Weights::Implicit(w0, w1));
    }
    Ok(Weights::Default)
}

#[inline]
fn combine(wt: Weights, c: usize, p: [Option<i32>; 2]) -> i32 {
    match (wt, p) {
        (Weights::Default, [Some(a), Some(b)]) => (a + b + 1) >> 1,
        (Weights::Default, [Some(a), None]) | (Weights::Default, [None, Some(a)]) => a,
        (Weights::Implicit(w0, w1), [Some(a), Some(b)]) => clip1((a * w0 + b * w1 + 32) >> 6),
        (Weights::Explicit(e), [Some(a), Some(b)]) => {
            let (lwd, w0, w1, o0, o1) = e[c];
            clip1(((a * w0 + b * w1 + (1 << lwd)) >> (lwd + 1)) + ((o0 + o1 + 1) >> 1))
        }
        (Weights::Explicit(e), [a, b]) => {
            let (lwd, w0, w1, o0, o1) = e[c];
            let (s, w, o) = match (a, b) {
                (Some(s), None) => (s, w0, o0),
                (None, Some(s)) => (s, w1, o1),
                _ => unreachable!("a block predicts from at least one list"),
            };
            if lwd >= 1 { clip1(((s * w + (1 << (lwd - 1))) >> lwd) + o) } else { clip1(s * w + o) }
        }
        (Weights::Implicit(..), [Some(a), None]) | (Weights::Implicit(..), [None, Some(a)]) => a,
        (_, [None, None]) => unreachable!("a block predicts from at least one list"),
    }
}

/// The inter prediction of macroblock (mbx, mby) of `pic` (motion `m`,
/// derived in the context of `pics`), in the I_PCM layout. `refs(dec)`
/// returns the decoded frame of reference picture `dec` (a decode index
/// into `pics`).
pub fn predict_mb<'a>(
    pps: &Pps,
    pic: &Picture,
    pics: &[Picture],
    m: &Motion,
    mbx: usize,
    mby: usize,
    refs: &dyn Fn(usize) -> Option<Frame<'a>>,
) -> Result<[u8; 384]> {
    let mut out = [0u8; 384];
    for by in 0..4 {
        for bx in 0..4 {
            let (gx, gy) = (mbx * 4 + bx, mby * 4 + by);
            let i = m.idx(gx, gy);
            if m.intra[i] != 0 {
                bail!("MB ({mbx}, {mby}) is intra");
            }
            let ri = [m.refidx[0][i], m.refidx[1][i]];
            let rp = [m.refpic[0][i], m.refpic[1][i]];
            let mut fr: [Option<Frame>; 2] = [None, None];
            for l in 0..2 {
                if ri[l] >= 0 {
                    if rp[l] < 0 {
                        bail!("MB ({mbx}, {mby}) list {l}: reference index without a picture");
                    }
                    fr[l] = Some(refs(rp[l] as usize).ok_or_else(|| anyhow!("no frame for reference picture {}", rp[l]))?);
                }
            }
            if fr[0].is_none() && fr[1].is_none() {
                bail!("MB ({mbx}, {mby}) block ({bx}, {by}) predicts from no list");
            }
            let wt = block_weights(pps, pic, pics, ri, rp)?;
            let mv = [m.mv[0][i], m.mv[1][i]];
            // luma 4x4
            for yy in 0..4 {
                for xx in 0..4 {
                    let (x, y) = ((gx * 4 + xx) as i32, (gy * 4 + yy) as i32);
                    let mut p = [None, None];
                    for l in 0..2 {
                        if let Some(f) = fr[l] {
                            let (mx, my) = (mv[l][0] as i32, mv[l][1] as i32);
                            p[l] = Some(luma_sample(f.y, f.w, f.h, x + (mx >> 2), y + (my >> 2), mx & 3, my & 3));
                        }
                    }
                    out[(by * 4 + yy) * 16 + bx * 4 + xx] = combine(wt, 0, p) as u8;
                }
            }
            // chroma 2x2 per component
            for c in 1..3 {
                let base = if c == 1 { 256 } else { 320 };
                for yy in 0..2 {
                    for xx in 0..2 {
                        let (x, y) = ((gx * 2 + xx) as i32, (gy * 2 + yy) as i32);
                        let mut p = [None, None];
                        for l in 0..2 {
                            if let Some(f) = fr[l] {
                                let (pl, w, h) = f.plane(c);
                                let (mx, my) = (mv[l][0] as i32, mv[l][1] as i32);
                                p[l] = Some(chroma_sample(pl, w, h, x + (mx >> 3), y + (my >> 3), mx & 7, my & 7));
                            }
                        }
                        out[base + (by * 2 + yy) * 8 + bx * 2 + xx] = combine(wt, c, p) as u8;
                    }
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// residual (8.5)
// ---------------------------------------------------------------------------

/// 4x4 zig-zag scan (frame): scan position -> raster index (row * 4 + col).
pub const ZIGZAG4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// 8x8 zig-zag scan (frame): scan position -> raster index (row * 8 + col).
pub const ZIGZAG8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57,
    50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// QPc as a function of qPI (Table 8-15), for qPI 30..=51.
const QPC: [i32; 22] = [29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39];

/// Chroma QP for luma QP `qp` and chroma_qp_index_offset `off` (8-bit).
pub fn chroma_qp(qp: i32, off: i32) -> i32 {
    let qpi = (qp + off).clamp(0, 51);
    if qpi < 30 { qpi } else { QPC[(qpi - 30) as usize] }
}

/// LevelScale4x4 with a flat weight scale (16 * normAdjust4x4), raster index.
pub fn level_scale4(m: usize, k: usize) -> i32 {
    const V: [[i32; 3]; 6] = [[10, 16, 13], [11, 18, 14], [13, 20, 16], [14, 23, 18], [16, 25, 20], [18, 29, 23]];
    let (i, j) = (k % 4, k / 4);
    let class = if i % 2 == 0 && j % 2 == 0 {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else {
        2
    };
    16 * V[m][class]
}

/// LevelScale8x8 with a flat weight scale (16 * normAdjust8x8), raster index.
pub fn level_scale8(m: usize, k: usize) -> i32 {
    const V: [[i32; 6]; 6] =
        [[20, 18, 32, 19, 25, 24], [22, 19, 35, 21, 28, 26], [26, 23, 42, 24, 33, 31], [28, 25, 45, 26, 35, 33], [32, 28, 51, 30, 40, 38], [36, 32, 58, 34, 46, 43]];
    let (i, j) = (k % 8, k / 8);
    let class = if i % 4 == 0 && j % 4 == 0 {
        0
    } else if i % 2 == 1 && j % 2 == 1 {
        1
    } else if i % 4 == 2 && j % 4 == 2 {
        2
    } else if (i % 4 == 0 && j % 2 == 1) || (i % 2 == 1 && j % 4 == 0) {
        3
    } else if (i % 4 == 0 && j % 4 == 2) || (i % 4 == 2 && j % 4 == 0) {
        4
    } else {
        5
    };
    16 * V[m][class]
}

/// Dequantise a 4x4 level (8.5.12.1, not a DC handled separately).
#[inline]
pub fn dequant4(c: i32, qp: i32, k: usize) -> i32 {
    let ls = level_scale4((qp % 6) as usize, k);
    if qp >= 24 { (c * ls) << (qp / 6 - 4) } else { (c * ls + (1 << (3 - qp / 6))) >> (4 - qp / 6) }
}

/// Dequantise an 8x8 level (8.5.13.1).
#[inline]
pub fn dequant8(c: i32, qp: i32, k: usize) -> i32 {
    let ls = level_scale8((qp % 6) as usize, k);
    if qp >= 36 { (c * ls) << (qp / 6 - 6) } else { (c * ls + (1 << (5 - qp / 6))) >> (6 - qp / 6) }
}

/// Inverse 4x4 transform of dequantised coefficients `d` (raster), giving
/// residual samples (8.5.12.2).
pub fn idct4(d: &[i32; 16]) -> [i32; 16] {
    let mut f = [0i32; 16];
    for r in 0..4 {
        let x = &d[r * 4..r * 4 + 4];
        let (e0, e1, e2, e3) = (x[0] + x[2], x[0] - x[2], (x[1] >> 1) - x[3], x[1] + (x[3] >> 1));
        f[r * 4] = e0 + e3;
        f[r * 4 + 1] = e1 + e2;
        f[r * 4 + 2] = e1 - e2;
        f[r * 4 + 3] = e0 - e3;
    }
    let mut out = [0i32; 16];
    for c in 0..4 {
        let (x0, x1, x2, x3) = (f[c], f[4 + c], f[8 + c], f[12 + c]);
        let (e0, e1, e2, e3) = (x0 + x2, x0 - x2, (x1 >> 1) - x3, x1 + (x3 >> 1));
        out[c] = (e0 + e3 + 32) >> 6;
        out[4 + c] = (e1 + e2 + 32) >> 6;
        out[8 + c] = (e1 - e2 + 32) >> 6;
        out[12 + c] = (e0 - e3 + 32) >> 6;
    }
    out
}

fn idct8_1d(d: [i32; 8]) -> [i32; 8] {
    let e0 = d[0] + d[4];
    let e1 = -d[3] + d[5] - d[7] - (d[7] >> 1);
    let e2 = d[0] - d[4];
    let e3 = d[1] + d[7] - d[3] - (d[3] >> 1);
    let e4 = (d[2] >> 1) - d[6];
    let e5 = -d[1] + d[7] + d[5] + (d[5] >> 1);
    let e6 = d[2] + (d[6] >> 1);
    let e7 = d[3] + d[5] + d[1] + (d[1] >> 1);
    let f0 = e0 + e6;
    let f1 = e1 + (e7 >> 2);
    let f2 = e2 + e4;
    let f3 = e3 + (e5 >> 2);
    let f4 = e2 - e4;
    let f5 = (e3 >> 2) - e5;
    let f6 = e0 - e6;
    let f7 = e7 - (e1 >> 2);
    [f0 + f7, f2 + f5, f4 + f3, f6 + f1, f6 - f1, f4 - f3, f2 - f5, f0 - f7]
}

/// Inverse 8x8 transform (8.5.13.2), raster in and out.
pub fn idct8(d: &[i32; 64]) -> [i32; 64] {
    let mut g = [0i32; 64];
    for r in 0..8 {
        let row = idct8_1d(d[r * 8..r * 8 + 8].try_into().expect("8"));
        g[r * 8..r * 8 + 8].copy_from_slice(&row);
    }
    let mut out = [0i32; 64];
    for c in 0..8 {
        let col = idct8_1d(std::array::from_fn(|r| g[r * 8 + c]));
        for r in 0..8 {
            out[r * 8 + c] = (col[r] + 32) >> 6;
        }
    }
    out
}

/// The residual of inter macroblock `mb` at luma QP `qp`, in the I_PCM
/// layout (256 Y, 64 Cb, 64 Cr).
pub fn residual(mb: &Mb, qp: i32, pps: &Pps) -> Result<[i32; 384]> {
    let mut out = [0i32; 384];
    if mb.intra {
        bail!("residual of an intra MB");
    }
    let Some(coef) = &mb.coef else {
        return Ok(out);
    };
    if mb.t8 {
        for (i8, lv) in coef.l8.iter().enumerate() {
            let Some(lv) = lv else { continue };
            let mut d = [0i32; 64];
            for (s, &c) in lv.iter().enumerate() {
                let k = ZIGZAG8[s];
                d[k] = dequant8(c, qp, k);
            }
            let r = idct8(&d);
            let (ox, oy) = ((i8 % 2) * 8, (i8 / 2) * 8);
            for k in 0..64 {
                out[(oy + k / 8) * 16 + ox + k % 8] = r[k];
            }
        }
    } else {
        for (blk, lv) in coef.l4.iter().enumerate() {
            let Some(lv) = lv else { continue };
            let mut d = [0i32; 16];
            for (s, &c) in lv.iter().enumerate() {
                let k = ZIGZAG4[s];
                d[k] = dequant4(c, qp, k);
            }
            let r = idct4(&d);
            let (bx, by) = BLK_XY[blk];
            for k in 0..16 {
                out[(by * 4 + k / 4) * 16 + bx * 4 + k % 4] = r[k];
            }
        }
    }
    for comp in 0..2 {
        let qpc = chroma_qp(qp, if comp == 0 { pps.chroma_qp_offset } else { pps.chroma_qp_offset2 });
        // chroma DC: 2x2 Hadamard, then scale (8.5.11)
        let mut dc = [0i32; 4];
        if let Some(c) = &coef.cdc[comp] {
            let c: [i32; 4] = std::array::from_fn(|i| c.get(i).copied().unwrap_or(0));
            let f = [c[0] + c[1] + c[2] + c[3], c[0] - c[1] + c[2] - c[3], c[0] + c[1] - c[2] - c[3], c[0] - c[1] - c[2] + c[3]];
            let ls = level_scale4((qpc % 6) as usize, 0);
            for i in 0..4 {
                dc[i] = ((f[i] * ls) << (qpc / 6)) >> 5;
            }
        }
        let base = 256 + comp * 64;
        for blk in 0..4 {
            let mut d = [0i32; 16];
            d[0] = dc[blk];
            if let Some(lv) = &coef.cac[comp][blk] {
                for (s, &c) in lv.iter().enumerate() {
                    let k = ZIGZAG4[s + 1];
                    d[k] = dequant4(c, qpc, k);
                }
            }
            if d.iter().all(|&v| v == 0) {
                continue;
            }
            let r = idct4(&d);
            let (ox, oy) = ((blk % 2) * 4, (blk / 2) * 4);
            for k in 0..16 {
                out[base + (oy + k / 4) * 8 + ox + k % 4] = r[k];
            }
        }
    }
    Ok(out)
}

/// Luma QP of every macroblock of a slice that starts at slice QP `qp`.
pub fn mb_qps(qp: i32, mbs: &[Mb]) -> Vec<i32> {
    let mut q = qp;
    mbs.iter()
        .map(|mb| {
            q = (q + mb.qpd + 52).rem_euclid(52);
            q
        })
        .collect()
}

/// clip(pred + residual), I_PCM layout.
pub fn add_residual(pred: &[u8; 384], res: &[i32; 384]) -> [u8; 384] {
    std::array::from_fn(|i| clip1(pred[i] as i32 + res[i]) as u8)
}

// ---------------------------------------------------------------------------
// residual corrections
// ---------------------------------------------------------------------------

/// Rows of the 4x4 inverse transform as reals: sample x of basis function i.
const PHI4: [[f64; 4]; 4] = [[1.0, 1.0, 1.0, 1.0], [1.0, 0.5, -0.5, -1.0], [1.0, -1.0, -1.0, 1.0], [0.5, -1.0, 1.0, -0.5]];

fn idct8_1d_f(d: [f64; 8]) -> [f64; 8] {
    let e0 = d[0] + d[4];
    let e1 = -d[3] + d[5] - d[7] - d[7] / 2.0;
    let e2 = d[0] - d[4];
    let e3 = d[1] + d[7] - d[3] - d[3] / 2.0;
    let e4 = d[2] / 2.0 - d[6];
    let e5 = -d[1] + d[7] + d[5] + d[5] / 2.0;
    let e6 = d[2] + d[6] / 2.0;
    let e7 = d[3] + d[5] + d[1] + d[1] / 2.0;
    let (f0, f1, f2, f3) = (e0 + e6, e1 + e7 / 4.0, e2 + e4, e3 + e5 / 4.0);
    let (f4, f5, f6, f7) = (e2 - e4, e3 / 4.0 - e5, e0 - e6, e7 - e1 / 4.0);
    [f0 + f7, f2 + f5, f4 + f3, f6 + f1, f6 - f1, f4 - f3, f2 - f5, f0 - f7]
}

/// Rows of the 8x8 inverse transform as reals.
fn phi8() -> [[f64; 8]; 8] {
    std::array::from_fn(|i| idct8_1d_f(std::array::from_fn(|k| (k == i) as i32 as f64)))
}

/// Least-squares level change of one n x n transform block. `err` is the
/// residual still wanted (target - current reconstruction), row-major;
/// `phi` the basis rows (orthogonal, so the projection on each is the
/// least-squares solution); `step(k)` the real dequantisation step of raster
/// position k. Returns the level change per raster position.
fn project<const N: usize>(err: &[f64], phi: &[[f64; N]; N], step: impl Fn(usize) -> f64, skip_dc: bool) -> Vec<i32> {
    let mut out = vec![0i32; N * N];
    for r in 0..N {
        for c in 0..N {
            let k = r * N + c;
            if skip_dc && k == 0 {
                continue;
            }
            let mut dot = 0.0;
            for y in 0..N {
                for x in 0..N {
                    dot += err[y * N + x] * phi[c][x] * phi[r][y];
                }
            }
            let n2: f64 = phi[c].iter().map(|v| v * v).sum::<f64>() * phi[r].iter().map(|v| v * v).sum::<f64>();
            // a level at k adds level * step * phi_c(x) * phi_r(y) / 64
            out[k] = (dot * 64.0 / (step(k) * n2)).round() as i32;
        }
    }
    out
}

/// Add level changes (scan order) to a block's levels.
fn add_levels(slot: &mut Option<Vec<i32>>, delta: &[i32]) {
    if delta.iter().all(|&v| v == 0) {
        return;
    }
    let v = slot.get_or_insert_with(|| vec![0; delta.len()]);
    if v.len() < delta.len() {
        v.resize(delta.len(), 0);
    }
    for (a, b) in v.iter_mut().zip(delta) {
        *a += b;
    }
}

/// Drop all-zero blocks and recompute coded_block_pattern from the levels.
fn normalise(mb: &mut Mb) {
    let t8 = mb.t8;
    let Some(c) = mb.coef.as_mut() else {
        mb.cbp = 0;
        return;
    };
    let nz = |s: &mut Option<Vec<i32>>| {
        if s.as_ref().is_some_and(|v| v.iter().all(|&x| x == 0)) {
            *s = None;
        }
        s.is_some()
    };
    let mut luma = 0u8;
    for i8 in 0..4 {
        let any = if t8 {
            nz(&mut c.l8[i8])
        } else {
            let mut a = false;
            for i4 in 0..4 {
                a |= nz(&mut c.l4[i8 * 4 + i4]);
            }
            a
        };
        if any {
            luma |= 1 << i8;
        }
    }
    let (mut dc, mut ac) = (false, false);
    for comp in 0..2 {
        dc |= nz(&mut c.cdc[comp]);
        for b in 0..4 {
            ac |= nz(&mut c.cac[comp][b]);
        }
    }
    let chroma = if ac {
        2
    } else if dc {
        1
    } else {
        0
    };
    mb.cbp = luma | (chroma << 4);
    if luma == 0 {
        mb.t8 = false;
    }
}

/// A residual correction of an inter macroblock: the record with its
/// levels changed, and the exact pre-deblocking reconstruction it gives.
#[derive(Clone, Debug)]
pub struct Correction {
    pub mb: Mb,
    pub recon: [u8; 384],
    /// Max |recon - target| over the macroblock.
    pub err: u8,
    /// The luma QP the levels are coded at.
    pub qp: i32,
}

/// QP reductions tried in turn by [`correct`]. A reduction by 6k codes the
/// original luma levels times 2^k, which reproduces their residual exactly
/// (see the `qp_minus_6_doubles_exactly` test), and halves the quantiser step
/// of the correction k times.
pub const QP_DROPS: [i32; 4] = [0, 6, 12, 18];

/// Change the levels of inter macroblock `mb` (luma QP `qp`) so that its
/// reconstruction over prediction `pred` comes as close as the quantiser
/// allows to `target` (both in the I_PCM layout). Motion is kept; a skipped
/// MB becomes an explicit P_L0_16x16 (its mvd is left for `patch_pictures`
/// to solve) or B_Direct_16x16. The transform size is kept when the MB codes
/// luma, else 4x4 (which also keeps the deblocked edges). The QP is kept if
/// that gets within `max_err`; otherwise it is lowered by [`QP_DROPS`] until
/// one does. Returns the first good enough correction, else the closest.
pub fn correct(mb: &Mb, qp: i32, pps: &Pps, pred: &[u8; 384], target: &[u8; 384], max_err: u8) -> Result<Correction> {
    use crate::records::{Kind, Mvds, Part};
    if mb.intra {
        bail!("correct: intra MB ({}, {})", mb.x, mb.y);
    }
    let mut n = mb.clone();
    if mb.skip {
        n.skip = false;
        n.cbp = 0;
        n.t8 = false;
        n.qpd = 0;
        n.coef = Some(Box::default());
        if mb.direct16 {
            n.kind = Kind::B;
            n.btype = Some(0);
            n.part = Some(Part::Direct);
        } else {
            n.kind = Kind::P;
            n.part = Some(Part::P16x16);
            n.refs = Some([vec![Some(0)], vec![None]]);
            n.mvds = Some(Mvds::Parts([vec![Some([0, 0])], vec![None]]));
            n.direct16 = false;
        }
    }
    if n.coef.is_none() {
        n.coef = Some(Box::default());
    }
    n.t8 = n.t8 && (n.cbp & 15) != 0;
    let mut best: Option<Correction> = None;
    for drop in QP_DROPS {
        let q = qp - drop;
        if q < 0 {
            break;
        }
        let mut c = solve(requantise(&n, qp, q, pps), q, pps, pred, target)?;
        if c.err > max_err {
            c = refine(c, pps, pred, target)?;
        }
        let done = c.err <= max_err;
        if best.as_ref().is_none_or(|b| c.err < b.err) {
            best = Some(c);
        }
        if done {
            break;
        }
    }
    Ok(best.expect("QP 0 at least"))
}

/// Residual of one luma block from its levels (scan order): 4x4 or 8x8.
fn luma_block(levels: Option<&Vec<i32>>, t8: bool, qp: i32) -> Vec<i32> {
    if t8 {
        let mut d = [0i32; 64];
        for (s, &c) in levels.into_iter().flatten().enumerate() {
            d[ZIGZAG8[s]] = dequant8(c, qp, ZIGZAG8[s]);
        }
        idct8(&d).to_vec()
    } else {
        let mut d = [0i32; 16];
        for (s, &c) in levels.into_iter().flatten().enumerate() {
            d[ZIGZAG4[s]] = dequant4(c, qp, ZIGZAG4[s]);
        }
        idct4(&d).to_vec()
    }
}

/// (max |error|, samples at the max, sum of squares): smaller is better.
fn score(err: impl Iterator<Item = i32>) -> (i32, usize, i64) {
    let (mut m, mut n, mut ss) = (0, 0, 0i64);
    for e in err {
        let a = e.abs();
        if a > m {
            (m, n) = (a, 1);
        } else if a == m {
            n += 1;
        }
        ss += (e * e) as i64;
    }
    (m, n, ss)
}

/// Greedy max-error refinement of a least-squares correction: change one
/// level by +-1 in the block holding the worst sample, as long as that makes
/// the worst error (then the number of samples at it, then the squared
/// error) smaller.
fn refine(c: Correction, pps: &Pps, pred: &[u8; 384], target: &[u8; 384]) -> Result<Correction> {
    const ITERS: usize = 16;
    let qp = c.qp;
    let mut n = c.mb;
    if n.coef.is_none() {
        n.coef = Some(Box::default());
    }
    for _ in 0..ITERS {
        let res = residual(&n, qp, pps)?;
        let e: Vec<i32> = (0..384).map(|i| target[i] as i32 - clip1(pred[i] as i32 + res[i])).collect();
        let cur = score(e.iter().copied());
        if cur.0 <= 1 {
            break;
        }
        let s = (0..384).max_by_key(|&i| (e[i].abs(), std::cmp::Reverse(i))).expect("384 samples");
        let mut best: Option<((i32, usize, i64), Mb)> = None;
        if s < 256 {
            let (x, y) = (s % 16, s / 16);
            let t8 = n.t8;
            // the block's samples (MB offsets) and its level slot
            let (cells, slot): (Vec<usize>, usize) = if t8 {
                let i8 = (y / 8) * 2 + x / 8;
                ((0..64).map(|k| ((i8 / 2) * 8 + k / 8) * 16 + (i8 % 2) * 8 + k % 8).collect(), i8)
            } else {
                let blk = BLK_XY.iter().position(|&b| b == (x / 4, y / 4)).expect("luma block");
                let (bx, by) = BLK_XY[blk];
                ((0..16).map(|k| (by * 4 + k / 4) * 16 + bx * 4 + k % 4).collect(), blk)
            };
            let outside = {
                let mut v = e.clone();
                for &i in &cells {
                    v[i] = 0;
                }
                v
            };
            let coef = n.coef.as_ref().expect("set above");
            let levels = if t8 { coef.l8[slot].clone() } else { coef.l4[slot].clone() };
            let len = if t8 { 64 } else { 16 };
            for p in 0..len {
                for dir in [1, -1] {
                    let mut l = levels.clone().unwrap_or_else(|| vec![0; len]);
                    l.resize(len, 0);
                    l[p] += dir;
                    let r = luma_block(Some(&l), t8, qp);
                    let mut v = outside.clone();
                    for (k, &i) in cells.iter().enumerate() {
                        v[i] = target[i] as i32 - clip1(pred[i] as i32 + r[k]);
                    }
                    let sc = score(v.into_iter());
                    if sc < cur && best.as_ref().is_none_or(|b| sc < b.0) {
                        let mut m = n.clone();
                        let cf = m.coef.as_mut().expect("set above");
                        if t8 {
                            cf.l8[slot] = Some(l);
                        } else {
                            cf.l4[slot] = Some(l);
                        }
                        best = Some((sc, m));
                    }
                }
            }
        } else {
            // chroma: the block's AC levels and the component's DC levels
            let comp = (s - 256) / 64;
            let k = (s - 256) % 64;
            let blk = (k / 8 / 4) * 2 + (k % 8) / 4;
            for slot in 0..19 {
                for dir in [1, -1] {
                    let mut m = n.clone();
                    let cf = m.coef.as_mut().expect("set above");
                    if slot < 4 {
                        let v = cf.cdc[comp].get_or_insert_with(|| vec![0; 4]);
                        v.resize(4, 0);
                        v[slot] += dir;
                    } else {
                        let v = cf.cac[comp][blk].get_or_insert_with(|| vec![0; 15]);
                        v.resize(15, 0);
                        v[slot - 4] += dir;
                    }
                    let r = residual(&m, qp, pps)?;
                    let sc = score((0..384).map(|i| target[i] as i32 - clip1(pred[i] as i32 + r[i])));
                    if sc < cur && best.as_ref().is_none_or(|b| sc < b.0) {
                        best = Some((sc, m));
                    }
                }
            }
        }
        match best {
            Some((_, m)) => n = m,
            None => break,
        }
    }
    normalise(&mut n);
    let recon = add_residual(pred, &residual(&n, qp, pps)?);
    let err = recon.iter().zip(target).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
    Ok(Correction { mb: n, recon, err, qp })
}

/// The record `mb` (levels at luma QP `from`) re-expressed at QP `to` <=
/// `from`: luma levels scaled by 2^((from - to) / 6) (exact when `from - to`
/// is a multiple of 6), chroma levels by 2^(difference of chroma QPs / 6),
/// rounded when that is not a whole power.
fn requantise(mb: &Mb, from: i32, to: i32, pps: &Pps) -> Mb {
    let mut n = mb.clone();
    if from == to {
        return n;
    }
    let scale = |v: &mut Option<Vec<i32>>, d: i32| {
        if let Some(v) = v {
            for x in v.iter_mut() {
                *x = if d % 6 == 0 { *x << (d / 6) } else { (*x as f64 * 2f64.powf(d as f64 / 6.0)).round() as i32 };
            }
        }
    };
    if let Some(c) = n.coef.as_mut() {
        let dl = from - to;
        c.l8.iter_mut().chain(c.l4.iter_mut()).for_each(|v| scale(v, dl));
        for comp in 0..2 {
            let off = if comp == 0 { pps.chroma_qp_offset } else { pps.chroma_qp_offset2 };
            let dc = chroma_qp(from, off) - chroma_qp(to, off);
            scale(&mut c.cdc[comp], dc);
            c.cac[comp].iter_mut().for_each(|v| scale(v, dc));
        }
    }
    n
}

/// Least-squares level changes at a fixed QP, a few times over (the
/// residual is not quite linear in the levels).
fn solve(mut n: Mb, qp: i32, pps: &Pps, pred: &[u8; 384], target: &[u8; 384]) -> Result<Correction> {
    let phi8 = phi8();
    let qpc = [chroma_qp(qp, pps.chroma_qp_offset), chroma_qp(qp, pps.chroma_qp_offset2)];
    const H: [[f64; 4]; 4] = [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]];
    let mut best: Option<Correction> = None;
    for _ in 0..3 {
        let res = residual(&n, qp, pps)?;
        let recon = add_residual(pred, &res);
        let err = recon.iter().zip(target).map(|(a, b)| a.abs_diff(*b)).max().unwrap_or(0);
        if best.as_ref().is_none_or(|b| err < b.err) {
            best = Some(Correction { mb: n.clone(), recon, err, qp });
        }
        if err == 0 {
            break;
        }
        let e: Vec<f64> = (0..384).map(|i| target[i] as f64 - recon[i] as f64).collect();
        let t8 = n.t8;
        let coef = n.coef.as_mut().expect("set above");
        let mut changed = false;
        if t8 {
            for i8 in 0..4 {
                let (ox, oy) = ((i8 % 2) * 8, (i8 / 2) * 8);
                let blk: Vec<f64> = (0..64).map(|k| e[(oy + k / 8) * 16 + ox + k % 8]).collect();
                let d = project::<8>(&blk, &phi8, |k| level_scale8((qp % 6) as usize, k) as f64 * 2f64.powi(qp / 6) / 64.0, false);
                let scan: Vec<i32> = (0..64).map(|s| d[ZIGZAG8[s]]).collect();
                changed |= scan.iter().any(|&v| v != 0);
                add_levels(&mut coef.l8[i8], &scan);
            }
        } else {
            for blk in 0..16 {
                let (bx, by) = BLK_XY[blk];
                let b: Vec<f64> = (0..16).map(|k| e[(by * 4 + k / 4) * 16 + bx * 4 + k % 4]).collect();
                let d = project::<4>(&b, &PHI4, |k| level_scale4((qp % 6) as usize, k) as f64 * 2f64.powi(qp / 6) / 16.0, false);
                let scan: Vec<i32> = (0..16).map(|s| d[ZIGZAG4[s]]).collect();
                changed |= scan.iter().any(|&v| v != 0);
                add_levels(&mut coef.l4[blk], &scan);
            }
        }
        // chroma: DC through the 2x2 Hadamard, AC per 4x4 block
        for comp in 0..2 {
            let q = qpc[comp];
            let base = 256 + comp * 64;
            let blocks: Vec<Vec<f64>> =
                (0..4).map(|b| (0..16).map(|k| e[base + ((b / 2) * 4 + k / 4) * 8 + (b % 2) * 4 + k % 4]).collect()).collect();
            // a DC level j adds H[b][j] * sdc / 64 to every sample of block b
            let sdc = level_scale4((q % 6) as usize, 0) as f64 * 2f64.powi(q / 6) / 32.0;
            let sums: Vec<f64> = blocks.iter().map(|b| b.iter().sum()).collect();
            let ddc: Vec<i32> = (0..4).map(|j| ((0..4).map(|b| H[b][j] * sums[b]).sum::<f64>() / sdc).round() as i32).collect();
            changed |= ddc.iter().any(|&v| v != 0);
            add_levels(&mut coef.cdc[comp], &ddc);
            for (b, blk) in blocks.iter().enumerate() {
                let d = project::<4>(blk, &PHI4, |k| level_scale4((q % 6) as usize, k) as f64 * 2f64.powi(q / 6) / 16.0, true);
                let scan: Vec<i32> = (1..16).map(|s| d[ZIGZAG4[s]]).collect();
                changed |= scan.iter().any(|&v| v != 0);
                add_levels(&mut coef.cac[comp][b], &scan);
            }
        }
        normalise(&mut n);
        if !changed {
            break;
        }
    }
    Ok(best.expect("at least one round"))
}

// ---------------------------------------------------------------------------
// dev recon
// ---------------------------------------------------------------------------

/// A category of inter MB for the check report.
fn category(pps: &Pps, pic: &Picture, mb: &Mb, m: &Motion) -> String {
    let t = pic.typ();
    let (gx, gy) = (mb.x as usize * 4, mb.y as usize * 4);
    let mut bi = false;
    for by in 0..4 {
        for bx in 0..4 {
            let i = m.idx(gx + bx, gy + by);
            bi |= m.refidx[0][i] >= 0 && m.refidx[1][i] >= 0;
        }
    }
    let wp = match t {
        SliceType::P if pps.weighted_pred => "wp",
        SliceType::B if pps.weighted_bipred_idc == 1 => "wp",
        SliceType::B if pps.weighted_bipred_idc == 2 && bi => "implicit",
        _ => "",
    };
    let kind = if mb.skip {
        "skip"
    } else if mb.direct16 {
        "direct"
    } else {
        "inter"
    };
    let res = if mb.cbp == 0 {
        "nores"
    } else if mb.t8 {
        "t8"
    } else {
        "t4"
    };
    format!("{}/{kind}/{}{}/{res}", t.as_str(), if bi { "bi" } else { "uni" }, if wp.is_empty() { String::new() } else { format!("/{wp}") })
}

/// `frame-dedup dev recon <file.h264> [--periods A-B] [--show N]`: decode
/// IDR periods with ffmpeg's deblocking switched off (`-skip_loop_filter
/// all`), so every picture, references included, is the pre-deblocking
/// reconstruction. Then reconstruct every inter macroblock from those frames
/// with [`predict_mb`] + [`residual`] and compare bit for bit. Prints counts
/// per category and the first mismatches.
pub fn dev_recon(args: &[String]) -> Result<()> {
    const USAGE: &str = "frame-dedup dev recon <file.h264> [--periods A-B] [--show N] [--deblocked]";
    let (mut path, mut periods, mut show, mut deblocked) = (None, None, 10usize, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--periods" => {
                let v = it.next().ok_or_else(|| anyhow!("--periods needs a value"))?;
                let (a, b) = v.split_once('-').unwrap_or((v, v));
                periods = Some((a.parse::<usize>()?, b.parse::<usize>()?));
            }
            "--show" => show = it.next().ok_or_else(|| anyhow!("--show needs a value"))?.parse()?,
            // negative control: compare against the normal (deblocked) decode
            "--deblocked" => deblocked = true,
            _ if path.is_none() => path = Some(a.clone()),
            _ => bail!("usage: {USAGE}"),
        }
    }
    let path = path.ok_or_else(|| anyhow!("usage: {USAGE}"))?;
    let path = Path::new(&path);
    crate::system::lower_priority();
    let (spss, ppss, pics) = crate::stream::load(path)?;
    let file = std::fs::File::open(path)?;
    let es = unsafe { Mmap::map(&file)? };
    let sps = spss.get(&0).ok_or_else(|| anyhow!("no SPS 0"))?;
    for p in ppss.values() {
        check_scope(sps, p, &pics)?;
    }
    let (w, h) = (sps.mb_w, sps.mb_h);
    let fs = frame_size(sps);
    let ranges = period_ranges(&pics);
    let (p0, p1) = periods.unwrap_or((0, ranges.len() - 1));
    let mut stats: BTreeMap<String, (usize, usize, u32)> = BTreeMap::new();
    let mut shown = 0usize;
    for g in p0..=p1.min(ranges.len() - 1) {
        let r = ranges[g].clone();
        let mut by_poc: Vec<usize> = r.clone().collect();
        by_poc.sort_by_key(|&d| pics[d].poc);
        let mut frames: HashMap<usize, Vec<u8>> = HashMap::new();
        let dec_it = iter_frames_with(path, &[(pics[r.start].au_start, pics[r.end - 1].au_end)], fs, FFMPEG_THREADS, if deblocked { &[] } else { &["-skip_loop_filter", "all"] })?;
        for (d, fr) in by_poc.iter().zip(dec_it) {
            frames.insert(*d, fr?);
        }
        if frames.len() != r.len() {
            bail!("period {g}: decoded {} of {} pictures", frames.len(), r.len());
        }
        let mut motions: HashMap<usize, Motion> = HashMap::new();
        for d in r.clone() {
            let pic = &pics[d];
            let pps = ppss.get(&pic.sl().pps_id).ok_or_else(|| anyhow!("no PPS"))?;
            let dd = decode_picture(&es, &spss, &ppss, pic).with_context(|| format!("picture {d}"))?;
            let col = if pic.typ() == SliceType::B {
                let c = *pic.lists[1].first().ok_or_else(|| anyhow!("B picture {d} without list 1"))?;
                Some(motions.get(&c).ok_or_else(|| anyhow!("co-located picture {c} of {d} not in the period"))?)
            } else {
                None
            };
            let m = derive_motion(pic, &dd.mbs, w, h, &pics, col).with_context(|| format!("motion of picture {d}"))?;
            let qps = mb_qps(dd.sl.qp, &dd.mbs);
            let cur = Frame::new(&frames[&d], sps);
            let refs = |dec: usize| frames.get(&dec).map(|f| Frame::new(f, sps));
            for (i, mb) in dd.mbs.iter().enumerate() {
                if mb.intra {
                    continue;
                }
                let (mx, my) = (mb.x as usize, mb.y as usize);
                let pred = predict_mb(pps, pic, &pics, &m, mx, my, &refs)?;
                let res = residual(mb, qps[i], pps)?;
                let got = add_residual(&pred, &res);
                let want = cur.mb(mx, my);
                let worst = got.iter().zip(want.iter()).map(|(a, b)| a.abs_diff(*b) as u32).max().unwrap_or(0);
                let cat = category(pps, pic, mb, &m);
                let e = stats.entry(cat.clone()).or_default();
                e.0 += 1;
                if worst != 0 {
                    e.1 += 1;
                    e.2 = e.2.max(worst);
                    if shown < show {
                        shown += 1;
                        let first = (0..384).find(|&k| got[k] != want[k]).unwrap_or(0);
                        println!(
                            "mismatch: period {g} picture {d} (POC {}) MB ({mx}, {my}) {cat} qp {}: worst {worst}, first at sample {first} (got {}, want {}, pred {}, res {})",
                            pic.poc, qps[i], got[first], want[first], pred[first], res[first]
                        );
                    }
                }
            }
            motions.insert(d, m);
        }
        let (n, bad): (usize, usize) = stats.values().fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));
        eprintln!("period {g}: {} pictures; total so far {n} inter MBs, {bad} mismatches", r.len());
    }
    println!("{:<34} {:>9} {:>9} {:>6}", "category", "MBs", "mismatch", "worst");
    for (k, (n, bad, worst)) in &stats {
        println!("{k:<34} {n:>9} {bad:>9} {worst:>6}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_weights_cases() {
        // B halfway between its references: equal weights
        assert_eq!(implicit_weights(2, 0, 4), (32, 32));
        // a quarter of the way: 3:1 towards the nearer reference
        assert_eq!(implicit_weights(1, 0, 4), (48, 16));
        // same POC: default
        assert_eq!(implicit_weights(1, 4, 4), (32, 32));
    }

    #[test]
    fn idct4_dc_only() {
        let mut d = [0; 16];
        d[0] = 64 * 5;
        assert!(idct4(&d).iter().all(|&v| v == 5));
    }

    #[test]
    fn idct8_dc_only() {
        let mut d = [0; 64];
        d[0] = 64 * 3;
        assert!(idct8(&d).iter().all(|&v| v == 3));
    }

    fn test_pps() -> Pps {
        Pps {
            id: 0,
            sps_id: 0,
            cabac: true,
            bottom_field_poc_present: false,
            num_slice_groups: 1,
            num_ref_idx_l0_default: 1,
            num_ref_idx_l1_default: 1,
            weighted_pred: false,
            weighted_bipred_idc: 0,
            pic_init_qp: 26,
            chroma_qp_offset: -3,
            deblocking_ctrl_present: true,
            constrained_intra: false,
            redundant_pic_cnt_present: false,
            transform_8x8: true,
            scaling_matrix: false,
            chroma_qp_offset2: -3,
        }
    }

    /// A correction of drift by noise: within the acceptance error at some
    /// QP of the ladder, the original QP for a flat offset.
    #[test]
    fn correct_reaches_target() {
        let pps = test_pps();
        let mut mb = Mb::new(0, 0);
        mb.kind = crate::records::Kind::P;
        mb.skip = true;
        let mut s = 12345u32;
        let mut rnd = || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (s >> 16) as i32
        };
        let pred: [u8; 384] = std::array::from_fn(|i| (100 + (i as i32 % 16) * 3 + rnd() % 5) as u8);
        // flat offset: the original QP is enough
        let target: [u8; 384] = std::array::from_fn(|i| pred[i] + 6);
        let c = correct(&mb, 30, &pps, &pred, &target, 3).unwrap();
        assert!(c.err <= 3, "flat: err {}", c.err);
        assert_eq!(c.qp, 30);
        assert!(!c.mb.skip && c.mb.cbp != 0);
        // noise: needs a lower QP
        let target: [u8; 384] = std::array::from_fn(|i| (pred[i] as i32 + rnd() % 7 - 3).clamp(0, 255) as u8);
        let c = correct(&mb, 30, &pps, &pred, &target, 3).unwrap();
        assert!(c.err <= 3, "noise: err {} at qp {}", c.err, c.qp);
        // the reconstruction it reports is the real one
        assert_eq!(c.recon, add_residual(&pred, &residual(&c.mb, c.qp, &pps).unwrap()));
    }

    /// Lowering QP by 6 and doubling the levels reproduces the dequantised
    /// coefficients exactly (both rounding branches and the crossover).
    #[test]
    fn qp_minus_6_doubles_exactly() {
        for qp in 6..52 {
            for k in 0..16 {
                for c in [-7, -1, 1, 3, 40] {
                    assert_eq!(dequant4(c, qp, k), dequant4(2 * c, qp - 6, k), "4x4 qp {qp} k {k} c {c}");
                }
            }
            for k in 0..64 {
                for c in [-7, -1, 1, 3, 40] {
                    assert_eq!(dequant8(c, qp, k), dequant8(2 * c, qp - 6, k), "8x8 qp {qp} k {k} c {c}");
                }
            }
        }
    }
}

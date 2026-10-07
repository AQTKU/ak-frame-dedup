//! Decode the output and the original and compare every kept frame.
//! Port of `frame_dedup/verify.py`.
//!
//! So far only the pixel helpers; they also include `repair.visible_maxdiff`.

use crate::h264hdr::Sps;

/// Max absolute difference over the bytes `a` and `b` (equal length).
#[inline]
fn max_absdiff(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).fold(0u8, |m, (&x, &y)| m.max(x.abs_diff(y)))
}

/// Same for exactly `N` bytes (lets the compiler unroll / vectorise).
#[inline]
fn max_absdiff_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u8 {
    let mut m = 0u8;
    for i in 0..N {
        m = m.max(a[i].abs_diff(b[i]));
    }
    m
}

/// Per-block max of |a - b| over a plane `w` wide in `BS` x `BS` blocks,
/// restricted to rows `y0..y1` and columns `x0..x1` (everything outside counts
/// as zero difference). Writes `max` into `out[(h / BS) * (w / BS)]`.
fn plane_blocks<const BS: usize>(a: &[u8], b: &[u8], w: usize, vis: (usize, usize, usize, usize), out: &mut [u8]) {
    let bw = w / BS;
    let (x0, x1, y0, y1) = vis;
    for y in y0..y1 {
        let ra = &a[y * w..(y + 1) * w];
        let rb = &b[y * w..(y + 1) * w];
        let orow = &mut out[(y / BS) * bw..(y / BS + 1) * bw];
        if x0 == 0 && x1 == w {
            for ((o, ca), cb) in orow.iter_mut().zip(ra.chunks_exact(BS)).zip(rb.chunks_exact(BS)) {
                let m = max_absdiff_n::<BS>(ca.try_into().unwrap(), cb.try_into().unwrap());
                *o = (*o).max(m);
            }
        } else {
            for (bx, o) in orow.iter_mut().enumerate() {
                let lo = (bx * BS).max(x0);
                let hi = ((bx + 1) * BS).min(x1);
                if lo < hi {
                    *o = (*o).max(max_absdiff(&ra[lo..hi], &rb[lo..hi]));
                }
            }
        }
    }
}

fn check(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize) -> (usize, usize) {
    let (w, h) = (mb_w * 16, mb_h * 16);
    assert!(a.len() == w * h * 3 / 2 && b.len() == a.len(), "frame size mismatch");
    (w, h)
}

/// Per-macroblock max absolute difference over Y, U and V, plus the luma-only
/// map. Both are row-major `mb_h` x `mb_w` arrays. Mirrors `verify.mb_maxdiff`.
pub fn mb_maxdiff(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = check(a, b, mb_w, mb_h);
    let n = mb_w * mb_h;
    let mut ly = vec![0u8; n];
    plane_blocks::<16>(&a[..w * h], &b[..w * h], w, (0, w, 0, h), &mut ly);
    let mut all = ly.clone();
    let (cw, ch) = (w / 2, h / 2);
    let (u0, v0) = (w * h, w * h * 5 / 4);
    for (o, e) in [(u0, v0), (v0, a.len())] {
        plane_blocks::<8>(&a[o..e], &b[o..e], cw, (0, cw, 0, ch), &mut all);
    }
    (all, ly)
}

/// Per-MB max absolute difference over Y, U and V counting only the pixels
/// inside the SPS cropping window. Mirrors `repair.visible_maxdiff`.
pub fn visible_maxdiff(a: &[u8], b: &[u8], sps: &Sps) -> Vec<u8> {
    let (mb_w, mb_h) = (sps.mb_w, sps.mb_h);
    let (w, h) = check(a, b, mb_w, mb_h);
    // Python doubles the SPS crop offsets (4:2:0): left, right, top, bottom in luma pixels
    let [cl, cr, ct, cb] = sps.crop.map(|c| 2 * c as usize);
    let mut out = vec![0u8; mb_w * mb_h];
    let vis = |s: usize, pw: usize, ph: usize| (cl / s, pw - cr / s, ct / s, ph - cb / s);
    plane_blocks::<16>(&a[..w * h], &b[..w * h], w, vis(1, w, h), &mut out);
    let (cw, ch) = (w / 2, h / 2);
    let (u0, v0) = (w * h, w * h * 5 / 4);
    for (o, e) in [(u0, v0), (v0, a.len())] {
        plane_blocks::<8>(&a[o..e], &b[o..e], cw, vis(2, cw, ch), &mut out);
    }
    out
}

/// `frame-dedup dev maxdiff <file.h264> <mb_w> <mb_h> <cl> <cr> <ct> <cb> <i> <j>`: decode frames
/// `0..=max(i, j)` of the file (from its start, which must be an IDR) and print the
/// canonical JSON of the macroblock difference maps between frames `i` and `j`.
pub fn dev_maxdiff(args: &[String]) -> anyhow::Result<()> {
    use anyhow::anyhow;
    let n = |i: usize| -> anyhow::Result<usize> { Ok(args.get(i).ok_or_else(|| anyhow!("missing argument {i}"))?.parse()?) };
    let path = std::path::PathBuf::from(args.first().ok_or_else(|| anyhow!("missing file"))?);
    let (mb_w, mb_h) = (n(1)?, n(2)?);
    let crop = [n(3)? as u32, n(4)? as u32, n(5)? as u32, n(6)? as u32];
    let (i, j) = (n(7)?, n(8)?);
    let fs = mb_w * mb_h * 384;
    let len = std::fs::metadata(&path)?.len() as usize;
    let mut frames = Vec::new();
    for (k, f) in crate::decode::iter_frames(&path, &[(0, len)], fs, crate::system::FFMPEG_THREADS)?.enumerate() {
        frames.push(f?);
        if k == i.max(j) {
            break;
        }
    }
    let (a, b) = (frames.get(i).ok_or_else(|| anyhow!("no frame {i}"))?, frames.get(j).ok_or_else(|| anyhow!("no frame {j}"))?);
    let sps = Sps { mb_w, mb_h, crop, ..test_sps_base() };
    let (m, ly) = mb_maxdiff(a, b, mb_w, mb_h);
    let vis = visible_maxdiff(a, b, &sps);
    println!("{}", serde_json::json!({"mb": m, "luma": ly, "visible": vis}));
    Ok(())
}

fn test_sps_base() -> Sps {
    Sps {
        profile_idc: 100,
        level_idc: 40,
        id: 0,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        log2_max_frame_num: 4,
        poc_type: 0,
        log2_max_poc_lsb: 4,
        delta_pic_order_always_zero: false,
        max_num_ref_frames: 1,
        gaps_allowed: false,
        mb_w: 0,
        mb_h: 0,
        frame_mbs_only: true,
        mbaff: false,
        direct_8x8_inference: true,
        crop: [0; 4],
        vui: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(mb_w: usize, mb_h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..mb_w * mb_h * 384)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Straightforward reference: the numpy algorithm, element by element.
    fn naive(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize, crop: [usize; 4]) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (mb_w * 16, mb_h * 16);
        let mut all = vec![0u8; mb_w * mb_h];
        let mut luma = all.clone();
        let planes = [(0, w, h, 1usize, 16usize), (w * h, w / 2, h / 2, 2, 8), (w * h * 5 / 4, w / 2, h / 2, 2, 8)];
        for (pi, (off, pw, ph, s, bs)) in planes.into_iter().enumerate() {
            for y in 0..ph {
                for x in 0..pw {
                    if y < crop[2] / s || y >= ph - crop[3] / s || x < crop[0] / s || x >= pw - crop[1] / s {
                        continue;
                    }
                    let d = a[off + y * pw + x].abs_diff(b[off + y * pw + x]);
                    let i = (y / bs) * mb_w + x / bs;
                    all[i] = all[i].max(d);
                    if pi == 0 {
                        luma[i] = luma[i].max(d);
                    }
                }
            }
        }
        (all, luma)
    }

    /// `cargo test --profile fastrel -- --ignored --nocapture timing`
    #[test]
    #[ignore]
    fn timing() {
        let (mw, mh) = (120, 68);
        let (a, b) = (frame(mw, mh, 1), frame(mw, mh, 2));
        let sps = test_sps(mw, mh, [0, 0, 0, 4]);
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(mb_maxdiff(&a, &b, mw, mh));
        }
        println!("mb_maxdiff: {:?} per 1080p frame", t.elapsed() / 100);
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(visible_maxdiff(&a, &b, &sps));
        }
        println!("visible_maxdiff: {:?} per 1080p frame", t.elapsed() / 100);
    }

    #[test]
    fn matches_naive() {
        let (mw, mh) = (5, 4);
        let (a, b) = (frame(mw, mh, 1), frame(mw, mh, 2));
        let (all, ly) = mb_maxdiff(&a, &b, mw, mh);
        assert_eq!((all, ly), naive(&a, &b, mw, mh, [0; 4]));
        for crop in [[0, 0, 0, 4], [1, 2, 3, 4], [0, 3, 0, 0], [8, 0, 8, 0]] {
            let sps = test_sps(mw, mh, crop.map(|c| c as u32));
            let want = naive(&a, &b, mw, mh, crop.map(|c| 2 * c)).0;
            assert_eq!(visible_maxdiff(&a, &b, &sps), want, "crop {crop:?}");
        }
    }

    fn test_sps(mb_w: usize, mb_h: usize, crop: [u32; 4]) -> Sps {
        Sps { mb_w, mb_h, crop, ..test_sps_base() }
    }
}

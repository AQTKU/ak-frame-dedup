//! Re-expressing macroblocks with explicit motion. Port of
//! `verify_motion.reexpress`, `reexpress_compact` and `has_direct`, plus the
//! motion re-expression test (`verify_motion.main`) as `dev reexpress`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::decode::{iter_frames, md5_hex};
use crate::h264hdr::{Picture, Ppss, SliceType, Spss};
use crate::mv::{Motion, derive_motion, derive_motion_target};
use crate::patch::segment_bytes;
use crate::picture::{Parsed, decode_picture, hex, sha256, try_encode_picture};
use crate::records::{B_SUB, Coefs, Kind, Mb, Mv, Mvds, P_SUB, Part, Pred, SubShape, b_type_index, canonical_text};
use crate::system::FFMPEG_THREADS;

/// `B_SUB_FOR`: the B sub-macroblock type of an 8x8 block per prediction.
fn b_sub_for(pred: Pred) -> u8 {
    match pred {
        Pred::L0 => 1,
        Pred::L1 => 2,
        Pred::Bi => 3,
    }
}

/// Number of mvds of a sub-macroblock type.
fn nparts(t: SliceType, st: u8) -> usize {
    let shape = if t == SliceType::P { P_SUB[st as usize] } else { B_SUB[st as usize].0 };
    match shape {
        SubShape::S8x8 => 1,
        SubShape::S8x4 | SubShape::S4x8 => 2,
        SubShape::S4x4 => 4,
        SubShape::Direct => unreachable!("direct sub-macroblock has no mvds"),
    }
}

/// What a re-expressed skip MB looks like: no residual, no qp delta.
pub(crate) fn clear_residual(n: &mut Mb) {
    n.cbp = 0;
    n.t8 = false;
    n.qpd = 0;
    n.coef = Some(Box::<Coefs>::default());
}

/// An equivalent record with explicit 8x8 partitions carrying `motion`.
/// Unused lists get reference index -1 (as in the motion field) and (0, 0)
/// mvds; the encoder never reads them.
pub fn reexpress(mb: &Mb, t: SliceType, motion: &Motion) -> Mb {
    if mb.intra {
        return mb.clone();
    }
    let mut n = mb.clone();
    let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
    let mut quad_sub = [0u8; 4];
    for (i, qs) in quad_sub.iter_mut().enumerate() {
        let (gx, gy) = (x0 + (i % 2) * 2, y0 + (i / 2) * 2);
        let k = motion.idx(gx, gy);
        let use0 = motion.refidx[0][k] >= 0;
        let use1 = motion.refidx[1][k] >= 0;
        let pred = if use0 && use1 {
            Pred::Bi
        } else if use0 {
            Pred::L0
        } else {
            Pred::L1
        };
        *qs = if mb.part == Some(Part::P8x8) && motion.direct[k] == 0 {
            mb.sub.expect("an 8x8 macroblock has sub_mb_types")[i]
        } else if t == SliceType::P {
            0
        } else {
            b_sub_for(pred)
        };
    }
    n.skip = false;
    n.kind = if t == SliceType::P { Kind::P } else { Kind::B };
    n.part = Some(Part::P8x8);
    n.sub = Some(quad_sub);
    n.direct16 = false;
    if t == SliceType::B {
        n.btype = Some(22);
    }
    let mut refs: [Vec<Option<i8>>; 2] = [Vec::with_capacity(4), Vec::with_capacity(4)];
    for (l, r) in refs.iter_mut().enumerate() {
        for i in 0..4 {
            r.push(Some(motion.refidx[l][motion.idx(x0 + (i % 2) * 2, y0 + (i / 2) * 2)]));
        }
    }
    n.refs = Some(refs);
    let mvds_of = || -> [Vec<Mv>; 4] { std::array::from_fn(|i| vec![[0, 0]; nparts(t, quad_sub[i])]) };
    n.mvds = Some(Mvds::Sub([mvds_of(), mvds_of()]));
    if mb.skip {
        clear_residual(&mut n);
    }
    n
}

/// An explicit B record for a skip / direct MB using the coarsest
/// partitioning that fits: direct motion is uniform per 8x8 quadrant
/// (direct_8x8_inference), so 16x16, 16x8 or 8x16 often suffice and cost far
/// fewer mvds than 8x8. mvds are left for target mode to solve. A B_8x8 MB
/// with direct sub-blocks goes through [`reexpress`].
pub fn reexpress_compact(mb: &Mb, motion: &Motion) -> Mb {
    if mb.part == Some(Part::P8x8) && !mb.skip {
        return reexpress(mb, SliceType::B, motion);
    }
    let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
    // per quadrant: for each list, (reference index, mv if the list is used)
    type Quad = [(i32, Option<[i16; 2]>); 2];
    let quad = |i: usize| -> Quad {
        let k = motion.idx(x0 + (i % 2) * 2, y0 + (i / 2) * 2);
        std::array::from_fn(|l| {
            let r = motion.refidx[l][k] as i32;
            (r, if r >= 0 { Some(motion.mv[l][k]) } else { None })
        })
    };
    let q = [quad(0), quad(1), quad(2), quad(3)];
    let (part, groups): (Part, Vec<Quad>) = if q[0] == q[1] && q[1] == q[2] && q[2] == q[3] {
        (Part::P16x16, vec![q[0]])
    } else if q[0] == q[1] && q[2] == q[3] {
        (Part::P16x8, vec![q[0], q[2]])
    } else if q[0] == q[2] && q[1] == q[3] {
        (Part::P8x16, vec![q[0], q[1]])
    } else {
        return reexpress(mb, SliceType::B, motion);
    };
    let preds: Vec<Pred> = groups
        .iter()
        .map(|g| {
            if g[0].0 >= 0 && g[1].0 >= 0 {
                Pred::Bi
            } else if g[0].0 >= 0 {
                Pred::L0
            } else {
                Pred::L1
            }
        })
        .collect();
    let mut n = mb.clone();
    n.skip = false;
    n.kind = Kind::B;
    n.part = Some(part);
    n.btype = Some(b_type_index(part, &preds).expect("every partitioning / prediction pair is a B macroblock type"));
    n.direct16 = false;
    let refs: [Vec<Option<i8>>; 2] =
        std::array::from_fn(|l| groups.iter().map(|g| if g[l].0 >= 0 { Some(g[l].0 as i8) } else { None }).collect());
    n.refs = Some(refs);
    let mvds: [Vec<Option<Mv>>; 2] =
        std::array::from_fn(|l| groups.iter().map(|g| if g[l].0 >= 0 { Some([0, 0]) } else { None }).collect());
    n.mvds = Some(Mvds::Parts(mvds));
    n.sub = None;
    if mb.skip {
        clear_residual(&mut n);
    }
    n
}

/// Whether an inter MB has direct-predicted blocks (B_Skip, B_Direct_16x16,
/// or B_8x8 with a direct sub-macroblock).
pub fn has_direct(mb: &Mb) -> bool {
    !mb.intra && (mb.skip || mb.part == Some(Part::Direct) || (mb.part == Some(Part::P8x8) && mb.sub.is_some_and(|s| s.contains(&0))))
}

// ===========================================================================
// dev reexpress
// ===========================================================================

/// Decode an Annex-B byte string with ffmpeg and return the MD5 of every frame.
pub fn frame_hashes(data: &[u8], fs: usize, tag: &str) -> Result<Vec<String>> {
    let tmp = std::env::temp_dir().join(format!("fd_reexpress_{}_{}.h264", std::process::id(), tag));
    std::fs::write(&tmp, data)?;
    let res = (|| -> Result<Vec<String>> {
        let mut out = Vec::new();
        for fr in iter_frames(&tmp, &[(0, data.len())], fs, FFMPEG_THREADS)? {
            out.push(md5_hex(&fr?));
        }
        Ok(out)
    })();
    let _ = std::fs::remove_file(&tmp);
    res
}

/// `frame-dedup dev reexpress <file.h264> <first P dec> <count>`: rewrite
/// every inter macroblock of the window as explicit 8x8, re-solve the mvds,
/// re-encode, decode the segment with ffmpeg and compare the frames with the
/// original segment. Port of `verify_motion.main`.
pub fn dev_reexpress(args: &[String]) -> Result<()> {
    let [path, first, count] = args else {
        bail!("usage: frame-dedup dev reexpress <file.h264> <first P dec> <count>");
    };
    let (first, count): (usize, usize) = (first.parse().context("bad first")?, count.parse().context("bad count")?);
    let (spss, ppss, pics) = crate::stream::load(Path::new(path))?;
    let file = std::fs::File::open(path)?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let es: &[u8] = &map;
    let sps = spss.get(&0).ok_or_else(|| anyhow!("no SPS 0"))?;
    let (w, h) = (sps.mb_w, sps.mb_h);
    let mut repl: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut motions: HashMap<usize, Motion> = HashMap::new();
    for d in first..first + count {
        let p = pics.get(d).ok_or_else(|| anyhow!("no picture {d}"))?;
        let t = p.typ();
        let mut dd = decode_picture(es, &spss, &ppss, p)?;
        let col = if t == SliceType::B { motions.get(&p.lists[1][0]) } else { None };
        if t == SliceType::B && col.is_none() {
            bail!("colocated picture {} for {} not in window", p.lists[1][0], d);
        }
        let m = derive_motion(p, &dd.mbs, w, h, &pics, col)?;
        if t == SliceType::I {
            motions.insert(d, m);
            continue;
        }
        let mut new: Vec<Mb> = dd.mbs.iter().map(|mb| reexpress(mb, t, &m)).collect();
        let m2 = derive_motion_target(p, &mut new, w, h, &pics, col, &m)?;
        let bad = motion_mismatches(&m, &m2);
        let nal = try_encode_picture(&dd, &spss, &ppss, &mut new)?;
        println!("dec {d} {}: {}B -> {}B re-expressed, motion mismatches {bad}", t.as_str(), dd.nal_len, nal.len());
        dd.mbs.clear();
        repl.insert(d, nal);
        motions.insert(d, m);
    }
    let start = (0..=first).rev().find(|&k| pics[k].idr).unwrap_or(0);
    let end = first + count;
    let fs = crate::decode::frame_size(sps);
    let a = frame_hashes(&segment_bytes(es, &pics, start, end, &HashMap::new()), fs, "a")?;
    let b = frame_hashes(&segment_bytes(es, &pics, start, end, &repl), fs, "b")?;
    let diff: Vec<usize> = a.iter().zip(&b).enumerate().filter(|(_, (x, y))| x != y).map(|(i, _)| i).collect();
    let shown: Vec<usize> = diff.iter().copied().take(20).collect();
    println!("segment {start}..{end}: frames orig {} new {}  differing {} {:?}", a.len(), b.len(), diff.len(), shown);
    if a.len() != b.len() || !diff.is_empty() {
        bail!("re-expressed segment does not decode identically");
    }
    Ok(())
}

// ===========================================================================
// dev motion
// ===========================================================================

/// Number of 4x4 blocks where the solved motion `m2` differs from `m` in
/// reference index or mv (Python `bad` of `verify_motion.main`).
fn motion_mismatches(m: &Motion, m2: &Motion) -> usize {
    let mut bad = 0usize;
    for l in 0..2 {
        for i in 0..m.refidx[l].len() {
            if m.refidx[l][i] >= 0 && (m.mv[l][i] != m2.mv[l][i] || m.refidx[l][i] != m2.refidx[l][i]) {
                bad += 1;
            }
        }
    }
    bad
}

/// `frame-dedup dev motion <file.h264> [--from N] [--count N] [--threads N] [--target]`
///
/// Per picture, the digest of the canonical motion field (see
/// `Motion::canonical_digests`). Derivation starts at the IDR of the period
/// holding `--from` (B pictures need their co-located motion), printing from
/// `--from` on. With `--target`, every inter picture is also re-expressed as
/// in `dev reexpress` and a second line per picture gives the digest of the
/// solved records, the number of motion mismatches, and the length and
/// SHA-256 of the re-encoded NAL.
pub fn dev_motion(args: &[String]) -> Result<()> {
    const USAGE: &str = "frame-dedup dev motion <file.h264> [--from N] [--count N] [--threads N] [--target]";
    let (mut path, mut from, mut count, mut threads, mut target) = (None, 0usize, None, 4usize, false);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |name: &str| -> Result<usize> {
            it.next().ok_or_else(|| anyhow!("{name} needs a value"))?.parse().with_context(|| format!("bad {name}"))
        };
        match a.as_str() {
            "--from" => from = num("--from")?,
            "--count" => count = Some(num("--count")?),
            "--threads" => threads = num("--threads")?.max(1),
            "--target" => target = true,
            _ if path.is_none() => path = Some(a.clone()),
            _ => bail!("usage: {USAGE}"),
        }
    }
    let path = path.ok_or_else(|| anyhow!("usage: {USAGE}"))?;
    let t0 = std::time::Instant::now();
    let (n, derive_s, derive_n) = motion_lines(&path, from, count, threads, target, &mut |l| println!("{l}"))?;
    eprintln!(
        "motion: {n} pictures in {:.2}s; derive_motion: {derive_n} inter pictures, mean {:.1} ms",
        t0.elapsed().as_secs_f64(),
        derive_s / derive_n.max(1) as f64 * 1e3
    );
    Ok(())
}

/// The lines of `dev motion` (see there), handed to `sink`. Returns the number
/// of pictures covered, the seconds spent in `derive_motion` (inter pictures
/// only) and the number of such pictures.
pub fn motion_lines(
    path: &str,
    from: usize,
    count: Option<usize>,
    threads: usize,
    target: bool,
    sink: &mut dyn FnMut(String),
) -> Result<(usize, f64, usize)> {
    let (spss, ppss, pics) = crate::stream::load(Path::new(path))?;
    let file = std::fs::File::open(path)?;
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let es: &[u8] = &map;
    let sps = spss.get(&0).ok_or_else(|| anyhow!("no SPS 0"))?;
    let (w, h) = (sps.mb_w, sps.mb_h);
    let a = from.min(pics.len());
    let b = count.map_or(pics.len(), |c| (a + c).min(pics.len()));
    let start = (0..=a.min(pics.len().saturating_sub(1))).rev().find(|&k| pics[k].idr).unwrap_or(0);
    let mut motions: HashMap<usize, Motion> = HashMap::new();
    let (mut derive_s, mut derive_n) = (0.0f64, 0usize);
    let mut d0 = start;
    while d0 < b {
        let d1 = (d0 + threads * 2).min(b);
        // parse a chunk in parallel, derive in decode order
        let parsed = parse_chunk(es, &spss, &ppss, &pics, d0, d1, threads);
        for (k, dd) in parsed.into_iter().enumerate() {
            let d = d0 + k;
            let p = &pics[d];
            let t = p.typ();
            let dd = dd.with_context(|| format!("picture {d}"))?;
            let col = if t == SliceType::B { motions.get(&p.lists[1][0]) } else { None };
            let t_derive = std::time::Instant::now();
            let m = derive_motion(p, &dd.mbs, w, h, &pics, col).with_context(|| format!("picture {d}"))?;
            if d >= a && t != SliceType::I {
                derive_s += t_derive.elapsed().as_secs_f64();
                derive_n += 1;
            }
            if d >= a {
                let (digest, fields) = m.canonical_digests();
                let line = json!({"dec": d, "type": t.as_str(), "digest": digest,
                    "intra": fields[0], "direct": fields[1], "refidx": fields[2], "mv": fields[3], "refpic": fields[4]});
                sink(canonical_text(&line));
                if target && t != SliceType::I {
                    let mut new: Vec<Mb> = dd.mbs.iter().map(|mb| reexpress(mb, t, &m)).collect();
                    let m2 = derive_motion_target(p, &mut new, w, h, &pics, col, &m)?;
                    let bad = motion_mismatches(&m, &m2);
                    let list = Value::Array(new.iter().map(Mb::canonical).collect());
                    let rec_digest = hex(&sha256(canonical_text(&list).as_bytes()));
                    let nal = try_encode_picture(&dd, &spss, &ppss, &mut new)?;
                    let line = json!({"dec": d, "target": true, "type": t.as_str(), "records": rec_digest, "bad": bad,
                        "nal_len": nal.len(), "nal_sha256": hex(&sha256(&nal)), "motion2": m2.canonical_digests().0});
                    sink(canonical_text(&line));
                }
            }
            motions.insert(d, m);
            if p.ref_idc != 0 || p.idr {
                // a co-located picture is always a reference picture still in the DPB
                let keep: HashSet<usize> = p.dpb_after.iter().copied().collect();
                motions.retain(|k, _| keep.contains(k));
            }
        }
        d0 = d1;
    }
    Ok((b - a, derive_s, derive_n))
}

/// Parse pictures `d0..d1` on `threads` threads; results in decode order.
fn parse_chunk(es: &[u8], spss: &Spss, ppss: &Ppss, pics: &[Picture], d0: usize, d1: usize, threads: usize) -> Vec<Result<Parsed>> {
    let next = AtomicUsize::new(d0);
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let d = next.fetch_add(1, Ordering::Relaxed);
                        if d >= d1 {
                            break;
                        }
                        out.push((d, decode_picture(es, spss, ppss, &pics[d])));
                    }
                    out
                })
            })
            .collect();
        let mut all: Vec<(usize, Result<Parsed>)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        all.sort_by_key(|(d, _)| *d);
        all.into_iter().map(|(_, r)| r).collect()
    })
}

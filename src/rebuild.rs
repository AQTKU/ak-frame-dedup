//! Anchor rebuild: drop the earlier twin K of a pair whose later twin D is an
//! IDR, by demoting D to a non-IDR I picture ("I2") that takes K's place.
//! Port of `frame_dedup/rebuild.py`.
//!
//! K and D encode the same source frame. I2 is D's slice data under a non-IDR
//! header carrying K's frame_num and K's POC, written in K's decode slot, so
//! B1-B3 see an anchor at exactly the same list positions and POC distances.
//! D's former IDR period is merged into K's: its POCs are shifted by the
//! constant POC(K) - POC(D) and frame_num continues from K's.
//!
//! The header side (demotion, planning, list check, eligibility, writing),
//! the B1-B3 re-encoding (`run_task`, with the PCM rounds) and the parallel
//! driver (`rebuild`, one task per pair, worker threads sharing the mapped
//! streams).

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

use crate::bits::{BitWriter, escape, set_bits, unescape};
use crate::h264hdr::{Picture, Ppss, SliceHeader, Sps, Spss, parse_slice};
use crate::decode::{decode_pocs, frame_size, iter_frames};
use crate::mv::{Motion, derive_motion, derive_motion_target, dpb_walk};
use crate::patch::{fix_qp, pcm_record, qp_chain, same_motion};
use crate::picture::{Parsed, decode_picture, try_encode_picture};
use crate::records::{Kind, Mb};
use crate::reexpress::{has_direct, reexpress_compact};
use crate::repair::{TmpFile, map_file, round1, run_pool};
use crate::system::{FFMPEG_THREADS, REPAIR_PEAK, pool_size};
use crate::verify::mb_maxdiff;

/// sei_message: payloadType 6 (recovery point), payloadSize 1,
/// recovery_frame_cnt ue 0 ('1'), exact_match 1, broken_link 0,
/// changing_slice_group_idc 00, payload alignment '100'; rbsp_trailing_bits
pub const RECOVERY_SEI: [u8; 5] = [0x06, 0x06, 0x01, 0xc4, 0x80];

/// A B1-B3 macroblock whose decoded result differs from the original by this
/// much or more (max over Y, U, V) is replaced by I_PCM, up to a budget. K and
/// D themselves differ by up to 23-90 levels; at this threshold the encoder's
/// local "refresh" spots are caught for a handful of MBs per picture.
pub const PCM_THRESHOLD: u32 = 16;
/// At most this fraction of a picture's MBs is patched, worst first. In fast
/// motion the IDR twin can be coded very differently from K (MAD 2-5, peaks
/// of 60-120), and holding every MB under the threshold would mean PCM-ing
/// most of the picture; there the motion masks the remaining differences.
pub const PCM_MAX_FRACTION: f64 = 0.02;
pub const MAX_ROUNDS: usize = 4;

const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// One re-encoding task: the B1-B3 pictures of one rebuild.
struct Task {
    /// Decode range of K's period through D in the source stream.
    seg: Range<usize>,
    /// First (an IDR) and last original decode index to decode for the
    /// original pixels of the leading pictures.
    orig_first: usize,
    orig_last: usize,
}

struct Ctx<'a> {
    orig_es: &'a Path,
    src_es: &'a Path,
    src_map: &'a [u8],
    spss: &'a Spss,
    ppss: &'a Ppss,
    orig_pics: &'a [Picture],
    src_pics: &'a [Picture],
    orig_of_src: &'a [usize],
    rebuilds: &'a [Rebuild],
    work: &'a Path,
}

/// Statistics of one leading picture (Python's `stats[b]` dict).
#[derive(Clone, Debug, Default)]
struct BStat {
    src_dec: usize,
    display_in: usize,
    direct_explicit: usize,
    bytes_before: usize,
    skip_explicit: usize,
    qp_carrier: usize,
    pcm: usize,
    bytes_after: usize,
    worst: u32,
    mbs_ge3: usize,
    mbs_diff: usize,
    worst_luma: u32,
    mbs_ge3_luma: usize,
    rounds: usize,
    pcm_capped: bool,
}

impl BStat {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "src_dec": self.src_dec, "display_in": self.display_in, "direct_explicit": self.direct_explicit,
            "bytes_before": self.bytes_before, "skip_explicit": self.skip_explicit, "qp_carrier": self.qp_carrier,
            "pcm": self.pcm, "bytes_after": self.bytes_after, "worst": self.worst, "mbs_ge3": self.mbs_ge3,
            "mbs_diff": self.mbs_diff, "worst_luma": self.worst_luma, "mbs_ge3_luma": self.mbs_ge3_luma,
            "rounds": self.rounds, "pcm_capped": self.pcm_capped,
        })
    }
}

struct TaskResult {
    repl: HashMap<usize, Vec<u8>>,
    stats: Vec<BStat>,
    seconds: f64,
}

/// Records for a B1-B3 picture: `base` (direct blocks made explicit) with the
/// MBs in `pcm` replaced by I_PCM copies of `yuv`, skips that can no longer
/// inherit their QP made explicit, mvds re-solved and the QP chain kept.
/// Returns (records, number of skips made explicit, QP carriers).
#[allow(clippy::too_many_arguments)]
fn build_b(
    pic: &Picture,
    dd: &Parsed,
    base: &[Mb],
    m0: &Motion,
    pcm: &BTreeSet<(u16, u16)>,
    yuv: &[u8],
    pics: &[Picture],
    intra: &Motion,
    w: usize,
    h: usize,
) -> Result<(Vec<Mb>, usize, usize)> {
    let mut new: Vec<Mb> = base.to_vec();
    for mb in new.iter_mut() {
        if pcm.contains(&(mb.x, mb.y)) {
            let (x, y) = (mb.x, mb.y);
            let mut r = pcm_record(yuv, w, x as usize, y as usize);
            r.x = x;
            r.y = y;
            *mb = r;
        }
    }
    let qp = pic.slices[0].qp;
    let target_qp = qp_chain(&dd.mbs, qp);
    let (mut pred, mut n_skip) = (qp, 0usize);
    for (i, mb) in new.iter_mut().enumerate() {
        if mb.kind == Kind::Pcm {
            continue;
        }
        if mb.skip && target_qp[i] != pred {
            *mb = reexpress_compact(mb, m0);
            n_skip += 1;
        }
        pred = target_qp[i];
    }
    let m2 = derive_motion_target(pic, &mut new, w, h, pics, Some(intra), m0)?;
    let bad: Vec<String> = new
        .iter()
        .filter(|mb| mb.kind != Kind::Pcm && !mb.intra && !same_motion(m0, &m2, mb.x as usize, mb.y as usize))
        .take(5)
        .map(|mb| format!("({}, {})", mb.x, mb.y))
        .collect();
    if !bad.is_empty() {
        bail!("AssertionError: dec {}: motion could not be preserved at [{}]", pic.dec, bad.join(", "));
    }
    let forced = fix_qp(&mut new, &target_qp, qp)?;
    Ok((new, n_skip, forced))
}

/// Re-encode B1-B3 of one rebuild. Returns their NALs and statistics.
fn run_task(cx: &Ctx, ri: usize, task: &Task) -> Result<TaskResult> {
    let t0 = Instant::now();
    let sps = cx.spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let (w, h) = (sps.mb_w, sps.mb_h);
    let fs = frame_size(sps);
    let r = &cx.rebuilds[ri];
    let (k, bs) = (r.k, &r.b);
    let src = cx.src_pics;
    let intra = Motion::all_intra(w, h);
    let dk = decode_picture(cx.src_map, cx.spss, cx.ppss, &src[k])?;
    let mk = derive_motion(&src[k], &dk.mbs, w, h, src, None)?;
    // (records, base records, motion with K)
    let mut recs: Vec<(Parsed, Vec<Mb>, Motion)> = Vec::new();
    let mut stats: Vec<BStat> = Vec::new();
    for &b in bs {
        let pb = &src[b];
        if pb.lists[1].first() != Some(&k) {
            bail!("AssertionError: picture {b} does not have K as its first list 1 entry");
        }
        let dd = decode_picture(cx.src_map, cx.spss, cx.ppss, pb)?;
        let m0 = derive_motion(pb, &dd.mbs, w, h, src, Some(&mk))?; // what B decoded with K
        let m1 = derive_motion(pb, &dd.mbs, w, h, src, Some(&intra))?; // what its records give with I2
        let mut base = Vec::with_capacity(dd.mbs.len());
        let mut n_dir = 0;
        for mb in &dd.mbs {
            if has_direct(mb) && !same_motion(&m0, &m1, mb.x as usize, mb.y as usize) {
                base.push(reexpress_compact(mb, &m0));
                n_dir += 1;
            } else {
                base.push(mb.clone());
            }
        }
        stats.push(BStat {
            src_dec: b,
            display_in: cx.orig_pics[cx.orig_of_src[b]].disp,
            direct_explicit: n_dir,
            bytes_before: dd.nal_len,
            ..Default::default()
        });
        recs.push((dd, base, m0));
    }
    drop(dk);

    // original pixels of B1-B3
    let orig_poc: Vec<i32> = bs.iter().map(|&b| cx.orig_pics[cx.orig_of_src[b]].poc).collect();
    let want: BTreeSet<i32> = orig_poc.iter().copied().collect();
    let frames = decode_pocs(cx.orig_es, cx.orig_pics, task.orig_first, task.orig_last + 1, &want, fs, FFMPEG_THREADS)?;
    let target: Vec<&Vec<u8>> = orig_poc.iter().map(|p| frames.get(p).ok_or_else(|| anyhow!("KeyError: {p}"))).collect::<Result<_>>()?;

    let tmp = TmpFile(cx.work.join(format!("rebuild_{k}.h264")));
    let mut pcm: Vec<BTreeSet<(u16, u16)>> = vec![BTreeSet::new(); bs.len()];
    let budget = (PCM_MAX_FRACTION * w as f64 * h as f64) as usize;
    let seg = &src[task.seg.clone()];
    let mut repl: HashMap<usize, Vec<u8>> = HashMap::new();
    for rnd in 0..MAX_ROUNDS {
        repl.clear();
        for (i, &b) in bs.iter().enumerate() {
            let (dd, base, m0) = &recs[i];
            let (mut new, n_skip, forced) = build_b(&src[b], dd, base, m0, &pcm[i], target[i], src, &intra, w, h)?;
            let nal = try_encode_picture(dd, cx.spss, cx.ppss, &mut new).with_context(|| format!("dec {b}"))?;
            stats[i].skip_explicit = n_skip;
            stats[i].qp_carrier = forced;
            stats[i].pcm = pcm[i].len();
            stats[i].bytes_after = nal.len();
            repl.insert(b, nal);
        }
        let entries = write_rebuilt(cx.src_es, seg, task.seg.start, cx.spss, cx.ppss, &tmp.0, std::slice::from_ref(r), &repl)?;
        let mut order: Vec<(i32, usize)> = entries.iter().map(|e| (e.poc, e.pic)).collect();
        order.sort_unstable();
        let mut got: HashMap<usize, Vec<u8>> = HashMap::new();
        {
            let size = std::fs::metadata(&tmp.0)?.len() as usize;
            let mut it = iter_frames(&tmp.0, &[(0, size)], fs, FFMPEG_THREADS)?;
            let mut fr = Vec::new();
            for &(_, dec) in &order {
                if !it.next_frame(&mut fr)? {
                    break;
                }
                if bs.contains(&dec) {
                    got.insert(dec, fr.clone());
                }
            }
        }
        if got.len() != bs.len() || !bs.iter().all(|b| got.contains_key(b)) {
            bail!("AssertionError: segment decode lost a leading picture");
        }
        let mut grew = false;
        for (i, &b) in bs.iter().enumerate() {
            let (m, ly) = mb_maxdiff(&got[&b], target[i], w, h);
            let st = &mut stats[i];
            st.worst = m.iter().copied().max().unwrap_or(0) as u32;
            st.mbs_ge3 = m.iter().filter(|&&v| v >= 3).count();
            st.mbs_diff = m.iter().filter(|&&v| v > 0).count();
            st.worst_luma = ly.iter().copied().max().unwrap_or(0) as u32;
            st.mbs_ge3_luma = ly.iter().filter(|&&v| v >= 3).count();
            st.rounds = rnd + 1;
            let mut more: Vec<(u8, u16, u16)> = Vec::new();
            for (j, &v) in m.iter().enumerate() {
                if v as u32 >= PCM_THRESHOLD {
                    let (x, y) = ((j % w) as u16, (j / w) as u16);
                    if !pcm[i].contains(&(x, y)) {
                        more.push((v, x, y));
                    }
                }
            }
            more.sort_by(|a, b| b.cmp(a)); // sorted(..., reverse=True)
            let room = budget as i64 - pcm[i].len() as i64;
            st.pcm_capped = more.len() as i64 > room;
            if !more.is_empty() && room > 0 {
                pcm[i].extend(more.iter().take(room as usize).map(|&(_, x, y)| (x, y)));
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    drop(tmp);
    Ok(TaskResult { repl, stats, seconds: round1(t0.elapsed().as_secs_f64()) })
}

/// Apply `rebuilds` (decode indices of the stream `src_es`, parsed as
/// `src_pics`; `orig_of_src` maps them to the original stream's) and write
/// `out_path`. Returns (the original decode index of every output picture in
/// output decode order, the report's `anchor_rebuilds` section with Python's
/// keys). Python: `rebuild.rebuild`.
#[allow(clippy::too_many_arguments)]
pub fn rebuild(
    orig_es: &Path,
    src_es: &Path,
    out_path: &Path,
    spss: &Spss,
    ppss: &Ppss,
    orig_pics: &[Picture],
    src_pics: &[Picture],
    orig_of_src: &[usize],
    rebuilds: &[Rebuild],
    workers: usize,
    work: &Path,
    log: &crate::system::Logger,
) -> Result<(Vec<usize>, serde_json::Value)> {
    let t0 = Instant::now();
    let chk = check_lists(spss, src_pics, 0, rebuilds)?;
    if !chk.fails.is_empty() {
        let first: Vec<String> = chk.fails.iter().take(3).map(|f| f.to_string()).collect();
        bail!("anchor rebuild would change reference lists: [{}]", first.join(", "));
    }
    let periods = crate::stream::period_ranges(src_pics);
    let mut tasks = Vec::new();
    for r in rebuilds {
        let kp = &src_pics[r.k];
        let seg = periods[kp.period].start..r.d + 1;
        let ob: Vec<usize> = r.b.iter().map(|&b| orig_of_src[b]).collect();
        let (Some(&lo), Some(&hi)) = (ob.iter().min(), ob.iter().max()) else {
            bail!("ValueError: min() arg is an empty sequence");
        };
        let orig_first = (0..=lo).rev().find(|&d| orig_pics[d].idr).ok_or_else(|| anyhow!("ValueError: max() arg is an empty sequence"))?;
        tasks.push(Task { seg, orig_first, orig_last: hi });
    }
    let nproc = if tasks.is_empty() { 0 } else { pool_size(workers, REPAIR_PEAK, Some(tasks.len())) };
    log.log(&format!(
        "anchor rebuild: {} pairs, {} leading B pictures, {nproc} workers",
        tasks.len(),
        rebuilds.iter().map(|r| r.b.len()).sum::<usize>()
    ));
    let mut results: Vec<TaskResult> = Vec::new();
    if !tasks.is_empty() {
        let src_map = map_file(src_es)?;
        let cx = Ctx { orig_es, src_es, src_map: &src_map, spss, ppss, orig_pics, src_pics, orig_of_src, rebuilds, work };
        let idx: Vec<(usize, &Task)> = tasks.iter().enumerate().collect();
        let n = tasks.len();
        let work_fn = |t: &(usize, &Task)| run_task(&cx, t.0, t.1);
        results = run_pool(nproc, &idx, &work_fn, |i, _| {
            if (i + 1) % 25 == 0 || i + 1 == n {
                log.log(&format!("  rebuild: {}/{n} pairs", i + 1));
            }
        })?;
    }
    let mut repl: HashMap<usize, Vec<u8>> = HashMap::new();
    for res in &results {
        for (b, nal) in &res.repl {
            repl.insert(*b, nal.clone());
        }
    }
    let entries = write_rebuilt(src_es, src_pics, 0, spss, ppss, out_path, rebuilds, &repl)?;
    let out_src: Vec<usize> = entries.iter().map(|e| e.pic).collect();

    // the written stream must parse to exactly the planned POCs and lists
    let (_, _, out_pics) = crate::stream::load(out_path)?;
    if out_pics.len() != chk.shadow.len() {
        bail!("AssertionError: the written stream has {} pictures, {} planned", out_pics.len(), chk.shadow.len());
    }
    for (q, s) in out_pics.iter().zip(&chk.shadow) {
        if q.poc != s.poc || q.lists != s.lists || q.idr != s.idr {
            bail!("AssertionError: {}", q.dec);
        }
    }

    let mut pairs = Vec::new();
    let mut worst = 0u32;
    let (mut n_leading, mut n_capped_pairs, mut pcm_total, mut direct_total, mut bytes_total) = (0usize, 0usize, 0usize, 0usize, 0i64);
    for (r, res) in rebuilds.iter().zip(&results) {
        let st = &res.stats;
        worst = worst.max(st.iter().map(|s| s.worst).max().unwrap_or(0));
        let pcm_mbs: usize = st.iter().map(|s| s.pcm).sum();
        let direct_explicit: usize = st.iter().map(|s| s.direct_explicit).sum();
        let bytes_added: i64 = st.iter().map(|s| s.bytes_after as i64 - s.bytes_before as i64).sum();
        let pcm_capped = st.iter().filter(|s| s.pcm_capped).count();
        n_leading += st.len();
        n_capped_pairs += pcm_capped;
        pcm_total += pcm_mbs;
        direct_total += direct_explicit;
        bytes_total += bytes_added;
        pairs.push(serde_json::json!({
            "display": orig_pics[orig_of_src[r.k]].disp,
            "kept_display": orig_pics[orig_of_src[r.d]].disp,
            "types": format!("{}Ii", src_pics[r.k].typ().as_str()),
            "leading_b": st.len(), "pcm_mbs": pcm_mbs, "direct_explicit": direct_explicit, "bytes_added": bytes_added,
            "worst": st.iter().map(|s| s.worst).max().unwrap_or(0),
            "mbs_ge3": st.iter().map(|s| s.mbs_ge3).sum::<usize>(),
            "worst_luma": st.iter().map(|s| s.worst_luma).max().unwrap_or(0),
            "pcm_capped": pcm_capped,
            "pictures": st.iter().map(BStat::to_json).collect::<Vec<_>>(),
            "seconds": res.seconds,
        }));
    }
    let sps = spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let summary = serde_json::json!({
        "pairs": pairs.len(), "leading_b_pictures": n_leading,
        "pcm_threshold": PCM_THRESHOLD, "pcm_max_per_picture": (PCM_MAX_FRACTION * sps.mb_w as f64 * sps.mb_h as f64) as usize,
        "pictures_pcm_capped": n_capped_pairs, "pcm_mbs": pcm_total, "direct_explicit": direct_total,
        "bytes_added": bytes_total, "worst_in_leading_b": worst, "workers": nproc,
        "worker_peak_rss_mb": (crate::system::peak_rss() as f64 / (1u64 << 20) as f64).round() as u64,
        "seconds": round1(t0.elapsed().as_secs_f64()), "pairs_detail": pairs,
    });
    Ok((out_src.iter().map(|&d| orig_of_src[d]).collect(), summary))
}

/// A planned rebuild (Python: `{'K', 'D', 'B'}`): K the dropped P, D the IDR
/// that takes its place, `b` the leading pictures that read K (decode
/// indices in the stream the rebuild is planned on).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rebuild {
    pub k: usize,
    pub d: usize,
    pub b: Vec<usize>,
}

// ---------------------------------------------------------------- headers

/// Return an IDR slice NAL rewritten as a non-IDR I slice (nal_unit_type 1,
/// same nal_ref_idc) with the given frame_num and pic_order_cnt_lsb:
/// idr_pic_id removed, dec_ref_pic_marking replaced by
/// adaptive_ref_pic_marking_mode_flag = 0. The CABAC slice data starts
/// byte-aligned and depends only on slice QP and slice type, so its bytes
/// are appended unchanged.
pub fn demote_idr(nal: &[u8], sl: &SliceHeader, spss: &Spss, ppss: &Ppss, frame_num: u32, poc_lsb: u32) -> Result<Vec<u8>> {
    if nal[0] & 0x1f != 5 || sl.nal_type != 5 {
        bail!("AssertionError: demote_idr needs an IDR slice");
    }
    let pps = ppss.get(&sl.pps_id).ok_or_else(|| anyhow!("KeyError: PPS {}", sl.pps_id))?;
    let sps = spss.get(&pps.sps_id).ok_or_else(|| anyhow!("KeyError: SPS {}", pps.sps_id))?;
    let rbsp = unescape(&nal[1..]);
    let (lfn, lpoc) = (sps.log2_max_frame_num, sps.log2_max_poc_lsb);
    let data_byte = sl.data_bit / 8;
    if rbsp.len() < data_byte {
        bail!("IndexError: slice shorter than its header");
    }
    for pos in sl.hdr_bits..data_byte * 8 {
        if (rbsp[pos >> 3] >> (7 - (pos & 7))) & 1 == 0 {
            bail!("AssertionError: cabac_alignment_one_bits expected");
        }
    }
    if sl.idr_pic_id_bit != Some(sl.frame_num_bit + lfn as usize) {
        bail!("AssertionError: idr_pic_id does not follow frame_num");
    }
    let mut w = BitWriter::with_capacity(data_byte + 8);
    w.put_bits_from(&rbsp, 0, sl.frame_num_bit); // up to frame_num; idr_pic_id dropped
    w.put(lfn, frame_num);
    w.put(lpoc, poc_lsb);
    w.put_bits_from(&rbsp, sl.poc_lsb_bit + lpoc as usize, sl.drpm_bit);
    w.put_bit(false); // adaptive_ref_pic_marking_mode_flag
    w.put_bits_from(&rbsp, sl.drpm_end_bit, sl.hdr_bits);
    let hdr_bits = w.len_bits();
    w.align_ones(); // cabac_alignment_one_bit
    let n_bits = w.len_bits();
    let mut new_rbsp = w.finish();
    new_rbsp.extend_from_slice(&rbsp[data_byte..]);
    let ref_idc = (nal[0] >> 5) & 3;
    let mut new = vec![(nal[0] & 0xe0) | 1];
    new.extend_from_slice(&escape(&new_rbsp));
    let chk = parse_slice(&new_rbsp[..new_rbsp.len().min(64)], 1, ref_idc, spss, ppss)?;
    macro_rules! same {
        ($($f:ident),*) => {$(
            if chk.$f != sl.$f {
                bail!("AssertionError: {} {:?} != {:?}", stringify!($f), chk.$f, sl.$f);
            }
        )*};
    }
    same!(first_mb, slice_type, pps_id, qp, deblock_idc, alpha_off, beta_off, delta_poc_bottom, cabac_init_idc);
    if chk.frame_num != frame_num || chk.poc_lsb != poc_lsb || !chk.mmco.is_empty() {
        bail!("AssertionError: demoted header does not read back");
    }
    if chk.hdr_bits != hdr_bits || chk.data_bit != n_bits {
        bail!("AssertionError: demoted header length does not read back");
    }
    Ok(new)
}

/// One picture of the rewritten stream (Python: a tuple `(picture, is_i2,
/// frame_num, poc)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderEntry {
    /// The original picture (decode index); for I2 this is D.
    pub pic: usize,
    pub i2: bool,
    pub frame_num: u32,
    pub poc: i32,
}

/// The rewritten stream in decode order. D is written in K's slot as I2 with
/// K's POC; every POC of D's former period is shifted by POC(K) - POC(D);
/// frame_num is re-derived as PrevRefFrameNum + 1 (0 at a real IDR), which is
/// what the input carries, since it has no frame_num gaps. `pics[d - base]`
/// is picture `d` (`base` = decode index of `pics[0]`).
pub fn plan_headers(pics: &[Picture], base: usize, spss: &Spss, rebuilds: &[Rebuild]) -> Result<Vec<HeaderEntry>> {
    let Some(sps) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    let max_fn: u32 = 1 << sps.log2_max_frame_num;
    let by_k: HashMap<usize, &Rebuild> = rebuilds.iter().map(|r| (r.k, r)).collect();
    let by_d: std::collections::HashSet<usize> = rebuilds.iter().map(|r| r.d).collect();
    let mut out = Vec::with_capacity(pics.len());
    let mut i2poc: HashMap<usize, i32> = HashMap::new();
    let (mut prev_fn, mut off): (u32, i32) = (0, 0);
    for p in pics {
        if by_d.contains(&p.dec) {
            let Some(&ip) = i2poc.get(&p.dec) else {
                bail!("KeyError: {} (the IDR of a rebuild comes before its K)", p.dec);
            };
            off = ip - p.poc;
            continue;
        }
        if let Some(r) = by_k.get(&p.dec) {
            let dp = &pics[r.d - base];
            let fnum = (prev_fn + 1) % max_fn;
            let poc = p.poc + off;
            out.push(HeaderEntry { pic: dp.dec, i2: true, frame_num: fnum, poc });
            i2poc.insert(dp.dec, poc);
            prev_fn = fnum;
            continue;
        }
        let fnum;
        if p.idr {
            off = 0;
            fnum = 0;
        } else {
            fnum = (prev_fn + 1) % max_fn;
        }
        out.push(HeaderEntry { pic: p.dec, i2: false, frame_num: fnum, poc: p.poc + off });
        if p.ref_idc != 0 {
            prev_fn = fnum;
        }
    }
    Ok(out)
}

/// A value in a [`Fail`]: Python's dict holds an int for POC failures and a
/// list for list failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailVal {
    Int(i64),
    List(Vec<usize>),
}

/// One failure of [`check_lists`] (Python: `{'dec', 'what', 'want', 'got'}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fail {
    pub dec: usize,
    pub what: String,
    pub want: FailVal,
    pub got: FailVal,
}

impl fmt::Display for FailVal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FailVal::Int(v) => write!(f, "{v}"),
            FailVal::List(v) => write!(f, "[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")),
        }
    }
}

/// Python's repr of the dict, as it appears in the report's fallback reasons.
impl fmt::Display for Fail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{'dec': {}, 'what': '{}', 'want': {}, 'got': {}}}", self.dec, self.what, self.want, self.got)
    }
}

/// The result of [`check_lists`].
pub struct ListCheck {
    pub fails: Vec<Fail>,
    /// The shadow pictures after `dpb_walk`, in new decode order (`lists`
    /// index this vector).
    pub shadow: Vec<Picture>,
    /// The original decode index of every shadow picture.
    pub src: Vec<usize>,
}

/// Simulate the decoder's reference lists for the rewritten stream from
/// headers alone. Every list entry the original could use must still point at
/// the same picture (B1-B3 read I2 where they read K).
pub fn check_lists(spss: &Spss, pics: &[Picture], base: usize, rebuilds: &[Rebuild]) -> Result<ListCheck> {
    let Some(sps) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    let max_lsb: i64 = 1 << sps.log2_max_poc_lsb;
    let entries = plan_headers(pics, base, spss, rebuilds)?;
    let mut shadow: Vec<Picture> = Vec::with_capacity(entries.len());
    for e in &entries {
        let p = &pics[e.pic - base];
        let mut sl = p.slices[0].clone();
        sl.frame_num = e.frame_num;
        sl.poc_lsb = (e.poc as i64).rem_euclid(max_lsb) as u32;
        shadow.push(Picture { slices: vec![sl], idr: p.idr && !e.i2, ref_idc: p.ref_idc, ..Default::default() });
    }
    dpb_walk(spss, &mut shadow)?;
    let mut sub: HashMap<usize, (usize, usize)> = HashMap::new();
    for r in rebuilds {
        for &b in &r.b {
            sub.insert(b, (r.k, r.d));
        }
    }
    let src: Vec<usize> = entries.iter().map(|e| e.pic).collect();
    let mut fails = Vec::new();
    for (e, q) in entries.iter().zip(&shadow) {
        let p = &pics[e.pic - base];
        if q.poc != e.poc {
            fails.push(Fail {
                dec: p.dec,
                what: "poc".into(),
                want: FailVal::Int(e.poc as i64),
                got: FailVal::Int(q.poc as i64),
            });
        }
        for l in 0..2 {
            let new: Vec<usize> = q.lists[l].iter().map(|&i| src[i]).collect();
            let mut old: Vec<usize> = if e.i2 { Vec::new() } else { p.lists[l].clone() };
            if let Some(&(k, d)) = sub.get(&p.dec) {
                for x in old.iter_mut() {
                    if *x == k {
                        *x = d;
                    }
                }
            }
            if new.get(..old.len()) != Some(&old[..]) {
                fails.push(Fail { dec: p.dec, what: format!("list{l}"), want: FailVal::List(old), got: FailVal::List(new) });
            }
        }
    }
    Ok(ListCheck { fails, shadow, src })
}

/// Header-only eligibility of a rebuild: D an IDR, K a non-IDR reference
/// picture ending the previous period, only non-reference pictures between
/// them in decode order, nothing outside those reading K, and the list check
/// passing. `periods` are the decode ranges of the IDR periods
/// ([`crate::stream::period_ranges`]). Returns `Ok(Ok(rebuild))`, or
/// `Ok(Err(reason))`.
pub fn candidate(
    spss: &Spss,
    pics: &[Picture],
    periods: &[Range<usize>],
    k: usize,
    d: usize,
) -> Result<std::result::Result<Rebuild, String>> {
    let (pk, pd) = (&pics[k], &pics[d]);
    if !pd.idr || pk.idr || pk.ref_idc == 0 {
        return Ok(Err("twin types".into()));
    }
    if pk.period + 1 != pd.period {
        return Ok(Err("twins not in adjacent IDR periods".into()));
    }
    if (k + 1..d).any(|x| pics[x].ref_idc != 0) {
        return Ok(Err("reference picture between the twins".into()));
    }
    let readers: Vec<usize> = pics[periods[pk.period].clone()]
        .iter()
        .filter(|q| q.lists[0].contains(&k) || q.lists[1].contains(&k))
        .map(|q| q.dec)
        .collect();
    if readers.iter().any(|&x| !(k + 1..d).contains(&x)) {
        return Ok(Err("K read outside the leading pictures".into()));
    }
    let r = Rebuild { k, d, b: readers };
    let range = periods[pk.period].start..periods[pd.period].end;
    let chk = check_lists(spss, &pics[range.clone()], range.start, std::slice::from_ref(&r))?;
    if let Some(f) = chk.fails.first() {
        return Ok(Err(format!("reference lists change: {f}")));
    }
    Ok(Ok(r))
}

// ---------------------------------------------------------------- writing

/// Byte offset of the start code (3 or 4 bytes) in front of a slice NAL.
pub fn first_sc(mm: &[u8], sl: &SliceHeader) -> usize {
    let mut first = sl.offset - 3;
    if mm[first - 1] == 0 {
        first -= 1;
    }
    first
}

/// Overwrite frame_num and pic_order_cnt_lsb of a slice NAL in place and
/// re-escape it.
pub fn patch_fn_poc(nal: &[u8], sl: &SliceHeader, sps: &Sps, fnum: u32, lsb: u32) -> Vec<u8> {
    let mut rbsp = unescape(&nal[1..]);
    set_bits(&mut rbsp, sl.frame_num_bit, sps.log2_max_frame_num, fnum as u64);
    set_bits(&mut rbsp, sl.poc_lsb_bit, sps.log2_max_poc_lsb, lsb as u64);
    let mut out = vec![nal[0]];
    out.extend_from_slice(&escape(&rbsp));
    out
}

/// Write `pics` (a contiguous run of a parsed stream starting at an IDR; the
/// first picture's decode index is `base`) with every rebuild applied: K
/// removed; D demoted and written in K's slot behind K's and D's non-VCL NALs
/// (D's SPS / PPS travel with it) and a recovery point SEI; frame_num and POC
/// rewritten per [`plan_headers`]. `repl`: {dec: NAL} for re-encoded
/// pictures. Returns the plan entries.
pub fn write_rebuilt(
    src_es: &Path,
    pics: &[Picture],
    base: usize,
    spss: &Spss,
    ppss: &Ppss,
    out_path: &Path,
    rebuilds: &[Rebuild],
    repl: &HashMap<usize, Vec<u8>>,
) -> Result<Vec<HeaderEntry>> {
    let sps = spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let max_lsb: i64 = 1 << sps.log2_max_poc_lsb;
    let by_k: HashMap<usize, &Rebuild> = rebuilds.iter().map(|r| (r.k, r)).collect();
    let entries = plan_headers(pics, base, spss, rebuilds)?;
    let mut at: HashMap<usize, (u32, i32)> = HashMap::new();
    let mut i2_of: HashMap<usize, (u32, i32)> = HashMap::new();
    for e in &entries {
        if e.i2 {
            i2_of.insert(e.pic, (e.frame_num, e.poc));
        } else {
            at.insert(e.pic, (e.frame_num, e.poc));
        }
    }
    let file = File::open(src_es).with_context(|| format!("cannot open {}", src_es.display()))?;
    let mm = unsafe { Mmap::map(&file)? };
    let out = File::create(out_path).with_context(|| format!("cannot create {}", out_path.display()))?;
    let mut o = BufWriter::with_capacity(1 << 22, out);
    for p in pics {
        let sl = &p.slices[0];
        let first = first_sc(&mm, sl);
        let end = sl.offset + sl.size;
        if let Some(r) = by_k.get(&p.dec) {
            let dp = &pics[r.d - base];
            let dsl = &dp.slices[0];
            let &(fnum, poc) = i2_of.get(&r.d).ok_or_else(|| anyhow!("KeyError: {}", r.d))?;
            o.write_all(&mm[p.au_start..first])?;
            o.write_all(&mm[dp.au_start..first_sc(&mm, dsl)])?;
            o.write_all(&START_CODE)?;
            o.write_all(&RECOVERY_SEI)?;
            o.write_all(&START_CODE)?;
            o.write_all(&demote_idr(&mm[dsl.offset..dsl.offset + dsl.size], dsl, spss, ppss, fnum, (poc as i64).rem_euclid(max_lsb) as u32)?)?;
            o.write_all(&mm[end..p.au_end])?;
            continue;
        }
        let Some(&(fnum, poc)) = at.get(&p.dec) else {
            // a demoted D: already written in K's slot
            o.write_all(&mm[end..p.au_end])?;
            continue;
        };
        o.write_all(&mm[p.au_start..first])?;
        let lsb = (poc as i64).rem_euclid(max_lsb) as u32;
        let patched;
        let mut nal: &[u8];
        if let Some(r) = repl.get(&p.dec) {
            o.write_all(&START_CODE)?;
            nal = r;
        } else {
            o.write_all(&mm[first..sl.offset])?;
            nal = &mm[sl.offset..end];
        }
        if fnum != sl.frame_num || lsb != sl.poc_lsb {
            patched = patch_fn_poc(nal, sl, sps, fnum, lsb);
            nal = &patched;
        }
        o.write_all(nal)?;
        o.write_all(&mm[end..p.au_end])?;
    }
    o.flush()?;
    Ok(entries)
}

// ---------------------------------------------------------------------------
// dev commands (oracle: port/oracle.py)
// ---------------------------------------------------------------------------

fn print_line(v: serde_json::Value) {
    println!("{}", crate::records::canonical_text(&v));
}

type Pair = (usize, usize, std::result::Result<Rebuild, String>);

/// The rebuild candidates between consecutive periods: for each period g,
/// the last non-IDR reference picture paired with the next IDR.
fn pair_verdicts(spss: &Spss, pics: &[Picture], periods: &[Range<usize>]) -> Result<Vec<Pair>> {
    let mut out = Vec::new();
    for (k, d) in crate::explicit::pair_candidates(pics, periods, 1) {
        out.push((k, d, candidate(spss, pics, periods, k, d)?));
    }
    Ok(out)
}

/// `dev rebuild-candidates <es>`: `rebuild::candidate` for (last reference
/// picture of period g, IDR of period g+1) for every g, then `check_lists`
/// over the whole stream with all accepted candidates.
pub fn dev_rebuild_candidates(args: &[String]) -> Result<()> {
    use serde_json::json;
    let [es] = args else {
        bail!("usage: frame-dedup dev rebuild-candidates <file.h264>");
    };
    let (spss, _ppss, pics) = crate::stream::load(Path::new(es))?;
    let periods = crate::stream::period_ranges(&pics);
    let t0 = std::time::Instant::now();
    let acc = pair_verdicts(&spss, &pics, &periods)?;
    eprintln!("candidates {:.1} s", t0.elapsed().as_secs_f64());
    let mut rebuilds = Vec::new();
    for (k, d, v) in &acc {
        match v {
            Ok(r) => {
                print_line(json!({"K": k, "D": d, "r": {"K": r.k, "D": r.d, "B": r.b}, "reason": null}));
                rebuilds.push(r.clone());
            }
            Err(why) => print_line(json!({"K": k, "D": d, "r": null, "reason": why})),
        }
    }
    let chk = check_lists(&spss, &pics, 0, &rebuilds)?;
    let fails: Vec<serde_json::Value> = chk
        .fails
        .iter()
        .map(|f| {
            let val = |v: &FailVal| match v {
                FailVal::Int(i) => json!(i),
                FailVal::List(l) => json!(l),
            };
            json!({"dec": f.dec, "what": f.what, "want": val(&f.want), "got": val(&f.got)})
        })
        .collect();
    print_line(json!({"fails": fails, "n": chk.shadow.len()}));
    for (q, &src) in chk.shadow.iter().zip(&chk.src) {
        print_line(json!({"src": src, "fn": q.slices[0].frame_num, "poc": q.poc, "idr": q.idr,
                          "lists": [q.lists[0], q.lists[1]]}));
    }
    Ok(())
}

/// `dev write-rebuilt <es> <spec.json> <outprefix>`: for each segment
/// `{"p0", "p1", "repl": {dec: src dec}}` (IDR periods p0..=p1 of the stream)
/// apply every accepted rebuild between consecutive periods and write
/// `<outprefix>_<i>.h264`; prints the rebuilds, the entries and the size.
pub fn dev_write_rebuilt(args: &[String]) -> Result<()> {
    use serde_json::{Value, json};
    let [es, spec, prefix] = args else {
        bail!("usage: frame-dedup dev write-rebuilt <file.h264> <spec.json> <outprefix>");
    };
    let (spss, ppss, pics) = crate::stream::load(Path::new(es))?;
    let periods = crate::stream::period_ranges(&pics);
    let acc = pair_verdicts(&spss, &pics, &periods)?;
    let spec: Value = serde_json::from_str(&std::fs::read_to_string(spec)?)?;
    let file = File::open(es)?;
    let mm = unsafe { Mmap::map(&file)? };
    for (i, seg) in spec["segments"].as_array().ok_or_else(|| anyhow!("spec has no segments"))?.iter().enumerate() {
        let (p0, p1) = (seg["p0"].as_u64().unwrap() as usize, seg["p1"].as_u64().unwrap() as usize);
        let range = periods[p0].start..periods[p1].end;
        let rebuilds: Vec<Rebuild> = acc
            .iter()
            .filter_map(|(_, _, v)| v.as_ref().ok())
            .filter(|r| range.contains(&r.k) && range.contains(&r.d))
            .cloned()
            .collect();
        let mut repl: HashMap<usize, Vec<u8>> = HashMap::new();
        if let Some(m) = seg["repl"].as_object() {
            for (k, v) in m {
                let s = &pics[v.as_u64().unwrap() as usize].slices[0];
                repl.insert(k.parse()?, mm[s.offset..s.offset + s.size].to_vec());
            }
        }
        let out = format!("{prefix}_{i}.h264");
        let t0 = std::time::Instant::now();
        let entries = write_rebuilt(Path::new(es), &pics[range.clone()], range.start, &spss, &ppss, Path::new(&out), &rebuilds, &repl)?;
        eprintln!("segment {i}: {:.1} s", t0.elapsed().as_secs_f64());
        let size = std::fs::metadata(&out)?.len();
        print_line(json!({"i": i, "rebuilds": rebuilds.iter().map(|r| json!([r.k, r.d, r.b])).collect::<Vec<_>>(),
                          "size": size, "entries": entries.len()}));
        for e in &entries {
            print_line(json!([e.pic, e.i2, e.frame_num, e.poc]));
        }
    }
    Ok(())
}

/// `frame-dedup dev rebuild <scenario.json> --out rebuilt.h264 [--report r.json] [--workers N] [--work DIR]`:
/// run the anchor rebuild on the streams named in the scenario (written by
/// `python -m port.oracle repair-scenario`: `orig_es`, `rebuild.src_es`,
/// `rebuild.src_orig`, `rebuild.rbs`), write the rebuilt stream and the
/// `anchor_rebuilds` report section. Prints the timing.
pub fn dev_rebuild(args: &[String]) -> Result<()> {
    use serde_json::{Value, json};
    const USAGE: &str = "frame-dedup dev rebuild <scenario.json> --out rebuilt.h264 [--report r.json] [--workers N] [--work DIR]";
    let (mut scen, mut out, mut report, mut workers, mut work) = (None, None, None, 4usize, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |n: &str| it.next().cloned().ok_or_else(|| anyhow!("{n} needs a value"));
        match a.as_str() {
            "--out" => out = Some(val("--out")?),
            "--report" => report = Some(val("--report")?),
            "--workers" => workers = val("--workers")?.parse()?,
            "--work" => work = Some(val("--work")?),
            _ if scen.is_none() => scen = Some(a.clone()),
            _ => bail!("usage: {USAGE}"),
        }
    }
    let scen = scen.ok_or_else(|| anyhow!("usage: {USAGE}"))?;
    let out = out.ok_or_else(|| anyhow!("--out is required"))?;
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&scen)?)?;
    let s = |v: &Value, k: &str| v[k].as_str().map(std::path::PathBuf::from).ok_or_else(|| anyhow!("scenario has no {k}"));
    let rj = &j["rebuild"];
    if rj.is_null() {
        bail!("the scenario has no rebuild section");
    }
    let orig_es = s(&j, "orig_es")?;
    let src_es = s(rj, "src_es")?;
    let work = match work {
        Some(w) => std::path::PathBuf::from(w),
        None => s(&j, "work")?,
    };
    let ids = |v: &Value| -> Vec<usize> { v.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect()).unwrap_or_default() };
    let src_orig = ids(&rj["src_orig"]);
    let rbs: Vec<Rebuild> = rj["rbs"]
        .as_array()
        .ok_or_else(|| anyhow!("scenario has no rebuild.rbs"))?
        .iter()
        .map(|r| Rebuild { k: r["K"].as_u64().unwrap() as usize, d: r["D"].as_u64().unwrap() as usize, b: ids(&r["B"]) })
        .collect();
    crate::system::lower_priority();
    let log = crate::system::Logger::new(None)?;
    let (_spss, _ppss, orig_pics) = crate::stream::load(&orig_es)?;
    let (nspss, nppss, src_pics) = crate::stream::load(&src_es)?;
    if src_pics.len() != src_orig.len() {
        bail!("the source stream has {} pictures, the scenario maps {}", src_pics.len(), src_orig.len());
    }
    let t0 = Instant::now();
    let (orig_of_out, rb) = rebuild(&orig_es, &src_es, Path::new(&out), &nspss, &nppss, &orig_pics, &src_pics, &src_orig, &rbs, workers, &work, &log)?;
    let secs = t0.elapsed().as_secs_f64();
    if let Some(r) = &report {
        std::fs::write(r, crate::records::canonical_text(&json!({"anchor_rebuilds": rb, "orig_of_out": orig_of_out, "wall_seconds": secs})))?;
    }
    println!(
        "rebuild: {} pairs, {} leading B pictures, {} PCM MBs, {:+} bytes; {:.1} s; process peak {} MB",
        rb["pairs"],
        rb["leading_b_pictures"],
        rb["pcm_mbs"],
        rb["bytes_added"].as_i64().unwrap_or(0),
        secs,
        crate::system::peak_rss() >> 20
    );
    Ok(())
}

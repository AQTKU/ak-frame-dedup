//! Explicit reference control for streams with MMCO / list modification:
//! header planning and slice header rewriting. Port of `frame_dedup/explicit.py`.
//!
//! Encoders such as x264 with B-pyramid write reference B pictures that
//! unmark older pictures with MMCO 1, and reorder (and, for weighted
//! prediction, duplicate) entries of every P list with
//! ref_pic_list_modification. Dropping a reference picture from such a stream
//! breaks both. Instead of letting the decoder's default rules shift lists
//! after a drop, the headers of the kept pictures are regenerated so that the
//! decoder sees, for every kept picture, exactly the original reference lists
//! (a dropped picture D replaced by its substitute sigma(D)) and exactly the
//! original set of short-term references after marking. Headers are
//! re-serialised bit by bit; the CABAC slice data is appended unchanged.
//!
//! Pictures are named by their original decode index. Functions that look
//! pictures up take a slice `pics` and a `base`: the decode index of
//! `pics[0]` (Python's `_Renumbered`), 0 for a whole stream.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::ops::Range;

use anyhow::{Result, anyhow, bail};

use crate::bits::{BitWriter, escape, unescape};
use crate::h264hdr::{Mmco, Picture, Ppss, SliceHeader, SliceType, Spss, parse_slice};
use crate::mv::{RefPic, dpb_walk, init_lists, modify_list};
use crate::rebuild::Rebuild;

/// The requested substitution cannot be expressed in the headers (Python:
/// `PlanError`). Carried inside an `anyhow::Error`; callers that must tell it
/// from other failures use [`is_plan_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError(pub String);

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PlanError {}

/// An `anyhow::Error` holding a [`PlanError`].
pub fn plan_error(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(PlanError(msg.into()))
}

pub fn is_plan_error(e: &anyhow::Error) -> bool {
    e.downcast_ref::<PlanError>().is_some()
}

/// `{original decode id: its substitute's id or None}` for every dropped
/// reference picture (Python: `sigma`).
pub type Sigma = HashMap<usize, Option<usize>>;

/// One picture of the new stream in decode order (Python: an entry dict).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The original picture (decode index in the original stream).
    pub pic: usize,
    /// Whether it is an IDR in the new stream.
    pub idr: bool,
    /// Its POC in the new stream.
    pub poc: i32,
    /// An original IDR written as a non-IDR I picture.
    pub demote: bool,
    /// The original picture whose reference role it takes over (anchor
    /// rebuild: I2 takes K's); its `dpb_after` is the target.
    pub as_: Option<usize>,
}

/// The planned header of one entry (Python: the header dict).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderPlan {
    pub frame_num: u32,
    pub poc_lsb: u32,
    pub rplm: [Vec<(u32, u32)>; 2],
    /// `None`: sliding window; `Some(ops)`: adaptive marking with these ops.
    pub mmco: Option<Vec<Mmco>>,
    pub demote: bool,
    /// The target lists, as original ids.
    pub lists: [Vec<usize>; 2],
    /// The original header is reproduced exactly.
    pub same: bool,
}

/// True when the stream uses MMCO or ref list modification anywhere.
pub fn stream_is_explicit(pics: &[Picture]) -> bool {
    pics.iter().any(|p| {
        let sl = &p.slices[0];
        !sl.mmco.is_empty() || sl.adaptive || !sl.rplm[0].is_empty() || !sl.rplm[1].is_empty()
    })
}

// ---------------------------------------------------------------- bits

fn put_rplm(w: &mut BitWriter, t: SliceType, rplm: &[Vec<(u32, u32)>; 2]) {
    for lst in 0..2 {
        if (lst == 0 && !matches!(t, SliceType::I | SliceType::SI)) || (lst == 1 && t == SliceType::B) {
            let cmds = &rplm[lst];
            if cmds.is_empty() {
                w.put_bit(false);
                continue;
            }
            w.put_bit(true);
            for &(idc, v) in cmds {
                w.ue(idc);
                w.ue(v);
            }
            w.ue(3);
        }
    }
}

/// dec_ref_pic_marking of a non-IDR reference picture: `None` = sliding
/// window, `Some` = adaptive marking with these ops.
fn put_drpm(w: &mut BitWriter, mmco: &Option<Vec<Mmco>>) {
    let Some(ops) = mmco else {
        w.put_bit(false);
        return;
    };
    w.put_bit(true);
    for m in ops {
        w.ue(m.op);
        for &a in &m.args {
            w.ue(a);
        }
    }
    w.ue(0);
}

/// Return the slice NAL with a rewritten header (`nal` includes its header
/// byte). `h`: frame_num, poc_lsb, rplm, mmco (ignored for a non-reference
/// picture) and demote (an IDR becomes a non-IDR slice: nal_unit_type 1,
/// idr_pic_id removed, explicit marking). The CABAC slice data is appended
/// byte for byte after cabac_alignment_one_bits.
pub fn rewrite_slice(nal: &[u8], sl: &SliceHeader, spss: &Spss, ppss: &Ppss, h: &HeaderPlan) -> Result<Vec<u8>> {
    let pps = ppss.get(&sl.pps_id).ok_or_else(|| anyhow!("KeyError: PPS {}", sl.pps_id))?;
    let sps = spss.get(&pps.sps_id).ok_or_else(|| anyhow!("KeyError: SPS {}", pps.sps_id))?;
    let (lfn, lpoc) = (sps.log2_max_frame_num, sps.log2_max_poc_lsb);
    let rbsp = unescape(&nal[1..]);
    let data_byte = sl.data_bit / 8;
    if rbsp.len() < data_byte {
        bail!("IndexError: slice shorter than its header");
    }
    // the header bits after hdr_bits are cabac_alignment_one_bits
    for pos in sl.hdr_bits..data_byte * 8 {
        if (rbsp[pos >> 3] >> (7 - (pos & 7))) & 1 == 0 {
            bail!("AssertionError: cabac_alignment_one_bits expected");
        }
    }
    let idr = sl.nal_type == 5;
    let demote = h.demote;
    if demote && !idr {
        bail!("AssertionError: only an IDR can be demoted");
    }
    let ref_idc = (nal[0] >> 5) & 3;
    let mut w = BitWriter::with_capacity(data_byte + 8);
    w.put_bits_from(&rbsp, 0, sl.frame_num_bit);
    w.put(lfn, h.frame_num);
    if idr && !demote {
        w.put_bits_from(&rbsp, sl.frame_num_bit + lfn as usize, sl.poc_lsb_bit); // idr_pic_id
    }
    w.put(lpoc, h.poc_lsb);
    w.put_bits_from(&rbsp, sl.poc_lsb_bit + lpoc as usize, sl.rplm_bit);
    put_rplm(&mut w, sl.slice_type, &h.rplm);
    w.put_bits_from(&rbsp, sl.rplm_end_bit, sl.drpm_bit); // pred_weight_table
    if ref_idc != 0 {
        if idr && !demote {
            w.put_bits_from(&rbsp, sl.drpm_bit, sl.drpm_end_bit);
        } else {
            put_drpm(&mut w, &h.mmco);
        }
    }
    w.put_bits_from(&rbsp, sl.drpm_end_bit, sl.hdr_bits);
    let hdr_bits = w.len_bits();
    w.align_ones(); // cabac_alignment_one_bit
    let n_bits = w.len_bits();
    let mut new_rbsp = w.finish();
    new_rbsp.extend_from_slice(&rbsp[data_byte..]);
    let nal_type = if demote { 1 } else { sl.nal_type };
    let mut new = vec![(nal[0] & 0xe0) | nal_type];
    new.extend_from_slice(&escape(&new_rbsp));
    let chk = parse_slice(&new_rbsp[..new_rbsp.len().min(n_bits / 8 + 8)], nal_type, ref_idc, spss, ppss)?;
    macro_rules! same {
        ($($f:ident),*) => {$(
            if chk.$f != sl.$f {
                bail!("AssertionError: {} {:?} != {:?}", stringify!($f), chk.$f, sl.$f);
            }
        )*};
    }
    same!(first_mb, slice_type, pps_id, qp, deblock_idc, alpha_off, beta_off, cabac_init_idc, num_ref_l0,
          num_ref_l1, direct_spatial, weighted);
    if chk.frame_num != h.frame_num || chk.poc_lsb != h.poc_lsb {
        bail!("AssertionError: frame_num / poc_lsb do not read back");
    }
    if chk.rplm != h.rplm {
        bail!("AssertionError: ref_pic_list_modification does not read back");
    }
    if ref_idc != 0 && !(idr && !demote) {
        let want = h.mmco.clone().unwrap_or_default();
        if chk.mmco != want || chk.adaptive != h.mmco.is_some() {
            bail!("AssertionError: dec_ref_pic_marking does not read back");
        }
    }
    if chk.hdr_bits != hdr_bits || chk.data_bit != n_bits {
        bail!("AssertionError: header length does not read back");
    }
    Ok(new)
}

// ---------------------------------------------------------------- planning

fn py_list(v: &[usize]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

/// Python's repr of a pair of lists of ids.
pub fn py_lists(l: &[Vec<usize>; 2]) -> String {
    format!("[{}, {}]", py_list(&l[0]), py_list(&l[1]))
}

/// ref_pic_list_modification commands naming the pictures `want` (decode
/// ids) in order.
fn rplm_cmds(want: &[usize], cur_fn: u32, refs: &[RefPic], max_fn: u32) -> Result<Vec<(u32, u32)>> {
    let (cur_fn, max_fn) = (cur_fn as i64, max_fn as i64);
    let mut num_of: HashMap<usize, i64> = HashMap::new();
    for r in refs {
        let f = r.frame_num as i64;
        num_of.insert(r.dec, if f > cur_fn { f - max_fn } else { f });
    }
    let mut cmds = Vec::new();
    let mut pred = cur_fn;
    for d in want {
        let Some(&num) = num_of.get(d) else {
            bail!("KeyError: {d}");
        };
        let nowrap = if num < 0 { num + max_fn } else { num };
        let diff = nowrap - pred;
        if diff < 0 {
            cmds.push((0, (-diff - 1) as u32));
        } else if diff > 0 {
            cmds.push((1, (diff - 1) as u32));
        } else {
            cmds.push((0, (max_fn - 1) as u32)); // the same picture again: a full wrap
        }
        pred = nowrap;
    }
    Ok(cmds)
}

/// Commands for one list giving exactly `want`: the original commands if
/// they still do, no commands if the initial list already is `want`, else the
/// shortest explicit prefix.
fn plan_list(
    init: &[usize],
    want: &[usize],
    orig_cmds: &[(u32, u32)],
    n_active: usize,
    cur_fn: u32,
    refs: &[RefPic],
    max_fn: u32,
) -> Result<Vec<(u32, u32)>> {
    if !orig_cmds.is_empty() {
        // Python: `except ValueError`
        if let Ok(l) = modify_list(init, orig_cmds, n_active, cur_fn, refs, max_fn) {
            if l == want {
                return Ok(orig_cmds.to_vec());
            }
        }
    }
    if init == want {
        return Ok(Vec::new());
    }
    for k in 1..=n_active {
        let cmds = rplm_cmds(&want[..k.min(want.len())], cur_fn, refs, max_fn)?;
        if modify_list(init, &cmds, n_active, cur_fn, refs, max_fn)? == want {
            return Ok(cmds);
        }
    }
    bail!("AssertionError: a full explicit list always reproduces itself")
}

/// Follow `sigma` from `d` until it is not a dropped picture (None if it ends
/// in a dropped picture with no substitute).
fn sg(sigma: &Sigma, mut d: usize) -> Option<usize> {
    loop {
        match sigma.get(&d) {
            Some(&Some(next)) => d = next,
            Some(&None) => return None,
            None => return Some(d),
        }
    }
}

/// Plan the headers of a rewritten stream.
///
/// `entries`: the pictures of the new stream in decode order. `sigma`: for
/// every dropped reference picture (and K of an anchor rebuild) its
/// substitute, or None for a dropped reference picture that nothing reads.
/// `pics[d - base]` is the original picture `d`.
///
/// Returns the header plans aligned with `entries`, each with `same` = true
/// when the original header is reproduced exactly. Fails with a [`PlanError`]
/// when the substitution cannot be expressed.
pub fn plan(spss: &Spss, entries: &[Entry], sigma: &Sigma, pics: &[Picture], base: usize) -> Result<Vec<HeaderPlan>> {
    let Some(sps) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    let max_fn: u32 = 1 << sps.log2_max_frame_num;
    let max_lsb: i64 = 1 << sps.log2_max_poc_lsb;
    let max_refs = (sps.max_num_ref_frames as usize).max(1);

    let mut refs: Vec<RefPic> = Vec::new();
    let mut prev_ref_fn: u32 = 0;
    let mut out: Vec<HeaderPlan> = Vec::with_capacity(entries.len());
    for e in entries {
        let p = &pics[e.pic - base];
        let sl = &p.slices[0];
        let t = sl.slice_type;
        let cur = p.dec;
        let fnum: u32;
        if e.idr {
            refs.clear();
            fnum = 0;
        } else {
            fnum = (prev_ref_fn + 1) % max_fn;
        }
        let have: HashSet<usize> = refs.iter().map(|r| r.dec).collect();
        let mut want: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        if !e.demote {
            for l in 0..2 {
                for &x in &p.lists[l] {
                    match sg(sigma, x) {
                        Some(y) => want[l].push(y),
                        None => {
                            return Err(plan_error(format!("picture {cur}: reads a dropped picture that has no substitute")));
                        }
                    }
                }
            }
        }
        for l in 0..2 {
            let missing: Vec<usize> = want[l].iter().copied().filter(|x| !have.contains(x)).collect();
            if !missing.is_empty() {
                return Err(plan_error(format!("picture {cur}: list {l} needs {}, not in the DPB", py_list(&missing))));
            }
        }
        let n_act = [sl.num_ref_l0 as usize, sl.num_ref_l1 as usize];
        let init = init_lists(t, &refs, fnum, e.poc, n_act[0], n_act[1], max_fn);
        let mut rplm: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()];
        for l in 0..2 {
            if !want[l].is_empty() {
                rplm[l] = plan_list(&init[l], &want[l], &sl.rplm[l], n_act[l], fnum, &refs, max_fn)?;
            }
        }
        let mut h = HeaderPlan {
            frame_num: fnum,
            poc_lsb: (e.poc as i64).rem_euclid(max_lsb) as u32,
            rplm,
            mmco: None,
            demote: e.demote,
            lists: want,
            same: false,
        };
        if p.ref_idc != 0 {
            let src = &pics[e.as_.unwrap_or(p.dec) - base];
            let target: BTreeSet<usize> = src.dpb_after.iter().filter_map(|&x| sg(sigma, x)).collect();
            if !target.contains(&cur) {
                return Err(plan_error(format!("picture {cur}: not in its own target DPB")));
            }
            let mut allowed = have.clone();
            allowed.insert(cur);
            if !target.iter().all(|x| allowed.contains(x)) {
                let need: Vec<usize> = target.iter().copied().filter(|x| !allowed.contains(x)).collect();
                return Err(plan_error(format!("picture {cur}: target DPB needs {}", py_list(&need))));
            }
            if target.len() > max_refs {
                return Err(plan_error(format!("picture {cur}: target DPB too large")));
            }
            let num = |r: &RefPic| -> i64 {
                let f = r.frame_num as i64;
                if f > fnum as i64 { f - max_fn as i64 } else { f }
            };
            if !e.idr {
                let mut slide = refs.clone();
                if slide.len() >= max_refs {
                    // min(slide, key=num): the first of equal keys
                    let pos = (0..slide.len()).min_by_key(|&i| num(&slide[i])).unwrap();
                    slide.remove(pos);
                }
                let mut kept: BTreeSet<usize> = slide.iter().map(|r| r.dec).collect();
                kept.insert(cur);
                if kept == target && !sl.adaptive {
                    h.mmco = None;
                } else {
                    // unmark what the target lacks, in the original's op order
                    // where the picNums still match (untouched headers stay
                    // byte-identical)
                    let mut order: HashMap<i64, usize> = HashMap::new();
                    for (i, m) in sl.mmco.iter().enumerate() {
                        let a0 = m.args.first().copied().unwrap_or(0) as i64;
                        order.insert(p.frame_num as i64 - (a0 + 1), i);
                    }
                    let n_order = order.len();
                    let mut gone: Vec<&RefPic> = refs.iter().filter(|r| !target.contains(&r.dec)).collect();
                    gone.sort_by_key(|r| (order.get(&num(r)).copied().unwrap_or(n_order), -num(r)));
                    let mut ops = Vec::with_capacity(gone.len());
                    for r in gone {
                        let diff = fnum as i64 - num(r) - 1;
                        if diff < 0 {
                            bail!("AssertionError: picture {} has no smaller picNum than the current one", r.dec);
                        }
                        ops.push(Mmco { op: 1, args: vec![diff as u32] });
                    }
                    h.mmco = Some(ops);
                }
                refs.retain(|r| target.contains(&r.dec));
            }
            refs.push(RefPic { dec: cur, frame_num: fnum, poc: e.poc });
            prev_ref_fn = fnum;
        }
        h.same = !h.demote
            && fnum == sl.frame_num
            && h.poc_lsb == sl.poc_lsb
            && h.rplm == sl.rplm
            && (p.ref_idc == 0
                || e.idr
                || (h.mmco.is_none() && !sl.adaptive)
                || (h.mmco.is_some() && sl.adaptive && h.mmco.as_deref() == Some(&sl.mmco[..])));
        out.push(h);
    }
    check(spss, entries, &out, pics, base)?;
    Ok(out)
}

/// Re-simulate the planned headers with `dpb_walk`: every list, POC and DPB
/// must be what the plan asked for.
pub fn check(spss: &Spss, entries: &[Entry], hdrs: &[HeaderPlan], pics: &[Picture], base: usize) -> Result<()> {
    let mut shadow: Vec<Picture> = Vec::with_capacity(entries.len());
    for (e, h) in entries.iter().zip(hdrs) {
        let p = &pics[e.pic - base];
        let mut sl = p.slices[0].clone();
        sl.frame_num = h.frame_num;
        sl.poc_lsb = h.poc_lsb;
        sl.rplm = h.rplm.clone();
        sl.mmco = h.mmco.clone().unwrap_or_default();
        sl.adaptive = h.mmco.is_some();
        if h.demote {
            sl.nal_type = 1;
        }
        shadow.push(Picture { slices: vec![sl], idr: e.idr, ref_idc: p.ref_idc, ..Default::default() });
    }
    if let Err(ex) = dpb_walk(spss, &mut shadow) {
        return Err(plan_error(format!("rewritten headers do not simulate: {ex}")));
    }
    for ((e, h), q) in entries.iter().zip(hdrs).zip(&shadow) {
        if q.poc != e.poc {
            return Err(plan_error(format!("picture {}: POC {}, wanted {}", e.pic, q.poc, e.poc)));
        }
        let got: [Vec<usize>; 2] = [0, 1].map(|l| q.lists[l].iter().map(|&i| entries[i].pic).collect());
        if got != h.lists {
            return Err(plan_error(format!("picture {}: lists {}, wanted {}", e.pic, py_lists(&got), py_lists(&h.lists))));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- stream level

/// `free(dec)`: a non-IDR reference picture that no picture's list names.
/// Dropping it (sigma None) leaves every other picture bit-exact.
pub fn free_checker(pics: &[Picture]) -> impl Fn(usize) -> bool + Sync + '_ {
    let listed: HashSet<usize> = pics.iter().flat_map(|p| p.lists.iter().flatten().copied()).collect();
    move |d| !pics[d].idr && pics[d].ref_idc != 0 && !listed.contains(&d)
}

/// The new stream in decode order as [`plan`] entries, for `drop` (decode
/// ids, including every rebuild's K) and anchor rebuilds `[(K, D)]`: D is
/// demoted and written in K's slot with K's POC ("I2"), every POC of D's
/// former IDR period is shifted by POC(K) - POC(D), and the periods merge.
pub fn build_entries(pics: &[Picture], base: usize, drop: &HashSet<usize>, rebuilds: &[(usize, usize)]) -> Result<Vec<Entry>> {
    let by_k: HashMap<usize, usize> = rebuilds.iter().copied().collect();
    let by_d: HashSet<usize> = rebuilds.iter().map(|&(_, d)| d).collect();
    let mut out = Vec::with_capacity(pics.len());
    let mut off: i32 = 0;
    let mut i2poc: HashMap<usize, i32> = HashMap::new();
    for p in pics {
        let d = p.dec;
        if by_d.contains(&d) {
            let Some(&ip) = i2poc.get(&d) else {
                bail!("KeyError: {d} (the IDR of a rebuild comes before its K)");
            };
            off = ip - p.poc;
            continue;
        }
        if p.idr {
            off = 0;
        }
        if let Some(&dd) = by_k.get(&d) {
            let dp = &pics[dd - base];
            let poc = p.poc + off;
            i2poc.insert(dp.dec, poc);
            out.push(Entry { pic: dp.dec, idr: false, poc, demote: true, as_: Some(p.dec) });
            continue;
        }
        if drop.contains(&d) {
            continue;
        }
        out.push(Entry { pic: d, idr: p.idr, poc: p.poc + off, demote: false, as_: None });
    }
    Ok(out)
}

/// Header plan for the whole stream. `sigma`: {dropped reference: substitute
/// or None}; `rebuilds`: [(K, D)] (sigma[K] = D is added). Returns (entries,
/// headers).
pub fn plan_stream(
    spss: &Spss,
    pics: &[Picture],
    drop: &HashSet<usize>,
    sigma: &Sigma,
    rebuilds: &[(usize, usize)],
) -> Result<(Vec<Entry>, Vec<HeaderPlan>)> {
    plan_stream_at(spss, pics, 0, drop, sigma, rebuilds)
}

/// [`plan_stream`] over a run of pictures starting at decode id `base`.
pub fn plan_stream_at(
    spss: &Spss,
    pics: &[Picture],
    base: usize,
    drop: &HashSet<usize>,
    sigma: &Sigma,
    rebuilds: &[(usize, usize)],
) -> Result<(Vec<Entry>, Vec<HeaderPlan>)> {
    let mut sig = sigma.clone();
    for &(k, d) in rebuilds {
        sig.insert(k, Some(d));
    }
    let mut all_drop = drop.clone();
    all_drop.extend(rebuilds.iter().map(|&(k, _)| k));
    let entries = build_entries(pics, base, &all_drop, rebuilds)?;
    let mut hdrs = Vec::with_capacity(entries.len());
    // plan per new IDR period (independent, and errors name the period)
    let mut start = 0;
    for i in 1..=entries.len() {
        if i == entries.len() || entries[i].idr {
            hdrs.extend(plan(spss, &entries[start..i], &sig, pics, base)?);
            start = i;
        }
    }
    Ok((entries, hdrs))
}

/// Explicit-mode anchor rebuild eligibility: D an IDR, K a non-IDR reference
/// picture in the period before it, and the merged periods' headers
/// plannable. Every picture between K and D in decode order is a leading
/// picture (K's period ends there and D flushed the DPB, so nothing else can
/// read K). `periods` are the decode ranges of the IDR periods
/// ([`crate::stream::period_ranges`]). Returns `Ok(Ok(rebuild))`, or
/// `Ok(Err(reason))` when the pair is not eligible.
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
    let seg_range = periods[pk.period].start..periods[pd.period].end;
    let seg = &pics[seg_range.clone()];
    match plan_stream_at(spss, seg, seg_range.start, &HashSet::new(), &Sigma::new(), &[(k, d)]) {
        Ok(_) => {}
        Err(e) if is_plan_error(&e) => return Ok(Err(format!("headers: {e}"))),
        Err(e) => return Err(e),
    }
    Ok(Ok(Rebuild { k, d, b: (k + 1..d).collect() }))
}

// ---------------------------------------------------------------------------
// dev commands (oracle: port/oracle.py)
// ---------------------------------------------------------------------------

/// A drop / substitution / rebuild scenario read from JSON (written by
/// `python -m port.oracle make-scenario`, shared by both sides of the check).
pub struct Scenario {
    pub drop: HashSet<usize>,
    pub sigma: Sigma,
    pub rebuilds: Vec<(usize, usize)>,
    pub timing: Option<(u32, u32)>,
    /// (picture, picture whose NAL bytes are written in its place)
    pub repl: Vec<(usize, usize)>,
}

pub fn read_scenario(path: &str) -> Result<Scenario> {
    use serde_json::Value;
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)?;
    let ids = |x: &Value| -> Vec<usize> { x.as_array().map(|a| a.iter().filter_map(|y| y.as_u64()).map(|y| y as usize).collect()).unwrap_or_default() };
    let mut sigma = Sigma::new();
    if let Some(m) = v.get("sigma").and_then(|s| s.as_object()) {
        for (k, val) in m {
            sigma.insert(k.parse()?, val.as_u64().map(|x| x as usize));
        }
    }
    let rebuilds = v
        .get("rebuilds")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize)).collect())
        .unwrap_or_default();
    let timing = v.get("timing").and_then(|t| t.as_array()).map(|t| (t[0].as_u64().unwrap() as u32, t[1].as_u64().unwrap() as u32));
    let mut repl = Vec::new();
    if let Some(m) = v.get("repl").and_then(|s| s.as_object()) {
        for (k, val) in m {
            repl.push((k.parse()?, val.as_u64().unwrap() as usize));
        }
        repl.sort();
    }
    Ok(Scenario { drop: ids(&v["drop"]).into_iter().collect(), sigma, rebuilds, timing, repl })
}

fn mmco_json(m: &Option<Vec<Mmco>>) -> serde_json::Value {
    use serde_json::{Value, json};
    match m {
        None => Value::Null,
        Some(ops) => Value::Array(
            ops.iter()
                .map(|c| {
                    let mut v = vec![json!(c.op)];
                    v.extend(c.args.iter().map(|&a| json!(a)));
                    Value::Array(v)
                })
                .collect(),
        ),
    }
}

/// One JSONL line of a plan (entry and header).
pub fn plan_line(i: usize, e: &Entry, h: &HeaderPlan) -> String {
    use serde_json::json;
    let line = json!({
        "i": i,
        "e": {"pic": e.pic, "idr": e.idr, "poc": e.poc, "demote": e.demote, "as": e.as_},
        "h": {"frame_num": h.frame_num, "poc_lsb": h.poc_lsb, "rplm": [h.rplm[0], h.rplm[1]], "mmco": mmco_json(&h.mmco),
              "demote": h.demote, "lists": [h.lists[0], h.lists[1]], "same": h.same},
    });
    crate::records::canonical_text(&line)
}

fn dev_load(path: &str) -> Result<(Spss, Ppss, Vec<Picture>)> {
    crate::stream::load(std::path::Path::new(path))
}

/// Print a plan as JSONL, or the PlanError.
fn print_plan(res: Result<(Vec<Entry>, Vec<HeaderPlan>)>) -> Result<Option<(Vec<Entry>, Vec<HeaderPlan>)>> {
    use std::io::Write;
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, out.lock());
    match res {
        Ok((entries, hdrs)) => {
            writeln!(out, "{}", serde_json::json!({"entries": entries.len()}))?;
            for (i, (e, h)) in entries.iter().zip(&hdrs).enumerate() {
                writeln!(out, "{}", plan_line(i, e, h))?;
            }
            out.flush()?;
            Ok(Some((entries, hdrs)))
        }
        Err(e) if is_plan_error(&e) => {
            writeln!(out, "{}", crate::records::canonical_text(&serde_json::json!({"plan_error": e.to_string()})))?;
            out.flush()?;
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// `dev plan-stream <es> <scenario.json>`: the header plan as JSONL.
pub fn dev_plan_stream(args: &[String]) -> Result<()> {
    let [es, scen] = args else {
        bail!("usage: frame-dedup dev plan-stream <file.h264> <scenario.json>");
    };
    let (spss, _ppss, pics) = dev_load(es)?;
    let sc = read_scenario(scen)?;
    print_plan(plan_stream(&spss, &pics, &sc.drop, &sc.sigma, &sc.rebuilds))?;
    Ok(())
}

/// `dev write-explicit <es> <scenario.json> <out.h264>`: plan, write; prints
/// the counts and the output size as a JSON line.
pub fn dev_write_explicit(args: &[String]) -> Result<()> {
    let [es, scen, out] = args else {
        bail!("usage: frame-dedup dev write-explicit <file.h264> <scenario.json> <out.h264>");
    };
    let (spss, ppss, pics) = dev_load(es)?;
    let sc = read_scenario(scen)?;
    let t0 = std::time::Instant::now();
    let (entries, hdrs) = plan_stream(&spss, &pics, &sc.drop, &sc.sigma, &sc.rebuilds)?;
    let t_plan = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (n_rw, n_sps) = crate::stream::write_explicit(
        std::path::Path::new(es),
        &pics,
        &spss,
        &ppss,
        std::path::Path::new(out),
        &entries,
        &hdrs,
        sc.timing,
    )?;
    let t_write = t1.elapsed().as_secs_f64();
    let size = std::fs::metadata(out)?.len();
    println!("{}", serde_json::json!({"n_rw": n_rw, "n_sps": n_sps, "size": size, "entries": entries.len()}));
    eprintln!("plan {t_plan:.1} s, write {t_write:.1} s");
    Ok(())
}

/// The reference pictures of each IDR period, the candidate Ks for the pair
/// ending at the next IDR: the last `n` non-IDR reference pictures.
pub fn pair_candidates(pics: &[Picture], periods: &[Range<usize>], n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for g in 0..periods.len().saturating_sub(1) {
        let d = periods[g + 1].start;
        let ks: Vec<usize> = pics[periods[g].clone()].iter().filter(|p| p.ref_idc != 0 && !p.idr).map(|p| p.dec).collect();
        for &k in ks.iter().rev().take(n).collect::<Vec<_>>().into_iter().rev() {
            out.push((k, d));
        }
    }
    out
}

/// `dev explicit-candidates <es> [n]`: `explicit::candidate` for the last n
/// (default 3) reference pictures of every period paired with the next IDR,
/// then the `free_checker` verdict of every picture.
pub fn dev_explicit_candidates(args: &[String]) -> Result<()> {
    use serde_json::json;
    let (es, n) = match args {
        [es] => (es, 3usize),
        [es, n] => (es, n.parse()?),
        _ => bail!("usage: frame-dedup dev explicit-candidates <file.h264> [n]"),
    };
    let (spss, _ppss, pics) = dev_load(es)?;
    let periods = crate::stream::period_ranges(&pics);
    let t0 = std::time::Instant::now();
    for (k, d) in pair_candidates(&pics, &periods, n) {
        let line = match candidate(&spss, &pics, &periods, k, d)? {
            Ok(r) => json!({"K": k, "D": d, "r": {"K": r.k, "D": r.d, "B": r.b}, "reason": null}),
            Err(why) => json!({"K": k, "D": d, "r": null, "reason": why}),
        };
        println!("{}", crate::records::canonical_text(&line));
    }
    eprintln!("candidates {:.1} s", t0.elapsed().as_secs_f64());
    let free = free_checker(&pics);
    let freed: Vec<usize> = (0..pics.len()).filter(|&d| free(d)).collect();
    println!("{}", crate::records::canonical_text(&json!({"free": freed})));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rplm_cmds_wrap() {
        let refs = [
            RefPic { dec: 1, frame_num: 14, poc: 0 },
            RefPic { dec: 2, frame_num: 15, poc: 0 },
            RefPic { dec: 3, frame_num: 2, poc: 0 },
        ];
        // current frame_num 3: picNums are 14-16=-2, -1, 2
        let c = rplm_cmds(&[3, 2, 1, 3], 3, &refs, 16).unwrap();
        assert_eq!(c, vec![(0, 0), (1, 12), (0, 0), (0, 11)]); // from explicit._rplm_cmds
        let c = rplm_cmds(&[3, 3], 3, &refs, 16).unwrap();
        assert_eq!(c, vec![(0, 0), (0, 15)]); // the same picture again: a full wrap
    }

    #[test]
    fn plan_list_default_and_explicit() {
        let refs = [
            RefPic { dec: 1, frame_num: 1, poc: 2 },
            RefPic { dec: 2, frame_num: 2, poc: 4 },
            RefPic { dec: 3, frame_num: 3, poc: 6 },
        ];
        let init = init_lists(SliceType::P, &refs, 4, 8, 3, 0, 16);
        assert_eq!(init[0], vec![3, 2, 1]);
        assert!(plan_list(&init[0], &[3, 2, 1], &[], 3, 4, &refs, 16).unwrap().is_empty());
        let c = plan_list(&init[0], &[2, 3, 1], &[], 3, 4, &refs, 16).unwrap();
        assert_eq!(modify_list(&init[0], &c, 3, 4, &refs, 16).unwrap(), vec![2, 3, 1]);
        // the original commands are kept when they still give the list
        let c = plan_list(&init[0], &[2, 3, 1], &[(0, 1)], 3, 4, &refs, 16).unwrap();
        assert_eq!(c, vec![(0, 1)]);
    }

    const MMCO_CLIP: &str = r"C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port\clips\mmco.h264";

    /// Planning an unmodified stream reproduces every header bit for bit, and
    /// `rewrite_slice` with such a plan gives back the original NALs. Skipped
    /// when the clip is missing.
    #[test]
    fn unmodified_stream_plan_is_identity() {
        let path = std::path::Path::new(MMCO_CLIP);
        if !path.exists() {
            return;
        }
        let (spss, ppss, pics) = crate::stream::load(path).unwrap();
        assert!(stream_is_explicit(&pics));
        let (entries, hdrs) = plan_stream(&spss, &pics, &HashSet::new(), &Sigma::new(), &[]).unwrap();
        assert_eq!(entries.len(), pics.len());
        let data = std::fs::read(path).unwrap();
        for (p, h) in pics.iter().zip(&hdrs) {
            assert!(h.same, "picture {}", p.dec);
            let sl = &p.slices[0];
            let nal = &data[sl.offset..sl.offset + sl.size];
            let mut h2 = h.clone();
            h2.same = false;
            assert_eq!(rewrite_slice(nal, sl, &spss, &ppss, &h2).unwrap(), nal, "picture {}", p.dec);
        }
    }

    /// A reference drop with a substitute plans, and a drop that nothing can
    /// stand in for is a PlanError.
    #[test]
    fn plan_error_is_distinguishable() {
        let path = std::path::Path::new(MMCO_CLIP);
        if !path.exists() {
            return;
        }
        let (spss, _ppss, pics) = crate::stream::load(path).unwrap();
        let victim = pics.iter().find(|p| p.ref_idc != 0 && !p.idr && pics.iter().any(|q| q.lists.iter().any(|l| l.contains(&p.dec)))).unwrap().dec;
        let drop = HashSet::from([victim]);
        let mut sigma = Sigma::new();
        sigma.insert(victim, None);
        let e = plan_stream(&spss, &pics, &drop, &sigma, &[]).unwrap_err();
        assert!(is_plan_error(&e), "{e}");
        assert!(e.to_string().ends_with("reads a dropped picture that has no substitute"), "{e}");
    }

    #[test]
    fn bit_builders() {
        let mut w = BitWriter::new();
        put_rplm(&mut w, SliceType::P, &[vec![(0, 1), (1, 0)], vec![]]);
        put_drpm(&mut w, &Some(vec![Mmco { op: 1, args: vec![2] }]));
        // 1 | 1 010 | 1 1 | 1 | 1 011 | 1 -> ue(0)=1, ue(1)=010, ue(1)=010 ...
        let bits = w.len_bits();
        assert!(bits > 0);
        let mut w = BitWriter::new();
        put_drpm(&mut w, &None);
        assert_eq!(w.len_bits(), 1);
    }
}

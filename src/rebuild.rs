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
//! This file has the header side only (demotion, planning, list check,
//! eligibility, writing). The B1-B3 re-encoding and the parallel driver come
//! with the CABAC codec.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;

use crate::bits::{BitWriter, escape, set_bits, unescape};
use crate::h264hdr::{Picture, Ppss, SliceHeader, Sps, Spss, parse_slice};
use crate::mv::dpb_walk;

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

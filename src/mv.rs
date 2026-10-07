//! Reference structure and motion reconstruction. Port of `frame_dedup/h264mv.py`.
//!
//! * `dpb_walk` simulates the decoded picture buffer from slice headers only.
//! * `derive_motion` turns macroblock records into a per-4x4 motion field.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::h264hdr::{Picture, SliceType, Spss};
use crate::records::Mb;

/// A short-term reference picture as the list initialisation sees it
/// (Python: a dict with 'dec', 'frame_num' and 'poc').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefPic {
    pub dec: usize,
    pub frame_num: u32,
    pub poc: i32,
}

/// Python slice `v[a:b]` (negative-free): empty when a >= b, clamped to the length.
fn py_slice<T>(v: &[T], a: usize, b: usize) -> &[T] {
    let b = b.min(v.len());
    let a = a.min(b);
    &v[a..b]
}

/// Apply ref_pic_list_modification (short-term commands, 8.2.4.3.1) to an
/// initial list of decode ids. Errors where Python raises ValueError.
pub fn modify_list(
    lst: &[usize],
    cmds: &[(u32, u32)],
    n_active: usize,
    cur_fn: u32,
    refs: &[RefPic],
    max_fn: u32,
) -> Result<Vec<usize>> {
    let (cur_fn, max_fn) = (cur_fn as i64, max_fn as i64);
    let picnum = |fnum: i64| if fnum > cur_fn { fnum - max_fn } else { fnum };
    // dict comprehension: a later reference with the same picNum overwrites
    let mut by_num: HashMap<i64, usize> = HashMap::new();
    for r in refs {
        by_num.insert(picnum(r.frame_num as i64), r.dec);
    }
    // (lst + [None] * (n_active + 1))[:n_active + 1]
    let mut cur: Vec<Option<usize>> = lst.iter().map(|&x| Some(x)).collect();
    cur.resize(cur.len().max(n_active + 1), None);
    cur.truncate(n_active + 1);
    let mut pred = cur_fn;
    let mut idx = 0usize;
    for &(idc, v) in cmds {
        let v = v as i64;
        let nw = if idc == 0 {
            let nw = pred - (v + 1);
            if nw < 0 { nw + max_fn } else { nw }
        } else if idc == 1 {
            let nw = pred + v + 1;
            if nw >= max_fn { nw - max_fn } else { nw }
        } else {
            bail!("ref list modification idc {idc} (long-term) is not supported");
        };
        pred = nw;
        let num = if nw > cur_fn { nw - max_fn } else { nw };
        let Some(&pic) = by_num.get(&num) else {
            bail!("ref list modification names picNum {num}, which is not a short-term reference");
        };
        // lst = lst[:idx] + [pic] + lst[idx:n_active]
        let mut next: Vec<Option<usize>> = py_slice(&cur, 0, idx).to_vec();
        next.push(Some(pic));
        next.extend_from_slice(py_slice(&cur, idx, n_active));
        cur = next;
        idx += 1;
        // rest = [x for x in lst[idx:] if x != pic]
        let rest: Vec<Option<usize>> = py_slice(&cur, idx, cur.len()).iter().copied().filter(|&x| x != Some(pic)).collect();
        // lst = (lst[:idx] + rest + [None] * (n_active + 1))[:n_active + 1]
        let mut next: Vec<Option<usize>> = py_slice(&cur, 0, idx).to_vec();
        next.extend(rest);
        next.resize(next.len().max(n_active + 1), None);
        next.truncate(n_active + 1);
        cur = next;
    }
    // lst = lst[:n_active]; drop trailing Nones. (Nones can only trail:
    // every command puts one picture at idx, so entries before idx are set.)
    cur.truncate(n_active);
    while cur.last() == Some(&None) {
        cur.pop();
    }
    Ok(cur.into_iter().flatten().collect())
}

/// Initial reference lists (8.2.4.2, frames, short-term only) as decode ids,
/// truncated to the active sizes.
pub fn init_lists(
    t: SliceType,
    refs: &[RefPic],
    cur_fn: u32,
    cur_poc: i32,
    n0: usize,
    n1: usize,
    max_fn: u32,
) -> [Vec<usize>; 2] {
    match t {
        SliceType::P => {
            let (cur_fn, max_fn) = (cur_fn as i64, max_fn as i64);
            let picnum = |r: &RefPic| {
                let f = r.frame_num as i64;
                if f > cur_fn { f - max_fn } else { f }
            };
            // sorted(refs, key=picnum, reverse=True): stable, ties keep list order
            let mut sorted: Vec<&RefPic> = refs.iter().collect();
            sorted.sort_by(|a, b| picnum(b).cmp(&picnum(a)));
            [sorted.iter().map(|r| r.dec).take(n0).collect(), Vec::new()]
        }
        SliceType::B => {
            let mut before: Vec<&RefPic> = refs.iter().filter(|r| r.poc < cur_poc).collect();
            before.sort_by(|a, b| b.poc.cmp(&a.poc));
            let mut after: Vec<&RefPic> = refs.iter().filter(|r| r.poc > cur_poc).collect();
            after.sort_by(|a, b| a.poc.cmp(&b.poc));
            let l0: Vec<usize> = before.iter().chain(after.iter()).map(|r| r.dec).collect();
            let mut l1: Vec<usize> = after.iter().chain(before.iter()).map(|r| r.dec).collect();
            if l1.len() > 1 && l1 == l0 {
                l1.swap(0, 1);
            }
            [l0.into_iter().take(n0).collect(), l1.into_iter().take(n1).collect()]
        }
        _ => [Vec::new(), Vec::new()],
    }
}

/// Header-only DPB simulation: POC, frame_num, final reference lists
/// (initial lists + ref_pic_list_modification) and reference marking
/// (sliding window or MMCO 1). Also sets `dpb_after`: the decode ids of the
/// short-term references after the picture is marked. Errors where Python
/// raises ValueError.
pub fn dpb_walk(spss: &Spss, pics: &mut [Picture]) -> Result<()> {
    let Some(sps) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    if sps.log2_max_frame_num > 31 {
        bail!("log2_max_frame_num {} is not supported", sps.log2_max_frame_num);
    }
    if sps.poc_type != 0 {
        bail!("KeyError: 'log2_max_poc_lsb' (POC type {})", sps.poc_type);
    }
    if sps.log2_max_poc_lsb > 31 {
        bail!("log2_max_poc_lsb {} is not supported", sps.log2_max_poc_lsb);
    }
    let max_fn = 1u32 << sps.log2_max_frame_num;
    let max_lsb = 1i64 << sps.log2_max_poc_lsb;
    let max_refs = (sps.max_num_ref_frames as usize).max(1);
    let mut refs: Vec<RefPic> = Vec::new(); // short-term reference pictures
    let (mut prev_msb, mut prev_lsb) = (0i64, 0i64);
    for (d, p) in pics.iter_mut().enumerate() {
        let sl = &p.slices[0];
        p.dec = d;
        if p.idr {
            refs.clear();
            prev_msb = 0;
            prev_lsb = 0;
        }
        let lsb = sl.poc_lsb as i64;
        let msb = if lsb < prev_lsb && prev_lsb - lsb >= max_lsb / 2 {
            prev_msb + max_lsb
        } else if lsb > prev_lsb && lsb - prev_lsb > max_lsb / 2 {
            prev_msb - max_lsb
        } else {
            prev_msb
        };
        let top = msb + lsb;
        let bottom = top + sl.delta_poc_bottom.unwrap_or(0) as i64;
        p.poc = top.min(bottom) as i32;
        p.frame_num = sl.frame_num;
        let mut lists = init_lists(sl.slice_type, &refs, sl.frame_num, p.poc, sl.num_ref_l0 as usize, sl.num_ref_l1 as usize, max_fn);
        for l in 0..2 {
            if !sl.rplm[l].is_empty() {
                let n_active = [sl.num_ref_l0, sl.num_ref_l1][l] as usize;
                lists[l] = modify_list(&lists[l], &sl.rplm[l], n_active, sl.frame_num, &refs, max_fn)?;
            }
        }
        p.lists = lists;
        // reference marking (sliding window, or MMCO 1: unmark a short-term picture)
        if p.ref_idc != 0 {
            let cur_fn = sl.frame_num as i64;
            let wrap = |r: &RefPic| {
                let f = r.frame_num as i64;
                if f > cur_fn { f - max_fn as i64 } else { f }
            };
            if sl.long_term_ref == Some(true) {
                bail!("long-term references are not supported");
            }
            if sl.adaptive || !sl.mmco.is_empty() {
                for m in &sl.mmco {
                    if m.op != 1 {
                        bail!("MMCO {} is not supported", m.op);
                    }
                    let num = cur_fn - (m.args[0] as i64 + 1);
                    let Some(pos) = refs.iter().position(|r| wrap(r) == num) else {
                        bail!("MMCO 1 names picNum {num}, which is not a short-term reference");
                    };
                    refs.remove(pos);
                }
            } else if !p.idr && refs.len() >= max_refs {
                // min(refs, key=wrap): the first of equal keys
                let pos = (0..refs.len()).min_by_key(|&i| wrap(&refs[i])).unwrap();
                refs.remove(pos);
            }
            refs.push(RefPic { dec: d, frame_num: p.frame_num, poc: p.poc });
            if refs.len() > max_refs {
                bail!("decoded picture buffer overflow at picture {d}");
            }
            prev_msb = msb;
            prev_lsb = lsb;
        }
        p.dpb_after = refs.iter().map(|r| r.dec).collect();
    }
    Ok(())
}

/// Per-4x4 motion field of one picture, stored row-major (index `gy * w4 + gx`).
/// Python's `Motion` and `CompactMotion` in one; the storage widths are
/// those of `CompactMotion`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Motion {
    pub w4: usize,
    pub h4: usize,
    /// Reference index per list, -1 where the list is unused.
    pub refidx: [Vec<i8>; 2],
    pub mv: [Vec<[i16; 2]>; 2],
    /// Decode index of the reference picture per list (in the stream whose
    /// `lists` were used), -1 for none.
    pub refpic: [Vec<i32>; 2],
    pub intra: Vec<u8>,
    pub direct: Vec<u8>,
}

impl Motion {
    pub fn new(mb_w: usize, mb_h: usize) -> Motion {
        let (w4, h4) = (mb_w * 4, mb_h * 4);
        let n = w4 * h4;
        Motion {
            w4,
            h4,
            refidx: [vec![-1; n], vec![-1; n]],
            mv: [vec![[0, 0]; n], vec![[0, 0]; n]],
            refpic: [vec![-1; n], vec![-1; n]],
            intra: vec![0; n],
            direct: vec![0; n],
        }
    }

    /// The motion field of an all-intra picture.
    pub fn all_intra(mb_w: usize, mb_h: usize) -> Motion {
        let mut m = Motion::new(mb_w, mb_h);
        m.intra.fill(1);
        m
    }

    #[inline]
    pub fn idx(&self, gx: usize, gy: usize) -> usize {
        gy * self.w4 + gx
    }
}

/// Reconstruct motion for a parsed picture (Python `derive_motion`).
/// `pics` is indexed by decode index and supplies POCs; `col` is the motion
/// of RefPicList1[0] (needed for B pictures with direct blocks). With
/// `target`, explicit partitions take their mvs from the target field and
/// their mvds are solved and written into `mbs`.
pub fn derive_motion(
    _pic: &Picture,
    _mbs: &mut [Mb],
    _mb_w: usize,
    _mb_h: usize,
    _pics: &[Picture],
    _col: Option<&Motion>,
    _target: Option<&Motion>,
) -> Motion {
    todo!("port h264mv.derive_motion")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::h264hdr::{Mmco, SliceHeader, Sps};
    use crate::stream::crc32;

    fn refs4() -> Vec<RefPic> {
        [(3, 14, 28), (5, 15, 20), (6, 2, 36), (9, 1, 30)]
            .iter()
            .map(|&(dec, frame_num, poc)| RefPic { dec, frame_num, poc })
            .collect()
    }

    /// Expected values from frame_dedup.h264mv.modify_list.
    #[test]
    fn modify_list_cases() {
        let r = refs4();
        assert_eq!(modify_list(&[3, 5, 6, 9], &[(0, 0)], 4, 3, &r, 16).unwrap(), vec![6, 3, 5, 9]);
        assert_eq!(modify_list(&[3, 5, 6, 9], &[(0, 1), (1, 0)], 3, 3, &r, 16).unwrap(), vec![9, 6, 3]);
        let e = modify_list(&[3, 5], &[(0, 2), (0, 0), (1, 5)], 2, 3, &r, 16).unwrap_err();
        assert_eq!(e.to_string(), "ref list modification names picNum 0, which is not a short-term reference");
        assert!(modify_list(&[6, 3], &[(1, 0)], 2, 15, &r, 16).is_err());
        let e = modify_list(&[3], &[(2, 0)], 2, 3, &r, 16).unwrap_err();
        assert_eq!(e.to_string(), "ref list modification idc 2 (long-term) is not supported");
    }

    /// The same pseudo-random cases as the Python script (an LCG, no `random`
    /// module); the chained CRC of the results comes from the Python functions.
    #[test]
    fn lists_match_python_corpus() {
        let mut s = 777u32;
        let mut rand = |n: u32| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 16) % n
        };
        let j = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let (mut chain, mut n_err) = (0u32, 0);
        for _ in 0..5000 {
            let start = rand(16);
            let mut rs: Vec<RefPic> = Vec::new();
            for i in 0..16 {
                let f = (start + i) % 16;
                if rand(3) == 0 {
                    let poc = (rand(13) as i32 - 6) * 2;
                    rs.push(RefPic { dec: 100 + rs.len(), frame_num: f, poc });
                }
            }
            let cur = rand(16);
            let n = 1 + rand(5) as usize;
            let t = [SliceType::P, SliceType::B, SliceType::I, SliceType::SP][rand(4) as usize];
            let cp = (rand(13) as i32 - 6) * 2;
            let n1 = rand(5) as usize;
            let il = init_lists(t, &rs, cur, cp, n, n1, 16);
            let mut cmds = Vec::new();
            let mut pred = cur as i64;
            for _ in 0..rand(5) {
                if !rs.is_empty() && rand(10) < 8 {
                    let fnum = rs[rand(rs.len() as u32) as usize].frame_num as i64;
                    let idc = rand(2);
                    let v = if idc == 0 { (pred - fnum - 1).rem_euclid(16) } else { (fnum - pred - 1).rem_euclid(16) };
                    pred = fnum;
                    cmds.push((idc, v as u32));
                } else {
                    let idc = [0, 1, 2][rand(3) as usize];
                    let v = rand(18);
                    if idc == 0 {
                        pred -= v as i64 + 1;
                        if pred < 0 {
                            pred += 16;
                        }
                    } else if idc == 1 {
                        pred += v as i64 + 1;
                        if pred >= 16 {
                            pred -= 16;
                        }
                    }
                    cmds.push((idc, v));
                }
            }
            let res = match modify_list(&il[0], &cmds, n, cur, &rs, 16) {
                Ok(l) => j(&l),
                Err(_) => {
                    n_err += 1;
                    "ERR".to_string()
                }
            };
            let text = format!("{}|{}|{}", j(&il[0]), j(&il[1]), res);
            // zlib.crc32(data, chain): continue from the previous value
            chain = crc32_continue(chain, text.as_bytes());
        }
        assert_eq!((chain, n_err), (0x3a1a_20db, 1315));
    }

    /// zlib.crc32(data, value)
    fn crc32_continue(prev: u32, data: &[u8]) -> u32 {
        let mut c = !prev;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }

    #[test]
    fn crc32_helper_agrees() {
        assert_eq!(crc32_continue(0, b"123456789"), crc32(b"123456789"));
    }

    fn sps(max_refs: u32) -> Spss {
        let s = Sps {
            profile_idc: 100,
            level_idc: 40,
            id: 0,
            chroma_format_idc: 1,
            separate_colour_plane: false,
            log2_max_frame_num: 4,
            poc_type: 0,
            log2_max_poc_lsb: 4,
            delta_pic_order_always_zero: false,
            max_num_ref_frames: max_refs,
            gaps_allowed: false,
            mb_w: 4,
            mb_h: 4,
            frame_mbs_only: true,
            mbaff: false,
            direct_8x8_inference: true,
            crop: [0; 4],
            vui: None,
        };
        let mut m = Spss::new();
        m.insert(0, s);
        m
    }

    fn pic(idr: bool, ref_idc: u8, t: SliceType, frame_num: u32, poc_lsb: u32, n0: u32, mmco: Vec<Mmco>) -> Picture {
        let sl = SliceHeader {
            first_mb: 0,
            slice_type: t,
            pps_id: 0,
            frame_num_bit: 0,
            frame_num,
            field_pic: false,
            bottom_field: None,
            idr_pic_id_bit: None,
            idr_pic_id: None,
            poc_lsb_bit: 0,
            poc_lsb,
            delta_poc_bottom: None,
            direct_spatial: None,
            num_ref_l0: n0,
            num_ref_l1: 0,
            rplm: [Vec::new(), Vec::new()],
            rplm_bit: 0,
            rplm_end_bit: 0,
            weighted: false,
            adaptive: !mmco.is_empty(),
            mmco,
            no_output_prior: None,
            long_term_ref: None,
            drpm_bit: 0,
            drpm_end_bit: 0,
            cabac_init_idc: 0,
            qp: 26,
            deblock_idc: 0,
            alpha_off: 0,
            beta_off: 0,
            hdr_bits: 0,
            data_bit: 0,
            nal_type: if idr { 5 } else { 1 },
            ref_idc,
            offset: 0,
            size: 0,
        };
        Picture { slices: vec![sl], idr, ref_idc, ..Default::default() }
    }

    #[test]
    fn dpb_walk_sliding_window_and_pocs() {
        let spss = sps(2);
        // IDR, then P P P (reference), a non-reference P, and POC lsb wrapping (16)
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 4, 2, vec![]),
            pic(false, 2, SliceType::P, 2, 8, 2, vec![]),
            pic(false, 0, SliceType::P, 3, 12, 2, vec![]),
            pic(false, 2, SliceType::P, 3, 0, 2, vec![]),
        ];
        dpb_walk(&spss, &mut pics).unwrap();
        assert_eq!(pics.iter().map(|p| p.poc).collect::<Vec<_>>(), vec![0, 4, 8, 12, 16]);
        assert_eq!(pics[2].lists[0], vec![1, 0]);
        assert_eq!(pics[2].dpb_after, vec![1, 2]);
        assert_eq!(pics[3].lists[0], vec![2, 1]);
        assert_eq!(pics[3].dpb_after, vec![1, 2]); // non-reference: unchanged
        assert_eq!(pics[4].dpb_after, vec![2, 4]);
    }

    #[test]
    fn dpb_walk_errors() {
        let spss = sps(2);
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 4, 1, vec![Mmco { op: 2, args: vec![0] }]),
        ];
        assert_eq!(dpb_walk(&spss, &mut pics).unwrap_err().to_string(), "MMCO 2 is not supported");
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 4, 1, vec![Mmco { op: 1, args: vec![5] }]),
        ];
        assert_eq!(
            dpb_walk(&spss, &mut pics).unwrap_err().to_string(),
            "MMCO 1 names picNum -5, which is not a short-term reference"
        );
        // MMCO 1 removes the IDR, then the buffer holds one picture: no sliding window afterwards
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 4, 1, vec![Mmco { op: 1, args: vec![0] }]),
        ];
        dpb_walk(&spss, &mut pics).unwrap();
        assert_eq!(pics[1].dpb_after, vec![1]);
        // an adaptive marking with no commands skips the sliding window: overflow
        let spss1 = sps(1);
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 4, 1, vec![]),
        ];
        pics[1].slices[0].adaptive = true;
        assert_eq!(dpb_walk(&spss1, &mut pics).unwrap_err().to_string(), "decoded picture buffer overflow at picture 1");
    }
}

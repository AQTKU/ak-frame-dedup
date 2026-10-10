//! Reference structure and motion reconstruction. Port of `frame_dedup/h264mv.py`.
//!
//! * `dpb_walk` simulates the decoded picture buffer from slice headers only.
//! * `derive_motion` turns macroblock records into a per-4x4 motion field.

use std::collections::HashMap;

use anyhow::{Result, anyhow, bail};

use crate::h264hdr::{Picture, SliceType, Spss};
use crate::records::{B_SUB, B_TYPES, Mb, Mvds, P_SUB, Part, Pred, SubShape};

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

    /// Canonical digests for the oracle comparison (`dev motion`, port/oracle.py
    /// `motion`): the SHA-256 of the little-endian packed arrays, row-major
    /// by 4x4 block, and the same over their concatenation. Fields: intra,
    /// direct, refidx (L0 then L1), mv (L0 then L1; x, y as i16), refpic
    /// (L0 then L1; i32, -1 where Python has None). Returns (whole, [intra,
    /// direct, refidx, mv, refpic]).
    pub fn canonical_digests(&self) -> (String, [String; 5]) {
        use crate::picture::{hex, sha256};
        let mut bufs: [Vec<u8>; 5] = Default::default();
        bufs[0] = self.intra.clone();
        bufs[1] = self.direct.clone();
        for l in 0..2 {
            bufs[2].extend(self.refidx[l].iter().map(|&v| v as u8));
            for v in &self.mv[l] {
                bufs[3].extend_from_slice(&v[0].to_le_bytes());
                bufs[3].extend_from_slice(&v[1].to_le_bytes());
            }
            for v in &self.refpic[l] {
                bufs[4].extend_from_slice(&v.to_le_bytes());
            }
        }
        let all: Vec<u8> = bufs.iter().flatten().copied().collect();
        (hex(&sha256(&all)), std::array::from_fn(|i| hex(&sha256(&bufs[i]))))
    }
}

// ===========================================================================
// derive_motion
// ===========================================================================

/// Reference index and mv of a neighbouring 4x4 block (Python `(ref, (x, y))`).
type Nb = (i32, [i32; 2]);
/// An unavailable neighbour: Python's `z = (-1, (0, 0))`.
const NB_NONE: Nb = (-1, [0, 0]);

fn median(a: i32, b: i32, c: i32) -> i32 {
    a.min(b).max(a.max(b).min(c))
}

/// Partition rectangles (x, y, w, h in 4x4 units, relative to the MB) in
/// Python's partition order; the first block of each is `mb_parts(part)[p][0]`.
fn mb_rects(part: Part) -> &'static [(i32, i32, i32, i32)] {
    match part {
        Part::P16x16 => &[(0, 0, 4, 4)],
        Part::P16x8 => &[(0, 0, 4, 2), (0, 2, 4, 2)],
        Part::P8x16 => &[(0, 0, 2, 4), (2, 0, 2, 4)],
        _ => unreachable!("mb_rects({part:?})"),
    }
}

/// Sub-partition rectangles relative to the 8x8 quadrant (`sub_parts(shape)`).
fn sub_rects(shape: SubShape) -> &'static [(i32, i32, i32, i32)] {
    match shape {
        SubShape::S8x8 => &[(0, 0, 2, 2)],
        SubShape::S8x4 => &[(0, 0, 2, 1), (0, 1, 2, 1)],
        SubShape::S4x8 => &[(0, 0, 1, 2), (1, 0, 1, 2)],
        SubShape::S4x4 => &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)],
        SubShape::Direct => unreachable!("sub_rects of a direct sub-macroblock"),
    }
}

/// Bit mask over the 16 4x4 blocks of an MB (bit `ly * 4 + lx`) for a rectangle.
fn rect_mask(lx: i32, ly: i32, w: i32, h: i32) -> u16 {
    let mut m = 0u16;
    for y in ly..ly + h {
        for x in lx..lx + w {
            m |= 1 << (y * 4 + x);
        }
    }
    m
}

/// Access to the macroblock records for `derive`: read-only, or read-write
/// when target mode has to write solved mvds.
trait Recs {
    fn recs(&self) -> &[Mb];
    fn recs_mut(&mut self) -> Option<&mut [Mb]>;
}

impl Recs for &[Mb] {
    fn recs(&self) -> &[Mb] {
        self
    }
    fn recs_mut(&mut self) -> Option<&mut [Mb]> {
        None
    }
}

impl Recs for &mut [Mb] {
    fn recs(&self) -> &[Mb] {
        self
    }
    fn recs_mut(&mut self) -> Option<&mut [Mb]> {
        Some(self)
    }
}

struct Derive<'a> {
    m: Motion,
    w4: i32,
    h4: i32,
    pic: &'a Picture,
    pics: &'a [Picture],
    col: Option<&'a Motion>,
    /// The MB being derived.
    cmx: i32,
    cmy: i32,
    /// 4x4 blocks of the current MB already assigned (Python's `done` set).
    done: u16,
}

impl<'a> Derive<'a> {
    /// Python `avail`: inside the picture and already decoded.
    #[inline]
    fn avail(&self, gx: i32, gy: i32) -> bool {
        if gx < 0 || gy < 0 || gx >= self.w4 || gy >= self.h4 {
            return false;
        }
        let (mx, my) = (gx >> 2, gy >> 2);
        if mx == self.cmx && my == self.cmy {
            return self.done >> ((gy & 3) * 4 + (gx & 3)) & 1 != 0;
        }
        my < self.cmy || (my == self.cmy && mx < self.cmx)
    }

    #[inline]
    fn nbr(&self, gx: i32, gy: i32, lst: usize) -> Option<Nb> {
        if !self.avail(gx, gy) {
            return None;
        }
        let i = (gy * self.w4 + gx) as usize;
        let mv = self.m.mv[lst][i];
        Some((self.m.refidx[lst][i] as i32, [mv[0] as i32, mv[1] as i32]))
    }

    /// A, B and C (or D) of the partition at (gx, gy), `w` blocks wide.
    #[inline]
    fn neighbours(&self, gx: i32, gy: i32, w: i32, lst: usize) -> (Nb, Nb, Nb) {
        let a = self.nbr(gx - 1, gy, lst);
        let mut b = self.nbr(gx, gy - 1, lst);
        let mut c = self.nbr(gx + w, gy - 1, lst);
        if c.is_none() {
            c = self.nbr(gx - 1, gy - 1, lst);
        }
        if b.is_none() && c.is_none() && a.is_some() {
            b = a;
            c = a;
        }
        (a.unwrap_or(NB_NONE), b.unwrap_or(NB_NONE), c.unwrap_or(NB_NONE))
    }

    /// Motion vector prediction (8.4.1.3) with the 16x8 / 8x16 directional rules.
    fn mvp(&self, gx: i32, gy: i32, w: i32, lst: usize, rf: i32, shape: Option<Part>, part: usize) -> [i32; 2] {
        let (a, b, c) = self.neighbours(gx, gy, w, lst);
        match shape {
            Some(Part::P16x8) => {
                if part == 0 && b.0 == rf {
                    return b.1;
                }
                if part == 1 && a.0 == rf {
                    return a.1;
                }
            }
            Some(Part::P8x16) => {
                if part == 0 && a.0 == rf {
                    return a.1;
                }
                if part == 1 && c.0 == rf {
                    return c.1;
                }
            }
            _ => {}
        }
        let n_match = (a.0 == rf) as u8 + (b.0 == rf) as u8 + (c.0 == rf) as u8;
        if n_match == 1 {
            return if a.0 == rf {
                a.1
            } else if b.0 == rf {
                b.1
            } else {
                c.1
            };
        }
        [median(a.1[0], b.1[0], c.1[0]), median(a.1[1], b.1[1], c.1[1])]
    }

    /// Assign a rectangle of 4x4 blocks (global coordinates) for one list.
    #[inline]
    fn setblk(&mut self, gx: i32, gy: i32, w: i32, h: i32, lst: usize, rf: i32, mv: [i32; 2], direct: u8) -> Result<()> {
        let refpic = if rf >= 0 {
            match self.pic.lists[lst].get(rf as usize) {
                Some(&p) => p as i32,
                None => bail!("list index out of range"),
            }
        } else {
            -1
        };
        let mv16 = [mv[0] as i16, mv[1] as i16];
        for y in gy..gy + h {
            let row = (y * self.w4) as usize;
            for x in gx..gx + w {
                let i = row + x as usize;
                self.m.refidx[lst][i] = rf as i8;
                self.m.mv[lst][i] = mv16;
                self.m.refpic[lst][i] = refpic;
                self.m.direct[i] = direct;
            }
        }
        Ok(())
    }

    /// Record a rectangle (global coordinates inside the current MB) as assigned.
    #[inline]
    fn mark(&mut self, gx: i32, gy: i32, w: i32, h: i32) {
        self.done |= rect_mask(gx - self.cmx * 4, gy - self.cmy * 4, w, h);
    }

    fn col_motion(&self) -> Result<&'a Motion> {
        match self.col {
            Some(c) => Ok(c),
            None => bail!("AttributeError: 'NoneType' object has no attribute 'intra' (co-located motion missing)"),
        }
    }

    /// Temporal direct for one 8x8 quadrant (direct_8x8_inference).
    fn temporal_direct(&mut self, i8: i32) -> Result<()> {
        let cm = self.col_motion()?;
        let pic = self.pic;
        let lists = &pic.lists;
        let Some(&col_dec) = lists[1].first() else { bail!("list index out of range") };
        let (ox, oy) = (self.cmx * 4 + (i8 % 2) * 2, self.cmy * 4 + (i8 / 2) * 2);
        // corner 4x4 of the co-located MB for this quadrant
        let cx = self.cmx * 4 + if i8 % 2 == 1 { 3 } else { 0 };
        let cy = self.cmy * 4 + if i8 / 2 == 1 { 3 } else { 0 };
        let ci = (cy * self.w4 + cx) as usize;
        let (mvcol, refcol): ([i32; 2], i32) = if cm.intra[ci] != 0 {
            ([0, 0], -1)
        } else if cm.refidx[0][ci] >= 0 {
            let v = cm.mv[0][ci];
            ([v[0] as i32, v[1] as i32], cm.refpic[0][ci])
        } else {
            let v = cm.mv[1][ci];
            ([v[0] as i32, v[1] as i32], cm.refpic[1][ci])
        };
        let ref0 = if refcol < 0 {
            0
        } else {
            match lists[0].iter().position(|&x| x as i32 == refcol) {
                Some(p) => p,
                None => bail!("ValueError: {refcol} is not in list"),
            }
        };
        let Some(&p0) = lists[0].get(ref0) else { bail!("list index out of range") };
        let poc0 = self.pics[p0].poc;
        let poc1 = self.pics[col_dec].poc;
        let tb = (self.pic.poc - poc0).clamp(-128, 127);
        let td = (poc1 - poc0).clamp(-128, 127);
        let (mv0, mv1);
        if td == 0 {
            mv0 = mvcol;
            mv1 = [0, 0];
        } else {
            // C integer division (truncates toward zero)
            let tx = (16384 + td.abs() / 2) / td;
            let dsf = ((tb * tx + 32) >> 6).clamp(-1024, 1023);
            mv0 = [(dsf * mvcol[0] + 128) >> 8, (dsf * mvcol[1] + 128) >> 8];
            mv1 = [mv0[0] - mvcol[0], mv0[1] - mvcol[1]];
        }
        self.setblk(ox, oy, 2, 2, 0, ref0 as i32, mv0, 1)?;
        self.setblk(ox, oy, 2, 2, 1, 0, mv1, 1)?;
        self.mark(ox, oy, 2, 2);
        Ok(())
    }

    /// MB-level reference indices and mv predictors of spatial direct
    /// (8.4.1.2.2): MinPositive over the A / B / C(or D) neighbours of the
    /// whole MB.
    fn spatial_pred(&self) -> [Nb; 2] {
        let (x0, y0) = (self.cmx * 4, self.cmy * 4);
        let mut out = [NB_NONE; 2];
        for lst in 0..2 {
            let a = self.nbr(x0 - 1, y0, lst).unwrap_or(NB_NONE);
            let b = self.nbr(x0, y0 - 1, lst).unwrap_or(NB_NONE);
            let mut c = self.nbr(x0 + 4, y0 - 1, lst);
            if c.is_none() {
                c = self.nbr(x0 - 1, y0 - 1, lst);
            }
            let c = c.unwrap_or(NB_NONE);
            let mut min_ref = i32::MAX;
            for n in [a, b, c] {
                if n.0 >= 0 && n.0 < min_ref {
                    min_ref = n.0;
                }
            }
            if min_ref == i32::MAX {
                out[lst] = NB_NONE;
                continue;
            }
            let n_match = (a.0 == min_ref) as u8 + (b.0 == min_ref) as u8 + (c.0 == min_ref) as u8;
            let mv = if n_match > 1 {
                [median(a.1[0], b.1[0], c.1[0]), median(a.1[1], b.1[1], c.1[1])]
            } else if a.0 == min_ref {
                a.1
            } else if b.0 == min_ref {
                b.1
            } else {
                c.1
            };
            out[lst] = (min_ref, mv);
        }
        if out[0].0 < 0 && out[1].0 < 0 {
            out = [(0, [0, 0]), (0, [0, 0])];
        }
        out
    }

    /// Spatial direct for one 8x8 quadrant (direct_8x8_inference): the
    /// MB-level prediction, with zero mvs where the co-located block is
    /// (nearly) still and uses reference index 0.
    fn spatial_direct(&mut self, i8: i32, sp: &[Nb; 2]) -> Result<()> {
        let cm = self.col_motion()?;
        let (ox, oy) = (self.cmx * 4 + (i8 % 2) * 2, self.cmy * 4 + (i8 / 2) * 2);
        let cx = self.cmx * 4 + if i8 % 2 == 1 { 3 } else { 0 };
        let cy = self.cmy * 4 + if i8 / 2 == 1 { 3 } else { 0 };
        let ci = (cy * self.w4 + cx) as usize;
        let mut col_zero = false;
        if cm.intra[ci] == 0 {
            let l = if cm.refidx[0][ci] >= 0 { 0 } else { 1 };
            let mvc = cm.mv[l][ci];
            col_zero = cm.refidx[l][ci] == 0 && (-1..=1).contains(&mvc[0]) && (-1..=1).contains(&mvc[1]);
        }
        for lst in 0..2 {
            let (rf, mv) = sp[lst];
            if rf < 0 {
                self.setblk(ox, oy, 2, 2, lst, -1, [0, 0], 1)?;
            } else {
                let mv = if col_zero && rf == 0 { [0, 0] } else { mv };
                self.setblk(ox, oy, 2, 2, lst, rf, mv, 1)?;
            }
        }
        self.mark(ox, oy, 2, 2);
        Ok(())
    }
}

fn rec_ref(mb: &Mb, lst: usize, p: usize) -> Result<i32> {
    match mb.refs.as_ref().and_then(|r| r[lst].get(p).copied().flatten()) {
        Some(v) => Ok(v as i32),
        None => bail!("TypeError: macroblock ({}, {}) has no reference index for list {lst} partition {p}", mb.x, mb.y),
    }
}

fn no_mvd(mb: &Mb) -> anyhow::Error {
    anyhow!("TypeError: macroblock ({}, {}) has no mvd to use", mb.x, mb.y)
}

/// The mvd of partition `p` (sub-partition `j` of quadrant `p` for 8x8 MBs).
fn rec_mvd(mb: &Mb, lst: usize, p: usize, j: Option<usize>) -> Result<[i32; 2]> {
    match (&mb.mvds, j) {
        (Some(Mvds::Parts(l)), None) => l[lst].get(p).copied().flatten().ok_or_else(|| no_mvd(mb)),
        (Some(Mvds::Sub(l)), Some(j)) => l[lst][p].get(j).copied().ok_or_else(|| no_mvd(mb)),
        _ => Err(no_mvd(mb)),
    }
}

fn set_mvd(mb: &mut Mb, lst: usize, p: usize, j: Option<usize>, v: [i32; 2]) -> Result<()> {
    let slot = match (&mut mb.mvds, j) {
        (Some(Mvds::Parts(l)), None) => l[lst].get_mut(p).map(|s| *s = Some(v)),
        (Some(Mvds::Sub(l)), Some(j)) => l[lst][p].get_mut(j).map(|s| *s = v),
        _ => None,
    };
    match slot {
        Some(()) => Ok(()),
        None => Err(no_mvd(mb)),
    }
}

/// Which prediction list a partition of an MB uses; P slices only have L0.
fn lst_unused(t: SliceType, pred: Pred, lst: usize) -> bool {
    (t == SliceType::P && lst == 1) || !pred.uses(lst)
}

fn derive<R: Recs>(
    pic: &Picture,
    recs: &mut R,
    mb_w: usize,
    mb_h: usize,
    pics: &[Picture],
    col: Option<&Motion>,
    target: Option<&Motion>,
) -> Result<Motion> {
    let t = pic.typ();
    if t == SliceType::I {
        return Ok(Motion::all_intra(mb_w, mb_h));
    }
    if t != SliceType::P && t != SliceType::B {
        bail!("slice type {} is not supported", t.as_str());
    }
    let spatial = t == SliceType::B && pic.slices[0].direct_spatial == Some(true);
    let mut d = Derive {
        m: Motion::new(mb_w, mb_h),
        w4: (mb_w * 4) as i32,
        h4: (mb_h * 4) as i32,
        pic,
        pics,
        col,
        cmx: 0,
        cmy: 0,
        done: 0,
    };
    let n = recs.recs().len();
    for i in 0..n {
        let (cmx, cmy, skip, intra, part, btype, sub) = {
            let mb = &recs.recs()[i];
            (mb.x as i32, mb.y as i32, mb.skip, mb.intra, mb.part, mb.btype, mb.sub)
        };
        d.cmx = cmx;
        d.cmy = cmy;
        d.done = 0;
        let (x0, y0) = (cmx * 4, cmy * 4);
        if intra {
            for y in y0..y0 + 4 {
                for x in x0..x0 + 4 {
                    d.m.intra[(y * d.w4 + x) as usize] = 1;
                }
            }
            for lst in 0..2 {
                d.setblk(x0, y0, 4, 4, lst, -1, [0, 0], 0)?;
            }
            continue;
        }
        if skip && t == SliceType::P {
            let a = d.nbr(x0 - 1, y0, 0);
            let b = d.nbr(x0, y0 - 1, 0);
            let zero = |n: Option<Nb>| matches!(n, Some((0, [0, 0])));
            let mv = if a.is_none() || b.is_none() || zero(a) || zero(b) { [0, 0] } else { d.mvp(x0, y0, 4, 0, 0, None, 0) };
            d.setblk(x0, y0, 4, 4, 0, 0, mv, 0)?;
            d.setblk(x0, y0, 4, 4, 1, -1, [0, 0], 0)?;
            continue;
        }
        let all_direct = skip || part == Some(Part::Direct);
        let any_direct = all_direct || (part == Some(Part::P8x8) && t == SliceType::B && sub.is_some_and(|s| s.contains(&0)));
        // MB-level spatial prediction, from neighbours outside the MB only (so
        // the same whenever it is computed); only direct blocks use it
        let sp = if spatial && any_direct { Some(d.spatial_pred()) } else { None };
        macro_rules! direct {
            ($i8:expr) => {
                match &sp {
                    Some(sp) => d.spatial_direct($i8, sp)?,
                    None => d.temporal_direct($i8)?,
                }
            };
        }
        if all_direct {
            for i8 in 0..4 {
                direct!(i8);
            }
            continue;
        }
        let Some(part) = part else { bail!("KeyError: 'part' (macroblock ({cmx}, {cmy}))") };
        if part != Part::P8x8 {
            let rects = mb_rects(part);
            let preds: &[Pred] = if t == SliceType::P {
                &[Pred::L0, Pred::L0][..rects.len()]
            } else {
                match btype {
                    Some(bt) => B_TYPES[bt as usize].1,
                    None => bail!("KeyError: 'btype' (macroblock ({cmx}, {cmy}))"),
                }
            };
            for (p, &(rx, ry, rw, rh)) in rects.iter().enumerate() {
                let (gx, gy) = (x0 + rx, y0 + ry);
                let Some(&pred) = preds.get(p) else { bail!("list index out of range") };
                for lst in 0..2 {
                    if lst_unused(t, pred, lst) {
                        d.setblk(gx, gy, rw, rh, lst, -1, [0, 0], 0)?;
                        continue;
                    }
                    let rf = rec_ref(&recs.recs()[i], lst, p)?;
                    let pm = d.mvp(gx, gy, rw, lst, rf, Some(part), p);
                    if let Some(tg) = target {
                        let tv = tg.mv[lst][tg.idx(gx as usize, gy as usize)];
                        let mb = &mut recs.recs_mut().ok_or_else(|| anyhow!("target mode needs writable records"))?[i];
                        set_mvd(mb, lst, p, None, [tv[0] as i32 - pm[0], tv[1] as i32 - pm[1]])?;
                    }
                    let dv = rec_mvd(&recs.recs()[i], lst, p, None)?;
                    d.setblk(gx, gy, rw, rh, lst, rf, [pm[0] + dv[0], pm[1] + dv[1]], 0)?;
                }
                d.mark(gx, gy, rw, rh);
            }
            continue;
        }
        // 8x8 partitions
        let Some(sub) = sub else { bail!("KeyError: 'sub' (macroblock ({cmx}, {cmy}))") };
        for qi in 0..4usize {
            let st = sub[qi] as usize;
            let (shape, pred) = if t == SliceType::P {
                match P_SUB.get(st) {
                    Some(&s) => (s, Pred::L0),
                    None => bail!("list index out of range"),
                }
            } else {
                match B_SUB.get(st) {
                    Some(&sp) => sp,
                    None => bail!("list index out of range"),
                }
            };
            let (ox, oy) = (x0 + (qi as i32 % 2) * 2, y0 + (qi as i32 / 2) * 2);
            if shape == SubShape::Direct {
                direct!(qi as i32);
                continue;
            }
            for (j, &(rx, ry, rw, rh)) in sub_rects(shape).iter().enumerate() {
                let (gx, gy) = (ox + rx, oy + ry);
                for lst in 0..2 {
                    if lst_unused(t, pred, lst) {
                        d.setblk(gx, gy, rw, rh, lst, -1, [0, 0], 0)?;
                        continue;
                    }
                    let rf = rec_ref(&recs.recs()[i], lst, qi)?;
                    let pm = d.mvp(gx, gy, rw, lst, rf, None, 0);
                    if let Some(tg) = target {
                        let tv = tg.mv[lst][tg.idx(gx as usize, gy as usize)];
                        let mb = &mut recs.recs_mut().ok_or_else(|| anyhow!("target mode needs writable records"))?[i];
                        set_mvd(mb, lst, qi, Some(j), [tv[0] as i32 - pm[0], tv[1] as i32 - pm[1]])?;
                    }
                    let dv = rec_mvd(&recs.recs()[i], lst, qi, Some(j))?;
                    d.setblk(gx, gy, rw, rh, lst, rf, [pm[0] + dv[0], pm[1] + dv[1]], 0)?;
                }
                d.mark(gx, gy, rw, rh);
            }
        }
    }
    Ok(d.m)
}

/// Reconstruct motion for a parsed picture (Python `derive_motion` without
/// `target`). `pics` is indexed by decode index and supplies POCs (and, for
/// temporal direct, the co-located picture's POC); `col` is the motion of
/// RefPicList1[0] (needed for B pictures with direct blocks). Errors where
/// Python raises (a reference index outside the list, a co-located reference
/// missing from list 0, direct blocks without `col`).
pub fn derive_motion(pic: &Picture, mbs: &[Mb], mb_w: usize, mb_h: usize, pics: &[Picture], col: Option<&Motion>) -> Result<Motion> {
    let mut r = mbs;
    derive(pic, &mut r, mb_w, mb_h, pics, col, None)
}

/// [`derive_motion`] in target mode: explicit partitions take their mvs from
/// `target`, and their mvds are solved and written into `mbs`, so the records
/// encode exactly that motion under the current neighbourhood.
pub fn derive_motion_target(
    pic: &Picture,
    mbs: &mut [Mb],
    mb_w: usize,
    mb_h: usize,
    pics: &[Picture],
    col: Option<&Motion>,
    target: &Motion,
) -> Result<Motion> {
    let mut r = mbs;
    derive(pic, &mut r, mb_w, mb_h, pics, col, Some(target))
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

    /// The rectangles `derive` assigns are exactly the partitions of `records::{mb_parts, sub_parts}`,
    /// in the same order, with the same first block (the one the mv predictor is taken at).
    #[test]
    fn partition_rectangles_match_the_tables() {
        use crate::records::{mb_parts, sub_parts};
        let cells = |&(x, y, w, h): &(i32, i32, i32, i32)| {
            let mut v: Vec<(usize, usize)> = (y..y + h).flat_map(|yy| (x..x + w).map(move |xx| (xx as usize, yy as usize))).collect();
            v.sort();
            v
        };
        for part in [Part::P16x16, Part::P16x8, Part::P8x16] {
            let (rects, parts) = (mb_rects(part), mb_parts(part));
            assert_eq!(rects.len(), parts.len());
            for (r, p) in rects.iter().zip(parts) {
                let mut sorted = p.to_vec();
                sorted.sort();
                assert_eq!(cells(r), sorted);
                assert_eq!((r.0 as usize, r.1 as usize), p[0]);
            }
        }
        for shape in [SubShape::S8x8, SubShape::S8x4, SubShape::S4x8, SubShape::S4x4] {
            let (rects, parts) = (sub_rects(shape), sub_parts(shape));
            assert_eq!(rects.len(), parts.len());
            for (r, p) in rects.iter().zip(parts) {
                let mut sorted = p.to_vec();
                sorted.sort();
                assert_eq!(cells(r), sorted);
                assert_eq!((r.0 as usize, r.1 as usize), p[0]);
            }
        }
    }

    #[test]
    fn small_helpers() {
        assert_eq!(median(3, 1, 2), 2);
        assert_eq!(median(-5, 7, 0), 0);
        assert_eq!(median(4, 4, -9), 4);
        assert_eq!(rect_mask(0, 0, 4, 4), 0xffff);
        assert_eq!(rect_mask(2, 1, 2, 2), 0b0000_1100_1100_0000);
        // C division truncates toward zero (the temporal direct scale factor)
        assert_eq!((16384 + 3 / 2) / -3, -5461);
    }

    /// Hand-computed P_Skip / mv prediction on a 3x2 MB picture (one reference, no residual).
    /// Row 0: coded (mvd 8,-4), coded (mvd 4,4), skip. Row 1: coded (mvd 6,6), skip, skip.
    #[test]
    fn p_skip_and_mv_prediction() {
        let mk = |x: u16, y: u16, mvd: Option<[i32; 2]>| {
            let mut mb = Mb::new(x, y);
            match mvd {
                None => mb.skip = true,
                Some(d) => {
                    mb.kind = crate::records::Kind::P;
                    mb.part = Some(Part::P16x16);
                    mb.refs = Some([vec![Some(0)], vec![None]]);
                    mb.mvds = Some(Mvds::Parts([vec![Some(d)], vec![None]]));
                }
            }
            mb
        };
        let mbs = vec![
            mk(0, 0, Some([8, -4])),
            mk(1, 0, Some([4, 4])),
            mk(2, 0, None),
            mk(0, 1, Some([6, 6])),
            mk(1, 1, None),
            mk(2, 1, None),
        ];
        let mut pic = Picture::default();
        pic.slices = vec![crate::patch::test_slice_header()];
        pic.lists = [vec![0], vec![]];
        let m = derive_motion(&pic, &mbs, 3, 2, &[], None).unwrap();
        let mv = |x: usize, y: usize| m.mv[0][m.idx(x * 4, y * 4)];
        // MB 0: no neighbours, predictor (0, 0). MB 1: only A: predictor (8, -4).
        assert_eq!(mv(0, 0), [8, -4]);
        assert_eq!(mv(1, 0), [12, 0]);
        // top-row skips: B unavailable, so zero motion
        assert_eq!(mv(2, 0), [0, 0]);
        // MB (0,1): A unavailable; B (8,-4) and C (12,0) both use ref 0, median with (0,0) is (8,0)
        assert_eq!(mv(0, 1), [14, 6]);
        // MB (1,1) skip: A (14,6), B (12,0), C (0,0): median (12,0)
        assert_eq!(mv(1, 1), [12, 0]);
        // MB (2,1) skip: B is a zero-mv reference-0 block: zero motion
        assert_eq!(mv(2, 1), [0, 0]);
        assert_eq!(m.refpic[0][m.idx(8, 4)], 0);
        assert_eq!(m.refidx[1][m.idx(8, 4)], -1);
        assert!(m.direct.iter().all(|&d| d == 0));
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
        scaling_matrix: false,
        bit_depth: [8, 8],
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
        pwt: None,
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

    /// An IDR K, its I twin D and a P reading [D, K]: without D the P's list
    /// is [K] while its header still makes two references active. With an
    /// older reference in the DPB the list only shifts ([K, older]).
    #[test]
    fn short_lists_after_a_hard_drop() {
        use crate::stream::{hard_drop_checker, short_lists};
        use std::collections::HashSet;
        let spss = sps(4);
        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::I, 1, 2, 0, vec![]),
            pic(false, 2, SliceType::P, 2, 8, 2, vec![]),
        ];
        dpb_walk(&spss, &mut pics).unwrap();
        assert_eq!(pics[2].lists[0], vec![1, 0]);
        assert_eq!(short_lists(&spss, &pics, &HashSet::from([1])).unwrap(), vec![2]);
        assert!(!hard_drop_checker(&spss, &pics)(1).unwrap());

        let mut pics = vec![
            pic(true, 3, SliceType::I, 0, 0, 0, vec![]),
            pic(false, 2, SliceType::P, 1, 2, 1, vec![]),
            pic(false, 2, SliceType::P, 2, 4, 2, vec![]),
            pic(false, 2, SliceType::P, 3, 6, 2, vec![]),
            pic(false, 2, SliceType::P, 4, 8, 2, vec![]),
        ];
        dpb_walk(&spss, &mut pics).unwrap();
        assert_eq!(pics[4].lists[0], vec![3, 2]);
        assert!(short_lists(&spss, &pics, &HashSet::from([3])).unwrap().is_empty());
        assert!(hard_drop_checker(&spss, &pics)(3).unwrap());
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

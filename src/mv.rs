//! Reference structure and motion reconstruction. Port of `frame_dedup/h264mv.py`.
//!
//! * `dpb_walk` simulates the decoded picture buffer from slice headers only.
//! * `derive_motion` turns macroblock records into a per-4x4 motion field.

use anyhow::Result;

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

/// Apply ref_pic_list_modification (short-term commands) to an initial
/// list. Errors where Python raises ValueError.
pub fn modify_list(
    _lst: &[usize],
    _cmds: &[(u32, u32)],
    _n_active: usize,
    _cur_fn: u32,
    _refs: &[RefPic],
    _max_fn: u32,
) -> Result<Vec<usize>> {
    todo!("port h264mv.modify_list")
}

/// Initial reference lists (8.2.4.2), truncated to the active sizes.
pub fn init_lists(
    _t: SliceType,
    _refs: &[RefPic],
    _cur_fn: u32,
    _cur_poc: i32,
    _n0: usize,
    _n1: usize,
    _max_fn: u32,
) -> [Vec<usize>; 2] {
    todo!("port h264mv.init_lists")
}

/// Header-only DPB simulation: sets `dec`, `poc`, `frame_num`, `lists` and
/// `dpb_after` of every picture. Errors where Python raises ValueError.
pub fn dpb_walk(_spss: &Spss, _pics: &mut [Picture]) -> Result<()> {
    todo!("port h264mv.dpb_walk")
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

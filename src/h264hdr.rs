//! Minimal H.264 Annex-B header parser (SPS / PPS / slice headers).
//! Port of `frame_dedup/h264hdr.py`.
//!
//! The types below are the shared interface used by every other module.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};

use crate::bits::{BitReader, escape, set_bits, unescape};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SliceType {
    P,
    B,
    I,
    SP,
    SI,
}

impl SliceType {
    /// slice_type % 5 (Table 7-6).
    pub fn from_code(v: u32) -> SliceType {
        match v % 5 {
            0 => SliceType::P,
            1 => SliceType::B,
            2 => SliceType::I,
            3 => SliceType::SP,
            _ => SliceType::SI,
        }
    }

    /// The Python string form ('P', 'B', 'I', 'SP', 'SI'), used in dumps and reports.
    pub fn as_str(self) -> &'static str {
        match self {
            SliceType::P => "P",
            SliceType::B => "B",
            SliceType::I => "I",
            SliceType::SP => "SP",
            SliceType::SI => "SI",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Timing {
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub fixed_frame_rate: bool,
    /// Bit offset of num_units_in_tick in the SPS RBSP (after the NAL header byte).
    pub bit: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hrd {
    pub cpb_cnt: u32,
    pub bit_rate_scale: u32,
    pub cpb_size_scale: u32,
    pub sched: Vec<(u32, u32, bool)>,
    pub init_cpb_delay_len: u32,
    pub cpb_removal_delay_len: u32,
    pub dpb_output_delay_len: u32,
    pub time_offset_len: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitstreamRestriction {
    pub mv_over_pic_boundaries: bool,
    pub max_bytes_per_pic_denom: u32,
    pub max_bits_per_mb_denom: u32,
    pub log2_max_mv_length_h: u32,
    pub log2_max_mv_length_v: u32,
    pub max_num_reorder_frames: u32,
    pub max_dec_frame_buffering: u32,
}

/// vui_parameters() (Annex E.1.1).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Vui {
    pub aspect_ratio_idc: Option<u32>,
    pub sar: Option<(u32, u32)>,
    pub overscan_appropriate: Option<bool>,
    pub video_format: Option<u32>,
    pub full_range: Option<bool>,
    pub colour: Option<(u32, u32, u32)>,
    pub chroma_loc: Option<(u32, u32)>,
    pub timing: Option<Timing>,
    pub nal_hrd: Option<Hrd>,
    pub vcl_hrd: Option<Hrd>,
    pub low_delay_hrd: Option<bool>,
    pub pic_struct_present: bool,
    pub bitstream_restriction: Option<BitstreamRestriction>,
    /// Bit offset just past the VUI in the SPS RBSP.
    pub end_bit: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sps {
    pub profile_idc: u32,
    pub level_idc: u32,
    pub id: u32,
    pub chroma_format_idc: u32,
    pub separate_colour_plane: bool,
    pub log2_max_frame_num: u32,
    pub poc_type: u32,
    pub log2_max_poc_lsb: u32,
    pub delta_pic_order_always_zero: bool,
    pub max_num_ref_frames: u32,
    pub gaps_allowed: bool,
    pub mb_w: usize,
    pub mb_h: usize,
    pub frame_mbs_only: bool,
    pub mbaff: bool,
    pub direct_8x8_inference: bool,
    /// left, right, top, bottom (in the SPS's units; the Python code doubles them for 4:2:0).
    pub crop: [u32; 4],
    pub vui: Option<Vui>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pps {
    pub id: u32,
    pub sps_id: u32,
    pub cabac: bool,
    pub bottom_field_poc_present: bool,
    pub num_slice_groups: u32,
    pub num_ref_idx_l0_default: u32,
    pub num_ref_idx_l1_default: u32,
    pub weighted_pred: bool,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp: i32,
    pub chroma_qp_offset: i32,
    pub deblocking_ctrl_present: bool,
    pub constrained_intra: bool,
    pub redundant_pic_cnt_present: bool,
    pub transform_8x8: bool,
    pub scaling_matrix: bool,
    pub chroma_qp_offset2: i32,
}

/// One dec_ref_pic_marking operation: memory_management_control_operation
/// and its arguments, in bitstream order (Python: a tuple `(op, *args)`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mmco {
    pub op: u32,
    pub args: Vec<u32>,
}

/// A parsed slice header. `*_bit` fields are bit offsets in the slice RBSP
/// (after the NAL header byte). Options are `None` where the Python dict
/// lacks the key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceHeader {
    pub first_mb: u32,
    pub slice_type: SliceType,
    pub pps_id: u32,
    pub frame_num_bit: usize,
    pub frame_num: u32,
    pub field_pic: bool,
    pub bottom_field: Option<bool>,
    pub idr_pic_id_bit: Option<usize>,
    pub idr_pic_id: Option<u32>,
    pub poc_lsb_bit: usize,
    pub poc_lsb: u32,
    pub delta_poc_bottom: Option<i32>,
    /// B slices only.
    pub direct_spatial: Option<bool>,
    pub num_ref_l0: u32,
    pub num_ref_l1: u32,
    /// ref_pic_list_modification commands per list: (modification_of_pic_nums_idc, value).
    pub rplm: [Vec<(u32, u32)>; 2],
    pub rplm_bit: usize,
    pub rplm_end_bit: usize,
    pub weighted: bool,
    pub mmco: Vec<Mmco>,
    /// adaptive_ref_pic_marking_mode_flag (non-IDR reference slices).
    pub adaptive: bool,
    pub no_output_prior: Option<bool>,
    pub long_term_ref: Option<bool>,
    pub drpm_bit: usize,
    pub drpm_end_bit: usize,
    pub cabac_init_idc: u32,
    pub qp: i32,
    pub deblock_idc: u32,
    pub alpha_off: i32,
    pub beta_off: i32,
    /// Bits of the header proper (before cabac_alignment_one_bits).
    pub hdr_bits: usize,
    /// Bit offset where slice_data() starts (byte-aligned for CABAC).
    pub data_bit: usize,
    // attached by parse_stream
    pub nal_type: u8,
    pub ref_idc: u8,
    /// Byte offset of the NAL header byte in the Annex-B file.
    pub offset: usize,
    /// NAL size in bytes (header byte included, trailing zero bytes excluded).
    pub size: usize,
}

/// A picture of an Annex-B stream, in decode order. `parse_stream` fills
/// `slices`, `idr`, `ref_idc`; `mv::dpb_walk` fills `dec`, `poc`,
/// `frame_num`, `lists`, `dpb_after`; `stream::load` fills `period`, `disp`,
/// `au_start`, `au_end`.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Picture {
    pub slices: Vec<SliceHeader>,
    pub idr: bool,
    pub ref_idc: u8,
    pub dec: usize,
    pub poc: i32,
    pub frame_num: u32,
    /// Final reference lists as decode indices (of the same stream).
    pub lists: [Vec<usize>; 2],
    /// Decode indices of the short-term references after this picture is marked.
    pub dpb_after: Vec<usize>,
    pub period: usize,
    pub disp: usize,
    pub au_start: usize,
    pub au_end: usize,
}

impl Picture {
    pub fn sl(&self) -> &SliceHeader {
        &self.slices[0]
    }

    /// Python's `p['type']`: the slice type of the first slice.
    pub fn typ(&self) -> SliceType {
        self.slices[0].slice_type
    }
}

pub type Spss = BTreeMap<u32, Sps>;
pub type Ppss = BTreeMap<u32, Pps>;

/// profile_idc values with the High-profile SPS extension (chroma format,
/// bit depths, scaling matrices).
const HIGH_PROFILES: [u32; 13] = [100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135];

fn skip_scaling_list(b: &mut BitReader, size: usize) -> Result<()> {
    let (mut last, mut nxt) = (8i64, 8i64);
    for _ in 0..size {
        if nxt != 0 {
            nxt = (last + b.se()? as i64 + 256).rem_euclid(256);
        }
        if nxt != 0 {
            last = nxt;
        }
    }
    Ok(())
}

pub fn parse_sps(rbsp: &[u8]) -> Result<Sps> {
    let mut b = BitReader::new(rbsp);
    let profile_idc = b.u(8)?;
    b.u(8)?;
    let level_idc = b.u(8)?;
    let id = b.ue()?;
    let mut chroma_format_idc = 1;
    let mut separate_colour_plane = false;
    if HIGH_PROFILES.contains(&profile_idc) {
        chroma_format_idc = b.ue()?;
        if chroma_format_idc == 3 {
            separate_colour_plane = b.flag()?;
        }
        b.ue()?;
        b.ue()?;
        b.u(1)?;
        if b.flag()? {
            // seq_scaling_matrix_present
            for i in 0..(if chroma_format_idc != 3 { 8 } else { 12 }) {
                if b.flag()? {
                    skip_scaling_list(&mut b, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    let log2_max_frame_num = b.ue()?.saturating_add(4);
    let poc_type = b.ue()?;
    let mut log2_max_poc_lsb = 0;
    let mut delta_pic_order_always_zero = false;
    if poc_type == 0 {
        log2_max_poc_lsb = b.ue()?.saturating_add(4);
    } else if poc_type == 1 {
        delta_pic_order_always_zero = b.flag()?;
        b.se()?;
        b.se()?;
        for _ in 0..b.ue()? {
            b.se()?;
        }
    }
    let max_num_ref_frames = b.ue()?;
    let gaps_allowed = b.flag()?;
    let mb_w = b.ue()? as usize + 1;
    let mb_h = b.ue()? as usize + 1;
    let frame_mbs_only = b.flag()?;
    let mut mbaff = false;
    if !frame_mbs_only {
        mbaff = b.flag()?;
    }
    let direct_8x8_inference = b.flag()?;
    let mut crop = [0; 4];
    if b.flag()? {
        // frame_cropping_flag: left, right, top, bottom
        crop = [b.ue()?, b.ue()?, b.ue()?, b.ue()?];
    }
    let vui = if b.flag()? { Some(parse_vui(&mut b)?) } else { None };
    Ok(Sps {
        profile_idc,
        level_idc,
        id,
        chroma_format_idc,
        separate_colour_plane,
        log2_max_frame_num,
        poc_type,
        log2_max_poc_lsb,
        delta_pic_order_always_zero,
        max_num_ref_frames,
        gaps_allowed,
        mb_w,
        mb_h,
        frame_mbs_only,
        mbaff,
        direct_8x8_inference,
        crop,
        vui,
    })
}

fn parse_hrd(b: &mut BitReader) -> Result<Hrd> {
    let cpb_cnt = b.ue1()?;
    let bit_rate_scale = b.u(4)?;
    let cpb_size_scale = b.u(4)?;
    let mut sched = Vec::new();
    for _ in 0..cpb_cnt {
        sched.push((b.ue()?, b.ue()?, b.flag()?));
    }
    Ok(Hrd {
        cpb_cnt,
        bit_rate_scale,
        cpb_size_scale,
        sched,
        init_cpb_delay_len: b.u(5)? + 1,
        cpb_removal_delay_len: b.u(5)? + 1,
        dpb_output_delay_len: b.u(5)? + 1,
        time_offset_len: b.u(5)?,
    })
}

/// vui_parameters() (Annex E.1.1). Records the bit offsets of the timing
/// fields so they can be patched in place.
fn parse_vui(b: &mut BitReader) -> Result<Vui> {
    let mut v = Vui::default();
    if b.flag()? {
        // aspect_ratio_info_present
        let idc = b.u(8)?;
        v.aspect_ratio_idc = Some(idc);
        if idc == 255 {
            v.sar = Some((b.u(16)?, b.u(16)?));
        }
    }
    if b.flag()? {
        v.overscan_appropriate = Some(b.flag()?);
    }
    if b.flag()? {
        // video_signal_type_present
        v.video_format = Some(b.u(3)?);
        v.full_range = Some(b.flag()?);
        if b.flag()? {
            v.colour = Some((b.u(8)?, b.u(8)?, b.u(8)?));
        }
    }
    if b.flag()? {
        // chroma_loc_info_present
        v.chroma_loc = Some((b.ue()?, b.ue()?));
    }
    if b.flag()? {
        // timing_info_present
        let bit = b.p;
        v.timing = Some(Timing {
            num_units_in_tick: b.u(32)?,
            time_scale: b.u(32)?,
            fixed_frame_rate: b.flag()?,
            bit,
        });
    }
    if b.flag()? {
        v.nal_hrd = Some(parse_hrd(b)?);
    }
    if b.flag()? {
        v.vcl_hrd = Some(parse_hrd(b)?);
    }
    if v.nal_hrd.is_some() || v.vcl_hrd.is_some() {
        v.low_delay_hrd = Some(b.flag()?);
    }
    v.pic_struct_present = b.flag()?;
    if b.flag()? {
        v.bitstream_restriction = Some(BitstreamRestriction {
            mv_over_pic_boundaries: b.flag()?,
            max_bytes_per_pic_denom: b.ue()?,
            max_bits_per_mb_denom: b.ue()?,
            log2_max_mv_length_h: b.ue()?,
            log2_max_mv_length_v: b.ue()?,
            max_num_reorder_frames: b.ue()?,
            max_dec_frame_buffering: b.ue()?,
        });
    }
    v.end_bit = b.p;
    Ok(v)
}

/// (num, den) frame rate from the SPS VUI, or None. For progressive frames
/// one frame lasts two ticks: fps = time_scale / (2 * num_units_in_tick).
pub fn sps_frame_rate(sps: &Sps) -> Option<(u64, u64)> {
    let t = sps.vui.as_ref()?.timing.as_ref()?;
    if t.num_units_in_tick == 0 || t.time_scale == 0 {
        return None;
    }
    Some((t.time_scale as u64, 2 * t.num_units_in_tick as u64))
}

/// The SPS NAL (header byte + escaped payload) with its VUI timing rewritten
/// in place. The fields are fixed-width u(32), so the RBSP keeps its length;
/// only emulation prevention can change.
pub fn patch_sps_timing(nal: &[u8], num_units_in_tick: u32, time_scale: u32) -> Result<Vec<u8>> {
    if nal.is_empty() {
        bail!("empty SPS NAL");
    }
    let mut rbsp = unescape(&nal[1..]);
    let sps = parse_sps(&rbsp)?;
    let Some(t) = sps.vui.as_ref().and_then(|v| v.timing.as_ref()) else {
        bail!("SPS has no VUI timing_info; in-place timing patch impossible");
    };
    set_bits(&mut rbsp, t.bit, 32, num_units_in_tick as u64);
    set_bits(&mut rbsp, t.bit + 32, 32, time_scale as u64);
    let chk = parse_sps(&rbsp)?;
    let ct = chk.vui.as_ref().and_then(|v| v.timing.as_ref());
    if ct.map(|t| t.num_units_in_tick) != Some(num_units_in_tick) {
        bail!("AssertionError: patched num_units_in_tick does not read back");
    }
    if ct.map(|t| t.time_scale) != Some(time_scale) {
        bail!("AssertionError: patched time_scale does not read back");
    }
    if chk.vui.as_ref().map(|v| v.end_bit) != sps.vui.as_ref().map(|v| v.end_bit) {
        bail!("AssertionError: patched SPS VUI changed length");
    }
    let mut out = vec![nal[0]];
    out.extend_from_slice(&escape(&rbsp));
    Ok(out)
}

pub fn parse_pps(rbsp: &[u8]) -> Result<Pps> {
    let mut b = BitReader::new(rbsp);
    let id = b.ue()?;
    let sps_id = b.ue()?;
    let cabac = b.flag()?;
    let bottom_field_poc_present = b.flag()?;
    let num_slice_groups = b.ue1()?;
    if num_slice_groups != 1 {
        bail!("AssertionError: FMO not supported");
    }
    let num_ref_idx_l0_default = b.ue1()?;
    let num_ref_idx_l1_default = b.ue1()?;
    let weighted_pred = b.flag()?;
    let weighted_bipred_idc = b.u(2)?;
    let pic_init_qp = 26 + b.se()?;
    b.se()?;
    let chroma_qp_offset = b.se()?;
    let deblocking_ctrl_present = b.flag()?;
    let constrained_intra = b.flag()?;
    let redundant_pic_cnt_present = b.flag()?;
    let mut transform_8x8 = false;
    let mut scaling_matrix = false;
    let mut chroma_qp_offset2 = chroma_qp_offset;
    // position of the rbsp_stop_one_bit, for more_rbsp_data()
    let mut last1 = rbsp.len() as i64 * 8 - 1;
    loop {
        if last1 < 0 {
            bail!("IndexError: PPS has no stop bit");
        }
        if (rbsp[(last1 >> 3) as usize] >> (7 - (last1 & 7))) & 1 != 0 {
            break;
        }
        last1 -= 1;
    }
    if (b.p as i64) < last1 {
        transform_8x8 = b.flag()?;
        scaling_matrix = b.flag()?;
        if scaling_matrix {
            bail!("AssertionError: PPS scaling matrices not supported");
        }
        chroma_qp_offset2 = b.se()?;
    }
    Ok(Pps {
        id,
        sps_id,
        cabac,
        bottom_field_poc_present,
        num_slice_groups,
        num_ref_idx_l0_default,
        num_ref_idx_l1_default,
        weighted_pred,
        weighted_bipred_idc,
        pic_init_qp,
        chroma_qp_offset,
        deblocking_ctrl_present,
        constrained_intra,
        redundant_pic_cnt_present,
        transform_8x8,
        scaling_matrix,
        chroma_qp_offset2,
    })
}

/// Parse a slice header from the slice's RBSP (NAL header byte removed).
/// For streams with `poc_type != 0` the POC fields are read and discarded
/// (the stream is rejected later by `stream::check_scope`), and `poc_lsb`,
/// `poc_lsb_bit` are 0.
pub fn parse_slice(rbsp: &[u8], nal_type: u8, nal_ref_idc: u8, spss: &Spss, ppss: &Ppss) -> Result<SliceHeader> {
    let mut b = BitReader::new(rbsp);
    let first_mb = b.ue()?;
    let st = b.ue()?;
    let t = SliceType::from_code(st);
    let pps_id = b.ue()?;
    let pps = ppss.get(&pps_id).ok_or_else(|| anyhow!("KeyError: PPS {pps_id}"))?;
    let sps = spss.get(&pps.sps_id).ok_or_else(|| anyhow!("KeyError: SPS {}", pps.sps_id))?;
    if sps.separate_colour_plane {
        b.u(2)?;
    }
    let frame_num_bit = b.p; // bit offset in the RBSP (after the NAL header byte)
    let frame_num = b.u(sps.log2_max_frame_num)?;
    let mut field_pic = false;
    let mut bottom_field = None;
    if !sps.frame_mbs_only {
        field_pic = b.flag()?;
        if field_pic {
            bottom_field = Some(b.flag()?);
        }
    }
    let (mut idr_pic_id_bit, mut idr_pic_id) = (None, None);
    if nal_type == 5 {
        idr_pic_id_bit = Some(b.p);
        idr_pic_id = Some(b.ue()?);
    }
    let (mut poc_lsb_bit, mut poc_lsb) = (0, 0);
    let mut delta_poc_bottom = None;
    if sps.poc_type == 0 {
        poc_lsb_bit = b.p;
        poc_lsb = b.u(sps.log2_max_poc_lsb)?;
        if pps.bottom_field_poc_present && !field_pic {
            delta_poc_bottom = Some(b.se()?);
        }
    } else if sps.poc_type == 1 && !sps.delta_pic_order_always_zero {
        b.se()?; // delta_poc0
        if pps.bottom_field_poc_present && !field_pic {
            b.se()?; // delta_poc1
        }
    }
    if pps.redundant_pic_cnt_present {
        b.ue()?;
    }
    let mut direct_spatial = None;
    if t == SliceType::B {
        direct_spatial = Some(b.flag()?);
    }
    let mut num_ref_l0 = pps.num_ref_idx_l0_default;
    let mut num_ref_l1 = pps.num_ref_idx_l1_default;
    if matches!(t, SliceType::P | SliceType::SP | SliceType::B) && b.flag()? {
        num_ref_l0 = b.ue1()?;
        if t == SliceType::B {
            num_ref_l1 = b.ue1()?;
        }
    }
    if t != SliceType::B {
        num_ref_l1 = 0;
    }
    if matches!(t, SliceType::I | SliceType::SI) {
        num_ref_l0 = 0;
    }
    // ref_pic_list_modification, spanning [rplm_bit, rplm_end_bit)
    let mut rplm: [Vec<(u32, u32)>; 2] = [Vec::new(), Vec::new()];
    let rplm_bit = b.p;
    for lst in 0..2 {
        if ((lst == 0 && !matches!(t, SliceType::I | SliceType::SI)) || (lst == 1 && t == SliceType::B)) && b.flag()? {
            loop {
                let idc = b.ue()?;
                if idc == 3 {
                    break;
                }
                rplm[lst].push((idc, b.ue()?));
            }
        }
    }
    let rplm_end_bit = b.p;
    // pred_weight_table
    let mut weighted = false;
    if (pps.weighted_pred && matches!(t, SliceType::P | SliceType::SP)) || (pps.weighted_bipred_idc == 1 && t == SliceType::B) {
        weighted = true;
        b.ue()?;
        if sps.chroma_format_idc != 0 {
            b.ue()?;
        }
        for n in [num_ref_l0, num_ref_l1] {
            for _ in 0..n {
                if b.flag()? {
                    b.se()?;
                    b.se()?;
                }
                if sps.chroma_format_idc != 0 && b.flag()? {
                    b.se()?;
                    b.se()?;
                    b.se()?;
                    b.se()?;
                }
            }
        }
    }
    // dec_ref_pic_marking, spanning [drpm_bit, drpm_end_bit)
    let mut mmco = Vec::new();
    let mut adaptive = false;
    let (mut no_output_prior, mut long_term_ref) = (None, None);
    let drpm_bit = b.p;
    if nal_ref_idc != 0 {
        if nal_type == 5 {
            no_output_prior = Some(b.flag()?);
            long_term_ref = Some(b.flag()?);
        } else if b.flag()? {
            adaptive = true;
            loop {
                let op = b.ue()?;
                if op == 0 {
                    break;
                }
                let mut args = Vec::new();
                if op == 1 || op == 3 {
                    args.push(b.ue()?);
                }
                if op == 2 {
                    args.push(b.ue()?);
                }
                if op == 3 || op == 6 {
                    args.push(b.ue()?);
                }
                if op == 4 {
                    args.push(b.ue()?);
                }
                mmco.push(Mmco { op, args });
            }
        }
    }
    let drpm_end_bit = b.p;
    let mut cabac_init_idc = 0;
    if pps.cabac && !matches!(t, SliceType::I | SliceType::SI) {
        cabac_init_idc = b.ue()?;
    }
    let qp = pps.pic_init_qp + b.se()?;
    let (mut deblock_idc, mut alpha_off, mut beta_off) = (0, 0, 0);
    if pps.deblocking_ctrl_present {
        deblock_idc = b.ue()?;
        if deblock_idc != 1 {
            alpha_off = 2 * b.se()?;
            beta_off = 2 * b.se()?;
        }
    }
    let hdr_bits = b.p;
    let data_bit = if pps.cabac { (b.p + 7) & !7 } else { b.p }; // cabac_alignment_one_bits
    Ok(SliceHeader {
        first_mb,
        slice_type: t,
        pps_id,
        frame_num_bit,
        frame_num,
        field_pic,
        bottom_field,
        idr_pic_id_bit,
        idr_pic_id,
        poc_lsb_bit,
        poc_lsb,
        delta_poc_bottom,
        direct_spatial,
        num_ref_l0,
        num_ref_l1,
        rplm,
        rplm_bit,
        rplm_end_bit,
        weighted,
        mmco,
        adaptive,
        no_output_prior,
        long_term_ref,
        drpm_bit,
        drpm_end_bit,
        cabac_init_idc,
        qp,
        deblock_idc,
        alpha_off,
        beta_off,
        hdr_bits,
        data_bit,
        nal_type,
        ref_idc: nal_ref_idc,
        offset: 0,
        size: 0,
    })
}

/// (payload_start, payload_end) of every NAL in an Annex-B buffer, exactly as
/// Python's `iter_nals` (start = just past 00 00 01; end excludes trailing zero bytes).
pub fn iter_nals(d: &[u8]) -> Vec<(usize, usize)> {
    // starts = [m.end() for m in re.finditer(rb'\x00\x00\x01', d)]
    let n = d.len();
    let mut starts: Vec<usize> = Vec::new();
    let mut i = 2;
    while i < n {
        // A start code covering position i needs d[i] <= 1, so a larger byte
        // lets the scan skip ahead; a match at i cannot overlap another one.
        let c = d[i];
        if c > 1 {
            i += 3;
        } else if c == 1 {
            if d[i - 1] == 0 && d[i - 2] == 0 {
                starts.push(i + 1);
            }
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = if k + 1 < starts.len() { starts[k + 1] - 3 } else { n };
        // trailing zero of a 4-byte start code belongs to the next NAL
        while e > s && d[e - 1] == 0 {
            e -= 1;
        }
        out.push((s, e));
    }
    out
}

/// Slice headers are parsed from at most this many bytes after the NAL header.
pub const HEAD_BYTES: usize = 512;

/// Pictures in decode order with their slice headers (Python: parse_stream).
pub fn parse_stream(data: &[u8]) -> Result<(Spss, Ppss, Vec<Picture>)> {
    let (mut spss, mut ppss, mut pics): (Spss, Ppss, Vec<Picture>) = (Spss::new(), Ppss::new(), Vec::new());
    for (s, e) in iter_nals(data) {
        if s >= data.len() {
            bail!("IndexError: NAL header past the end of the stream");
        }
        let hdr = data[s];
        let nal_type = hdr & 0x1f;
        let ref_idc = (hdr >> 5) & 3;
        if nal_type == 7 {
            let sps = parse_sps(&unescape(&data[s + 1..e.max(s + 1)]))?;
            spss.insert(sps.id, sps);
        } else if nal_type == 8 {
            let pps = parse_pps(&unescape(&data[s + 1..e.max(s + 1)]))?;
            ppss.insert(pps.id, pps);
        } else if nal_type == 1 || nal_type == 5 {
            let end = e.min(s + 1 + HEAD_BYTES).max(s + 1);
            let mut h = parse_slice(&unescape(&data[s + 1..end]), nal_type, ref_idc, &spss, &ppss)?;
            h.offset = s;
            h.size = e - s;
            if h.first_mb == 0 {
                pics.push(Picture { slices: vec![h], idr: nal_type == 5, ref_idc, ..Default::default() });
            } else {
                match pics.last_mut() {
                    Some(p) => p.slices.push(h),
                    None => bail!("IndexError: slice with first_mb != 0 before any picture"),
                }
            }
        }
    }
    Ok((spss, ppss, pics))
}

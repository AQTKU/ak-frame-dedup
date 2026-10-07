//! Minimal H.264 Annex-B header parser (SPS / PPS / slice headers).
//! Port of `frame_dedup/h264hdr.py`.
//!
//! The types below are the shared interface used by every other module. The
//! parsing functions are to be filled in by the port (see port/PORTING.md).

use std::collections::BTreeMap;

use anyhow::Result;

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

pub fn parse_sps(_rbsp: &[u8]) -> Result<Sps> {
    todo!("port h264hdr.parse_sps")
}

pub fn parse_pps(_rbsp: &[u8]) -> Result<Pps> {
    todo!("port h264hdr.parse_pps")
}

pub fn parse_slice(_rbsp: &[u8], _nal_type: u8, _nal_ref_idc: u8, _spss: &Spss, _ppss: &Ppss) -> Result<SliceHeader> {
    todo!("port h264hdr.parse_slice")
}

/// (num, den) frame rate from the SPS VUI, or None.
pub fn sps_frame_rate(_sps: &Sps) -> Option<(u64, u64)> {
    todo!("port h264hdr.sps_frame_rate")
}

/// The SPS NAL (header byte + escaped payload) with its VUI timing rewritten in place.
pub fn patch_sps_timing(_nal: &[u8], _num_units_in_tick: u32, _time_scale: u32) -> Result<Vec<u8>> {
    todo!("port h264hdr.patch_sps_timing")
}

/// (payload_start, payload_end) of every NAL in an Annex-B buffer, exactly as
/// Python's `iter_nals` (start = just past 00 00 01; end excludes trailing zero bytes).
pub fn iter_nals(_data: &[u8]) -> Vec<(usize, usize)> {
    todo!("port h264hdr.iter_nals")
}

/// Pictures in decode order with their slice headers (Python: parse_stream).
pub fn parse_stream(_data: &[u8]) -> Result<(Spss, Ppss, Vec<Picture>)> {
    todo!("port h264hdr.parse_stream")
}

//! Picture-level decode and encode of slice data. Port of
//! `roundtrip_test.decode_picture` / `encode_picture`.

use anyhow::Result;

use crate::h264hdr::{Picture, Ppss, SliceHeader, Spss};
use crate::records::Mb;

/// A parsed picture (Python: the dict returned by `decode_picture`).
#[derive(Clone, Debug)]
pub struct Parsed {
    /// The NAL header byte.
    pub nal_header: u8,
    /// Size of the original NAL in bytes (Python `len(dd['nal'])`).
    pub nal_len: usize,
    /// The slice header the records are encoded under.
    pub sl: SliceHeader,
    /// RBSP bytes before slice_data(): `rbsp[..sl.data_bit / 8]` (header plus
    /// cabac_alignment_one_bits). Python keeps these as `bits[:data_bit]`.
    pub head: Vec<u8>,
    pub mbs: Vec<Mb>,
    /// Bit position just past the last CABAC bit read (Python `end`).
    pub end_bit: usize,
}

/// Parse the slice data of `pic` from the Annex-B buffer `es`.
pub fn decode_picture(_es: &[u8], _spss: &Spss, _ppss: &Ppss, _pic: &Picture) -> Result<Parsed> {
    todo!("port roundtrip_test.decode_picture")
}

/// Encode `mbs` under `p`'s header. Returns the NAL (header byte + escaped
/// RBSP, no start code), byte-identical to Python's `encode_picture`. The
/// records are updated the way the Python encoder updates them (derived
/// fields such as `dc_l`, `dc_c`, `part` for B).
pub fn encode_picture(_p: &Parsed, _spss: &Spss, _ppss: &Ppss, _mbs: &mut [Mb]) -> Vec<u8> {
    todo!("port roundtrip_test.encode_picture")
}

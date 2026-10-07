//! CABAC slice-data codec (High profile, 4:2:0, 8-bit, progressive).
//! Port of `frame_dedup/h264cabac.py` (`Coder`, `SliceCodec`).
//!
//! One code path decodes bins from a bitstream or encodes them from the
//! records; keep that design, it is why the round trip is byte-exact.

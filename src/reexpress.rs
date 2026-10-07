//! Re-expressing macroblocks with explicit motion. Port of
//! `verify_motion.reexpress`, `reexpress_compact` and `has_direct`, plus the
//! motion re-expression test (`verify_motion.main`) as `dev reexpress`.

use crate::h264hdr::SliceType;
use crate::mv::Motion;
use crate::records::Mb;

/// An equivalent record with explicit 8x8 partitions carrying `motion`.
pub fn reexpress(_mb: &Mb, _t: SliceType, _motion: &Motion) -> Mb {
    todo!("port verify_motion.reexpress")
}

/// An explicit B record for a skip / direct MB using the coarsest partitioning that fits.
pub fn reexpress_compact(_mb: &Mb, _motion: &Motion) -> Mb {
    todo!("port verify_motion.reexpress_compact")
}

pub fn has_direct(_mb: &Mb) -> bool {
    todo!("port verify_motion.has_direct")
}

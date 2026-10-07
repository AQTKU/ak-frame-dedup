//! Repair of hard drops: exact I_PCM repair, and measured drift repair in
//! explicit mode. Port of `frame_dedup/repair.py`.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use crate::explicit::Sigma;
use crate::h264hdr::{Picture, Ppss, Spss};
use crate::system::Logger;

/// Inputs of [`repair`] (Python: `repair(orig_es, new_es, spss, ppss, pics,
/// new_pics, drop, workers, log, sigma, work, orig_of_out)`).
pub struct RepairArgs<'a> {
    /// The original Annex-B stream.
    pub orig_es: &'a Path,
    /// The deduplicated stream (stage 1) to repair.
    pub new_es: &'a Path,
    /// Parameter sets of the original stream.
    pub spss: &'a Spss,
    pub ppss: &'a Ppss,
    /// The original stream's pictures (after `stream::load`).
    pub pics: &'a [Picture],
    /// The deduplicated stream's pictures (after `stream::load`).
    pub new_pics: &'a [Picture],
    /// Original decode indices that were dropped.
    pub drop: &'a HashSet<usize>,
    pub workers: usize,
    /// Explicit mode: dropped reference picture -> its substitute (or None).
    pub sigma: Option<&'a Sigma>,
    /// Work directory (temporary segment files).
    pub work: &'a Path,
    /// Explicit mode: the original decode index of every new picture.
    pub orig_of_out: Option<&'a [usize]>,
}

/// Returns ({new decode index: replacement NAL}, the report's `repair`
/// section with Python's keys).
pub fn repair(_a: &RepairArgs, _log: &Logger) -> Result<(HashMap<usize, Vec<u8>>, Value)> {
    todo!("port repair.repair")
}

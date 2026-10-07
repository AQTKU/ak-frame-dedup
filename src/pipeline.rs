//! End-to-end pipeline: MKV in, constant-frame-rate deduplicated MKV out.
//! Port of `frame_dedup/pipeline.py`.

use std::path::PathBuf;

use clap::Parser;

/// Command-line arguments; the same options as `python -m frame_dedup`.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "frame-dedup",
    version = crate::VERSION,
    about = "Remove inserted duplicate frames (23.976/24/25 fps content delivered at 29.97/30 fps) \
             from an H.264 MKV without re-encoding. Writes a constant-frame-rate MKV at the source rate."
)]
pub struct Args {
    /// input .mkv
    pub input: PathBuf,
    /// output .mkv (required unless --dry-run)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// write a JSON report here
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// decode output and original and compare every kept frame by POC
    #[arg(long)]
    pub verify: bool,
    /// with --verify: write every differing macroblock (severity, location,
    /// input and output frame numbers) as JSON here
    #[arg(long)]
    pub mb_report: Option<PathBuf>,
    /// work directory for intermediates (default: work/<input name>)
    #[arg(long)]
    pub work: Option<PathBuf>,
    /// keep the work directory
    #[arg(long)]
    pub keep_work: bool,
    /// parallel workers (default: half the logical CPUs); memory-heavy stages
    /// use fewer if RAM is short
    #[arg(long)]
    pub workers: Option<usize>,
    /// never demote an IDR to drop the earlier twin of a pair whose later
    /// twin is an IDR (anchor rebuild); use the fallbacks instead
    #[arg(long)]
    pub no_rebuild: bool,
    /// when an IDR-later pair cannot be rebuilt, keep both twins (A/V
    /// corrected only at black runs) instead of dropping the nearest
    /// low-motion non-reference frame
    #[arg(long)]
    pub keep_idr_pairs: bool,
    /// detect and plan only; write no video
    #[arg(long)]
    pub dry_run: bool,
}

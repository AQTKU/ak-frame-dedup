//! `frame-dedup dev <command> ...`: developer subcommands that dump
//! intermediate results as JSON for comparison with the Python oracle
//! (port/oracle.py). Each command lives in the module it exercises; this is
//! only the dispatch table.

use anyhow::{Result, bail};

pub fn main(args: &[String]) -> Result<()> {
    let Some((cmd, rest)) = args.split_first() else {
        bail!("usage: frame-dedup dev <command> [args...]");
    };
    match cmd.as_str() {
        "framemd5" => crate::decode::dev_framemd5(rest),
        "mux-cmd" | "mux" | "duration-check" => crate::mux::dev_main(cmd, rest),
        "maxdiff" => crate::verify::dev_maxdiff(rest),
        "sysinfo" => crate::system::dev_sysinfo(),
        "headers" => crate::stream::dev_headers(rest),
        "escape" => crate::stream::dev_escape(rest),
        "patch-sps" => crate::stream::dev_patch_sps(rest),
        "plan-stream" => crate::explicit::dev_plan_stream(rest),
        "write-explicit" => crate::explicit::dev_write_explicit(rest),
        "explicit-candidates" => crate::explicit::dev_explicit_candidates(rest),
        "write-stream" => crate::stream::dev_write_stream(rest),
        "new-lists" => crate::stream::dev_new_lists(rest),
        "rebuild-candidates" => crate::rebuild::dev_rebuild_candidates(rest),
        "write-rebuilt" => crate::rebuild::dev_write_rebuilt(rest),
        _ => bail!("unknown dev command {cmd:?}"),
    }
}

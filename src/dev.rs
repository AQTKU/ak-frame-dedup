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
        _ => bail!("unknown dev command {cmd:?}"),
    }
}

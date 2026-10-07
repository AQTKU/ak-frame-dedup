//! `frame-dedup dev <command> ...`: developer subcommands that dump
//! intermediate results as JSON for comparison with the Python oracle
//! (port/oracle.py). Each command lives in the module it exercises; this is
//! only the dispatch table.

use anyhow::{Result, bail};

pub fn main(args: &[String]) -> Result<()> {
    let Some((cmd, rest)) = args.split_first() else {
        bail!("usage: frame-dedup dev <command> [args...]");
    };
    let _ = rest;
    match cmd.as_str() {
        _ => bail!("unknown dev command {cmd:?}"),
    }
}

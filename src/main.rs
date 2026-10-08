use std::path::{Path, PathBuf};

use clap::{CommandFactory, Parser, error::ErrorKind};

use frame_dedup::{dev, pipeline, system};

// The measured repair allocates millions of small vectors per picture from
// many threads at once; mimalloc's per-thread heaps keep them from contending.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("dev") {
        if let Err(e) = dev::main(&argv[2..]) {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
        return;
    }
    let mut args = pipeline::Args::parse();
    system::lower_priority();
    let workers = args.workers.unwrap_or_else(system::default_workers).max(1);
    args.workers = Some(workers);
    // Same checks, in the same order, as `__main__.py` (argparse's `ap.error` exits with 2).
    let fail = |msg: String| -> ! { pipeline::Args::command().error(ErrorKind::ValueValidation, msg).exit() };
    if !args.dry_run && args.output.is_none() {
        fail("-o/--output is required".into());
    }
    if let Some(out) = &args.output {
        if abspath(out) == abspath(&args.input) {
            fail("output must differ from input".into());
        }
    }
    if !args.input.exists() {
        fail(format!("{} does not exist", args.input.display()));
    }
    if args.mb_report.is_some() && !args.verify {
        fail("--mb-report needs --verify".into());
    }
    if args.work.is_none() {
        let stem = args.input.file_stem().map(PathBuf::from).unwrap_or_default();
        args.work = Some(Path::new("work").join(stem));
    }
    // `pipeline::run` returns the exit code.
    std::process::exit(pipeline::run(args));
}

/// Python's `os.path.abspath` (made absolute and normalised, symlinks untouched).
fn abspath(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

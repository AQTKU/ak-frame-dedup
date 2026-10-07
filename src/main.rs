use clap::Parser;

use frame_dedup::{dev, pipeline};

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.get(1).map(String::as_str) == Some("dev") {
        if let Err(e) = dev::main(&argv[2..]) {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
        return;
    }
    let args = pipeline::Args::parse();
    let _ = args;
    eprintln!("the pipeline is not ported yet");
    std::process::exit(1);
}

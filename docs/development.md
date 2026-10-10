# Development

## Building

- Rust: a recent stable toolchain (the crate uses edition 2024; it is
  developed on 1.97).
- `cargo build --release` gives `target/release/frame-dedup`. The
  likelihood tables (`src/cadence_model.json`) are compiled in.
- `cargo build --profile fastrel` is an optimised build without LTO that
  links much faster, for day-to-day work.
- The dev profile uses `opt-level = 1`: CABAC and motion derivation are far
  too slow unoptimised.
- `.cargo/config.toml` limits cargo to 6 parallel jobs, because the
  development machine has crashed under full load.

The version number comes from `Cargo.toml` (`frame-dedup --version`, the
report's `version` field, and the release workflow all read it).

## Releases

`.github/workflows/release.yml` builds `frame-dedup` for Windows x64 and
Linux x64 (a static musl binary that runs on any distribution) and
publishes a GitHub release tagged `v<version>` from `Cargo.toml`. Run it
from the Actions tab (*Release* → *Run workflow*). It refuses to run if
that tag already exists, so bump the version in `Cargo.toml` first.

## Tests

`cargo test --release`. The oracle tests in `tests/` compare with cached
outputs of the Python v1.2.0 implementation under `work/port/oracle/` and
skip when those are missing. The Python implementation and its oracle
tooling were removed in 2.0.0; they remain in git history (commit
`571a346`).

`tools/e2e.ps1` runs the release build on the test files and compares
per-stream MD5s with reference outputs.

`tools/mb_review.py <input.mkv>` runs the release build with `--verify
--mb-report --mb-threshold 3` and writes `review/<name>-mb3/index.html`: the
worst differing frames (default 150), each with the original and output
switchable or flickering, a difference overlay, the listed macroblocks
outlined, and lossless 4x crops of its worst blocks. `--mb-json mb.json
--output out.mkv` reuses an earlier run; `--threshold` and `--max-frames`
adjust it. Needs Python with numpy and Pillow.
`python -m http.server --directory review` serves the pages.

## Dev tools

`frame-dedup dev <command>`:
- `train-model <labelled.mkv>` retrains the likelihood tables
  (`src/cadence_model.json`). The tables were trained on sample A, whose
  timestamps label the duplicates (see [results.md](results.md) for the
  test files).
- `roundtrip in.h264` is the CABAC round trip.
- `recon in.h264 [--periods A-B]` checks the inter reconstruction of
  `recon.rs` against ffmpeg decodes without deblocking (see "Residual
  corrections" in [how-it-works.md](how-it-works.md)).
- `reexpress in.h264 <P dec idx> <count>` is the motion re-expression test.
- `headers`, `records`, `motion`, `features`, `detect` and others dump
  intermediate results as JSON.

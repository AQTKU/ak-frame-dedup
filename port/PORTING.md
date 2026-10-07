# Porting frame_dedup to Rust

The whole tool is being rewritten in Rust as one crate (`Cargo.toml`, `src/`) at the repository root. The Python package `frame_dedup/` (v1.2.0) stays in the tree, **unchanged**, as the oracle until the port reproduces it; then it is deleted. Read `README.md` ("How it works", "Explicit mode") for what the tool does.

## The rule: byte-identical

Every Rust module must reproduce its Python counterpart exactly: same parsed values, same records, same motion fields, same NAL bytes, same decisions, same report numbers. The final acceptance test is per-stream MD5 equality of the output MKVs with v1.2.0's. "Close" is a bug. When Python does something odd (an off-by-one, a quirky tie-break, a regex's non-overlapping semantics, numpy's float32 summation order), reproduce it and leave a short comment saying it mirrors the Python.

Port **behaviour**, not syntax: use Rust structs, enums, slices and indices instead of dicts; mmap instead of reading files; threads instead of processes. But keep the algorithm and its order of operations, since order decides tie-breaks and floating-point results.

## Layout

One module per Python module (see `src/lib.rs`):

| Rust | Python |
|---|---|
| `bits.rs` | escape / unescape / set_bits / `Bits` (h264hdr.py), to_bits / from_bits (roundtrip_test.py) |
| `h264hdr.rs` | h264hdr.py (types are already defined there) |
| `records.rs` | macroblock record type, shared tables (already written) |
| `cabac.rs` | h264cabac.py |
| `picture.rs` | roundtrip_test.decode_picture / encode_picture |
| `mv.rs` | h264mv.py (`Motion` already defined) |
| `reexpress.rs` | verify_motion.py |
| `patch.rs` | patch.py |
| `explicit.rs` | explicit.py |
| `stream.rs` | stream.py |
| `rebuild.rs` | rebuild.py |
| `repair.rs` | repair.py |
| `decode.rs` | decode.py |
| `verify.rs` | verify.py |
| `mux.rs` | mux.py |
| `evidence.rs`, `cadence.rs`, `plan.rs` | the same |
| `system.rs` | system.py |
| `pipeline.rs`, `main.rs` | pipeline.py, `__main__.py` |
| `dev.rs` | dispatch for `frame-dedup dev <cmd>` dump commands |

`sniff.py` is not ported (unused). `roundtrip_test.main` and `verify_motion.main` become `dev` commands.

**Shared types are fixed interfaces**: `h264hdr::{Sps, Pps, SliceHeader, Picture, ...}`, `records::Mb` and friends, `mv::Motion`, `picture::Parsed`, and the stub function signatures. Implement against them. If one is genuinely wrong for the job, change it minimally and say so in your final report; don't silently redesign it.

Stay inside the files your task names. You may add `pub fn` helpers to your own modules. If you need a change in another module, report it instead of making it (other agents are working on those files in parallel).

## Building

```
cargo build --profile fastrel        # day-to-day: optimised, quick to link
cargo test --profile fastrel
```

`.cargo/config.toml` limits builds to 6 jobs. Dependencies are in `Cargo.toml` (anyhow, clap, memmap2, serde, serde_json, windows-sys); add another only if it is clearly worth it, and say so in your report. Errors: `anyhow::Result` where Python raises; keep Python's error messages.

## Checking against the Python oracle

Each module gets a `dev` command in Rust and an equivalent dumper in `port/oracle.py` (add a subcommand there; run it with `python -m port.oracle <cmd> ...` from the repository root). Both print the same canonical JSON. Compare with `python -m port.compare a.jsonl b.jsonl` or your own script.

- Canonical JSON: sorted keys, compact separators (`port/canon.py: text()`; `records::canonical_text`). Python tuples become lists, None becomes null, booleans stored as ints in Python stay ints (0/1). Enums are their Python strings ('P', 'B', 'I4', '16x16', ...).
- For big outputs (macroblock records, motion fields) emit one JSON object per picture per line (JSONL) with a SHA-256 digest of the canonical text, and dump the full object only on request (`--full`) or for mismatching pictures.
- The Python oracle is slow (the CABAC parse is about 0.8 s per 1080p picture). Cache its output under `work/port/oracle/` and reuse it.
- Write Rust unit tests for small pure functions, and an integration check (a `dev` command or a test that skips when the data is missing) for the oracle comparison. Record in your report exactly what you compared and the result.

## Test data (outside git)

- `C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port\clips\{sample,foliage,smoke,mmco}.h264`: 6-second Annex-B clips. `mmco` is the Amazon x264 encode (explicit mode: MMCO, list modification, weighted P, spatial direct, B-pyramid); the others are the Canadian encodes (temporal direct, non-reference B).
- The full test MKVs are in the repository root (`sample.mkv` 1 GB, `foliage.mkv`, `smoke.mkv` 2 GB, `nodupes.mkv`, `mmco-or-ref-list-modification.mkv` 6 GB), with outputs and reports of earlier runs in `work/` (`*_report.json`, `work/mmco/clip_report.json`, `full_report.json`). `work/mmco/clip.mkv` is a 5-minute cut of the Amazon file.
- If you work in a git worktree, the media is not there: use the absolute paths above. The Python package in your worktree is identical to the main checkout's.

## Machine limits (important)

This machine (32 threads, 31.7 GB) has crashed under full load. Other agents are building and testing at the same time.
- Never run the full pipeline (Python or Rust) on a full-length file unless your task says so.
- At most one ffmpeg process at a time from your work, with `-threads 2`.
- Keep test runs to the clips; a Python oracle run over more than ~200 pictures needs a reason.
- Don't start long-running background processes and leave them.

## Done means

- `cargo build --profile fastrel` passes with no warnings in your files, and `cargo test --profile fastrel` passes.
- Your oracle comparison shows identical output on all four clips (or on what your task specifies), and you report the commands and numbers.
- Commit your work on your branch with a clear message (end it with the `Co-Authored-By` line you were given, if any). Do not push, merge, or touch other branches.
- Your final message: what you ported, any deviation from the Python and why, interface changes, what you compared and the results, timing (Rust vs Python) where meaningful, and anything left undone.

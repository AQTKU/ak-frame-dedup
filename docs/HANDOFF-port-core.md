# Handoff: port the bitstream core to a compiled extension

This brief is for a fresh session. Read `README.md` first ("How it works" and "Explicit mode"). This is stage 1 of 2. Stage 2, `docs/HANDOFF-reconstruction.md` (pre-deblock reconstruction and a byte budget for patches), builds on this port.

## Why

The tool is correct but slow on the Amazon file (`mmco-or-ref-list-modification.mkv`, 84 min, 2,716 hard pairs and 423 anchor rebuilds):

- A full run with `--verify` takes 94 min.
- The repair stage alone takes 79 min, about 11 CPU-hours across 9 workers.

Almost none of that is real math. It is the Python interpreter running millions of tiny integer steps. Profile of one typical repair period (about 30 s; `work/mmco/t_prof.py`):

| work | share | nature |
|---|---|---|
| CABAC parse and re-encode (`h264cabac.py`: `Coder`, `SliceCodec`) | ~57% | 6.6 million bins: a few integer ops and a table lookup each, about 1 µs per bin in Python |
| `h264mv.derive_motion` | ~23% | about 130k 4x4 blocks per picture: neighbour lookups and medians, several calls per picture |
| ffmpeg decodes and numpy frame diffs | ~15% | already compiled |
| the rest | ~5% | orchestration |

numpy can't help: every step depends on the one before it. In Rust or C++ those steps cost about 10-20 ns, a 50-100× speedup on about 80% of the repair time. Expect the full Amazon repair to drop from 79 min to roughly 10-15 min. After that, ffmpeg's decodes in the measure loop dominate.

Stage 2 adds motion-compensation interpolation, intra prediction and transforms per macroblock. That is only practical with a compiled core.

## Goal and acceptance criterion

**Byte-identical output.** With the extension, every output must be the same file as with the pure-Python v1.2.0, per stream MD5 (`ffmpeg -i X -map 0:v -c copy -f md5 -`, also for audio and subtitles):

- `sample.mkv`, `foliage.mkv` and `smoke.mkv` (old path, exact I_PCM repair, anchor rebuilds);
- `work/mmco/clip.mkv` (5 minutes of the Amazon file, explicit mode);
- once at the end, the full Amazon file.

This is a strong, simple criterion. The port is a pure speed change, and every intermediate step can be checked against the Python implementation, which stays in the tree as the oracle and as a fallback.

**Before porting anything, regenerate v1.2.0 baselines.**
- `work/sample_out.mkv` is confirmed identical for v1.2.0.
- `work/foliage_out.mkv` and `work/smoke_rb_out.mkv` are v1.1 outputs. Rerun them with v1.2.0, expect them to be identical, and say so if they are not.
- `work/mmco/clip_out.mkv` was made mid-development and **must be regenerated**.
- `work/mmco/full_out.mkv` comes from the second full run. Later commits changed only memory use; regenerate it if you need an exact baseline.
- Keep the reports (`--report`) too: the PCM counts and rounds must match as well.

## Scope: what moves, what stays

Port, in this order. Each step is useful on its own.

1. **CABAC slice-data codec**: `Coder` and `SliceCodec` in `h264cabac.py` (about 1,040 lines).
   - One code path decodes bins from a bitstream or encodes them from records.
   - Keep that design: it is why the round trip is byte-exact.
   - The tables are in `frame_dedup/cabac_tables.json`.
   - Scope: High profile, 4:2:0, 8-bit, frame_mbs_only, one slice, CABAC, transform_8x8, I_PCM.
2. **Motion derivation**: `h264mv.derive_motion` and `Motion` / `CompactMotion` (about 300 lines).
   - P_Skip, mv prediction including the 16x8 / 8x16 rules, temporal direct, spatial direct (`spatial_pred`, colZero), direct_8x8_inference.
   - **Target mode** (`target=`): solve mvds so that explicit partitions carry given motion.
3. **The per-picture work of `patch.patch_pictures`**:
   - PCM substitution;
   - skip / direct MBs made explicit (`to_explicit` → `verify_motion.reexpress` / `reexpress_compact`);
   - the conversion loop;
   - QP chain and `fix_qp`;
   - mvd solving;
   - encode.

   The Python loop over pictures, the cache logic (`sig`, `cache`) and the choice of what to patch can stay in Python.

Stays in Python:
- headers (`h264hdr.py`, `explicit.py`);
- DPB simulation (`dpb_walk`);
- cadence detection, planning, the measure loop in `repair.run_task_explicit`;
- `rebuild.py` orchestration, mux, verify.

Most of these are cheap. If a later profile says otherwise, port more.

## Interface: keep the records native

Converting about 8,000 macroblock dicts per picture between Rust and Python would eat much of the gain. Make the boundary *picture-level operations* on a native object:

- `Picture.parse(nal_bytes, slice_hdr, sps, pps)`
  - Parses and holds the records natively.
  - `slice_hdr` is the dict from `h264hdr.parse_slice`. The codec needs `data_bit`, `qp`, `cabac_init_idc`, `slice_type`, `num_ref_l0/l1`, `direct_spatial` and the PPS / SPS flags.
- `pic.motion(ctx, col=None, target=None) -> Motion`
  - `ctx` carries the picture's reference context as small arrays: list entries as picture ids, POC per id (temporal direct and implicit weights), the current POC.
  - `col` is the co-located picture's `Motion`.
  - Expose numpy views (`intra`, `refidx[2]`, `mv[2]`, `refpic[2]`) for `PixelOracle`, `changed_blocks`, `same_motion` and the tests.
- `pic.patch(pcm={(x, y): samples}, keep=Motion, col=Motion, col_pcm=set, compact=bool) -> (nal_bytes, stats, Motion)`
  - Does exactly what one iteration of `patch_pictures`' loop body does.
  - `stats` has the same keys: `pcm`, `direct_explicit`, `skip_explicit`, `qp_carrier`, `bytes_before`, `bytes_after`.
- `pic.records()` returns Python dicts in **today's format**. It is slow and meant only for tests and tools (`verify_motion`, `roundtrip_test`, `rebuild._build_b`). It means nothing else has to be ported at once.

**Today's record format** (what `records()` must reproduce, and what the tests compare):
- Common keys: `x`, `y`, `skip`, `kind` (`'I4' 'I8' 'I16' 'PCM' 'P' 'B' 'skip'`), `intra`, `pcm`, `i16`, `direct16`, `t8`, `cbp`, `qpd`, `dcL`, `dcC` [2], `cmode`.
- Intra: `ipred` and `imodes` (16 or 4 entries; `ipred` −1 means the predicted mode was used), `i16pred`, `i16cbpC`, `i16cbpL`.
- Inter:
  - `part` (`'16x16' '16x8' '8x16' '8x8' 'D'`);
  - `btype` for B (index into `B_TYPES`);
  - `sub` [4] for 8x8;
  - `refs` [2][parts] (None when the list is unused);
  - `mvds` [2][parts] of (x, y) or None. For 8x8: [2][4][subparts].
- `coef`: a dict keyed by `('LDC',)`, `('L8', i8)`, `('L4', idx)`, `('CDC', comp)`, `('CAC', comp, i)`, each a list of levels in scan order.
- PCM: `samples` (384 values: Y then U then V).

**Termination:** x264 writes a pseudo-random bit after the CABAC flush. Decoded data never reproduces those trailing bits; our encoder writes the canonical flush. The Rust encoder must produce **exactly what the Python encoder produces**, or outputs won't be byte-identical.

## Validation ladder

Each rung uses Python as the oracle. Write these as pytest tests that skip when the test MKVs or the extension are missing.

1. **CABAC decode:** `records()` equals Python's `decode_picture(...)['mbs']` for every picture of a 20 s clip of each of the four files.
2. **CABAC encode:** a byte-exact round trip on all pictures (`roundtrip_test` semantics), *and* identical bytes to Python's `encode_picture` for edited records (PCM-swapped, re-expressed).
3. **Motion:** fields equal to Python's for every 4x4 block, both lists, refidx / mv / refpic / intra, on P, temporal-direct B and spatial-direct B pictures. The motion re-expression test (`python -m frame_dedup.verify_motion`) must still decode bit-identically with the native path.
4. **Patch:** identical NAL bytes and stats compared with Python `patch_pictures`.
   - Record a corpus of real tasks: `work/mmco/t_task.py` rebuilds the full run's repair tasks from `work/mmco/full_work`.
   - Include: the old-mode hard drops (sample, smoke), an explicit-mode period with forced B pictures, a merged rebuild period (891), and old-path rebuild B1-B3 (`tests/test_rebuild.py`).
5. **End to end:** the MD5 criterion above.

Keep `FRAME_DEDUP_PURE=1` (or a missing extension) selecting the Python path, so the oracle stays runnable.

## Build setup on this machine

- Rust 1.97, `stable-x86_64-pc-windows-msvc`, with `cargo`. The owner has built Rust tools before (`dovi_tool`).
- Visual Studio 2019 is installed, which provides the MSVC linker.
- Python 3.12.10. **maturin is not installed** (`pip install maturin`; ask first).
- Suggested layout:
  - a `native/` crate using PyO3 with `abi3-py312`, built with `maturin develop --release`;
  - Python imports it as `frame_dedup._native`, with a fallback when it's absent.
- Cargo profile: `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`.
- Release the GIL in long native calls (`py.allow_threads`), even though the pipeline uses processes today.

## Practicalities

- **Machine:** 32 threads and 31.7 GB of RAM, with a history of crashing under full load.
  - Keep the throttled defaults: half the workers, below-normal priority, pools sized by `system.pool_size`.
  - Run one heavy pipeline at a time.
  - Explicit repair workers peak at about 1.0 GB in Python. Native records are much smaller, so `REPAIR_PEAK_EXPLICIT` and `PARSED_CAP` can probably go up afterwards. Measure with `work/mmco/t_worker.py <period>`, which runs one task in a spawned worker and prints its peak RSS.
- **Timing:**
  - `work/mmco/t_prof.py <period>` profiles one clip task.
  - `work/mmco/t_task.py <period>...` runs full-run tasks in-process.
  - Report before and after numbers per stage, and the full Amazon runtime.
- **A free experiment first (15 min, optional):** PyPy often speeds this style of code up 5-10× unchanged. numpy and multiprocessing make it uncertain. If it works, it's a useful data point, not a replacement for the port.
- **Owner's preferences:** bit-exact wherever possible; noise-level differences confined to frames that read a substitute are acceptable; visible substitutes are the worst outcome. This port must not change any output.
- **Work files:** `work/mmco/` holds the Amazon intermediates:
  - `in.h264`, the extracted stream (5.9 GB);
  - `full_work/` (in.h264, dedup.h264, out.h264, run.log);
  - `clip.mkv` and `clip_work/`;
  - two full-run reports (`full_report_v1.json` is the first run, `full_report.json` the second).

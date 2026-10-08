# Handoff: spend more of the freed bytes on repair (byte budget, then residual corrections)

> **Status (2026-10-08, branch `repair-budget`, version 2.1.0):** steps 1 and 2 are done and are the
> default; step 3 was tried and does not pay (see README, "Residual corrections" and Limitations).
> `recon.rs` reconstructs inter MBs bit-exactly (checked with `-skip_loop_filter all` on all four
> files); corrections use a QP ladder (QP−6k doubles levels exactly), predicted references and a
> deblocking-aware target. Full Amazon file: MBs ≥ 8 from ~2.1M to 10, worst 103 → 38, +122 MB
> (24% of the freed bytes). smoke.mkv's rebuilt leading B pictures: ≥ 16 from 12.8k to 27.
> The side idea (keep both twins and trim at a sync point) is still open.

This brief is for a fresh session. Read `README.md` first, especially "How it works" and "Explicit mode". `docs/HANDOFF.md` and `docs/HANDOFF-anchor-rebuild.md` are older design background; skim them only if something here is unclear.

## The goal in one paragraph

The tool removes the duplicate frames that streaming services insert when converting 23.976 fps film to 29.97 fps, by H.264 bitstream surgery with no re-encode. Dropping a picture that other pictures read changes what they predict from. On the x264 B-pyramid encodes (the Amazon file) the dropped twin D and its substitute K are two separate encodes of the same source frame, so every block that read D now reads slightly different pixels. The tool repairs this by measurement: decode, compare with the original, and replace macroblocks that drifted by 16 levels or more with I_PCM copies of the original pixels. Dropping the duplicates frees about 7% of the stream. The repair spends only about 12% of that freed room. **The owner wants more of that room spent on fidelity.** The plan:
1. replace the fixed threshold with a byte budget;
2. make each patch much cheaper with residual corrections instead of PCM.

## Where things stand (v2.0.0, branch `main`)

- The whole tool is Rust (crate at the repo root, binary `frame-dedup`). It was ported from Python v1.2.0 with per-stream MD5-identical output. The Python is gone; its last commit is `571a346`.
- 2.0.0 also added A/V sync points at hard cuts and static frames (`plan::sync_correct`). This is done and verified; nothing to do there.
- There is no git remote. Commit on a feature branch and merge into `main` when the owner OKs it.
- Build: `cargo build --release`. Tests: `cargo test --release`, which takes about 2 minutes. The three oracle tests in `tests/` compare against cached Python outputs in `work/port/oracle/`.

### The three kinds of input

- **The Canadian encodes** (`sample.mkv`, `foliage.mkv`, `smoke.mkv`) have non-reference B pictures and temporal direct prediction.
  - Their few hard drops (26-93 per file) are repaired *exactly*: header-only reference analysis plus I_PCM (`repair::run_task_exact`).
  - They leave 1-8 levels around the PCM patches, from deblocking across PCM edges.
- **x264 B-pyramid encodes with MMCO and list modification** (`mmco-or-ref-list-modification.mkv`, Amazon) go through **explicit mode** (`src/explicit.rs`).
  - Headers are rewritten so that every list entry and the DPB contents equal the originals, with D replaced by K.
  - D and K differ by encode noise everywhere: luma MAD 0.5-1.2, and almost no MB matches exactly.
  - Repair is by measurement (`repair::run_task_explicit`, `src/repair.rs` around line 521). Per IDR period of the new stream:
    1. patch, write the segment and decode it with ffmpeg;
    2. decode the original in lockstep;
    3. diff each picture over the visible area (`verify::visible_maxdiff`);
    4. queue MBs at `DRIFT_THRESHOLD` = 16 or more, worst first, capped at `DRIFT_MAX_FRACTION` = 2% of the picture (163 MBs at 1080p);
    5. patch in decode order up to the first reference picture (`DRIFT_SEQUENTIAL_ROUNDS` = 6, then everything), and repeat (`DRIFT_MAX_ROUNDS` = 12).
- **nodupes.mkv** must fail out with exit code 2. It's a regression check only.

### Baseline numbers to beat

5-minute Amazon clip, `work/mmco/clip.mkv`. Rust 2.0 output: `work/port/e2e/clip.rs.mkv`, report `clip.json`, per-MB export `clip.mb.json`.

| | |
|---|---|
| duplicates dropped / hard pairs | 1,798 / 154 |
| video stream: original → after drops → after repair | 355.8 MB → 331.7 MB (−24.1 MB) → 334.8 MB (+3.0 MB) |
| frames bit-exact / differing | 4,623 / 2,572 |
| differing frames: median frame MAD | 0.11 |
| differing frames: median / p90 / max pixel error | 9 / 15 / 41 |
| MBs off by ≥ 4 / ≥ 8 / ≥ 16 / ≥ 32 (luma, visible) | 1.21M / 176k / 14.3k / 48 |
| pictures still over the threshold (PCM cap hit) | 39 |

Full Amazon file. Rust 2.0 with sync points: `work/sync/full.mkv` and `work/sync/full.json`.

| | |
|---|---|
| frames | 150,499 → 120,400; 2,716 hard pairs; 423 anchor rebuilds (they chain: up to 18 IDR periods merge into one new period of up to 864 pictures) |
| repair | 149,483 PCM MBs, **+56.2 MB** (1.0% of the stream; the drops freed about 7%); 202,687 skips and 130,196 direct MBs made explicit |
| left over the threshold | 291 pictures, **all at the 2% PCM cap**; visible worst 56 |
| verify | 73,476 bit-exact, 46,924 differ, **0 outside** hard-drop neighbourhoods and rebuilt leading pictures |
| runtime | 33.5 min with 16 workers: proxy 3.6 min, repair 21 min, verify 8 min. Process peak 6.7 GB. |

## Step 1: a byte budget instead of a fixed threshold

Replace the stopping rule of the measure-and-patch loop. The loop itself stays as it is.

- **Budget per repair task** (one IDR period of the new stream; merged periods are one task): β × (bytes of the pictures dropped in it).
  - Dropped picture sizes are in the headers: sum `slices[..].size` of the original pictures.
  - Start with β ≈ 0.5. Unspent budget may carry over, but tasks run in parallel, so per-task budgets are simpler.
- **Spend worst-first by a visibility score,** not raw max difference:
  - **Propagation:** an error in a reference picture lives until the next IDR. Weight it by the pictures decoded after it in the period, or by the downstream error measured last round.
  - **Masking:** drift in flat or static areas is far more visible than in texture or motion. Weight it by the inverse local variance of the original MB.
  - **Luma over chroma.**
- **Floor:** stop below about 3-4 levels whatever the budget. Below that, the kept twin is as faithful to the source as the original was.
- **Keep a per-picture cap** in some form. It bounds the work per round, and a picture that needs more than a few percent of PCM is better served by idea 2.
- **Cost:** PCM is about 400 bytes per MB. Measure the actual cost from the patch stats rather than assuming it.
- **Report** budget, spend and leftover per period, plus the distribution of remaining differences. `--mb-report` already exports every differing MB.

With PCM as the only currency, even the full budget only takes the threshold from 16 to roughly 10-12 on busy content. Step 2 is what makes a low floor affordable. Step 1 is still worth doing first: it is small, it puts the scoring and reporting in place, and step 2 slots into it as a cheaper kind of patch.

## Step 2: residual corrections

The idea: keep the MB's mode and motion, and add a correction to its residual instead of replacing it with PCM.

### The formulation (simpler than it first looks)

For an inter MB at QP q in a picture that reads K where it read D:

```
orig recon = clip(orig_pred + r),   r = residual from the original coefficients c
new recon  = clip(new_pred + r'),   want r' ≈ r + (orig_pred - new_pred)
```

- The dequantiser and inverse transform are linear in the levels (up to the small rounding in dequantisation at low QP).
- So the correction can be made **in the coefficient domain**: `levels' = c + quantise(forward_transform(orig_pred - new_pred), q)`.
- Only the drift term is quantised, so the leftover error is a fraction of the quantiser step: about ±1-3 levels, against up to 15 today. A correction should cost tens of bytes, against about 400 for PCM.

This needs **no intra prediction, no full picture reconstruction and no deblocking filter**:
- **Both predictions come from frames ffmpeg already gives us** in the measure loop. References are stored after deblocking, and that is exactly what motion compensation reads.
  - `orig_pred` uses the original decode's frames and the original motion.
  - `new_pred` uses the new stream's decoded frames and the same motion.
- **Motion** comes from `mv::derive_motion`: per-4x4 ref indices, reference pictures and mvs, temporal and spatial direct included. It was validated bit-exact.
- **Intra MBs** never read D directly; they only drift through drifted neighbours. Let them fall back to PCM.

### What to build (new module, e.g. `src/recon.rs`)

1. **Inter prediction:**
   - luma 6-tap quarter-pel (8.4.2.2.1);
   - chroma eighth-pel bilinear (8.4.2.2.2);
   - partitions down to 4x4;
   - bi-prediction averaging;
   - weighted prediction. **The Amazon PPS has `weighted_pred = 1` (explicit P weights) and `weighted_bipred_idc = 2` (implicit B weights from POC distances, 8.4.2.3).**
2. **Explicit weight tables.** `h264hdr::parse_slice` (around `src/h264hdr.rs:595`) parses `pred_weight_table` but throws the values away. Store `luma_log2_weight_denom`, `chroma_log2_weight_denom` and the per-reference weights and offsets on the slice header.
3. **Forward 4x4 / 8x8 integer transforms and a quantiser** at the MB's QP. Use plain rounding first, and trellis only if bytes matter. Inter MBs use luma 4x4 or 8x8 (no Hadamard) and chroma 2x2 DC plus 4x4 AC. Chroma QP comes from `chroma_qp_offset` (−3 on the Amazon file) and the QP table. Scaling lists are flat on all four test files (`scaling_matrix: 0`), but fail loudly if not.
4. **Writing a correction into the CABAC record** (`records::Mb`: `coef` (`Coefs`: `l4`, `l8`, `cdc`, `cac`), `cbp`, `t8`, `qpd`, `dc_c`, `skip`, `direct16`):
   - Recompute `cbp` and the coded-block flags from the new coefficients. Check how `src/cabac.rs` decides what to write.
   - **P_Skip** becomes explicit P_L0_16x16 with the same motion. `patch::to_explicit` does this and re-solves the mvd.
   - **B_Skip** becomes B_Direct_16x16 with residual; its derived motion is unchanged.
   - **Transform size:** if the MB had no luma residual, `t8` was not coded. Choosing 4x4 is always legal; 8x8 has constraints on sub-partitions and direct inference.
   - **QP:** keep the MB's QP. A skip MB has QP = QPpred, so `qpd = 0` keeps it. If a correction ever takes cbp to 0, `qpd` stops being coded and the QP chain shifts. `patch.rs` already maintains the QP chain (`qp_carriers`), so route through it.
   - Everything else in the picture stays exact around a modified MB through `patch::patch_pictures`: skips that must become explicit, mvd re-solving, direct blocks. Corrections should be a new kind of entry in its `patches` / `targets` alongside PCM, not a separate path.
5. **In the measure loop,** try a correction first, then PCM if the MB is intra, or if a correction at q can't reach the floor. A second option before PCM is a correction at a lower QP via `mb_qp_delta`. Correct reference pictures first, in decode order, as today.

### Validate the prediction before using it: the no-deblock trick

- Rewrite one picture's slice header with `disable_deblocking_filter_idc = 1`. `explicit::rewrite_slice` already rebuilds headers bit by bit; extend it with that field.
- Decode the segment with ffmpeg. That picture comes out pre-deblock, and every picture before it is unchanged.
- For MBs with cbp = 0, your `orig_pred` must equal the decoded MB bit for bit.
- To check MBs with residual too, add the 4x4 / 8x8 inverse transform and dequantiser (about 150 lines) and compare `pred + residual`.
- Do this on hundreds of pictures across all four test files (P, B, weighted, direct, both encoders) before relying on it.
- Make it a `frame-dedup dev ...` command, as the other modules do (`src/dev.rs`).

### Caveats to measure, not assume

- **Deblocking strength:** an edge gets bS = 2 when either 4x4 block has coefficients. A correction that adds coefficients changes the filtering across that edge. This is a second-order error; the next measure round sees it.
- **Clipping** at 0/255, and dequantisation rounding at low QP, make the linearity approximate. Again, the loop measures.
- **Near-exact, not exact.** That is fine per the owner: noise-level differences are acceptable, and visible substitutes are not.

## Step 3 (optional): corrections on the Canadian files

Their exact repair leaves 1-8 levels around PCM patches from deblocking at PCM edges. With residual corrections, many of those MBs need no PCM at all. Only try this after steps 1 and 2 work on the Amazon file. Compare with the 2.0 outputs (`work/port/e2e/*.rs.mkv`, which are identical to v1.2.0).

## Side idea, independent and small

Drop rule 7, the "visible substitute", fires when an IDR-later pair can't be anchor-rebuilt. It drops a nearby real frame: a visible one-frame stutter next to a held frame.

- Since 2.0.0 the planner has sync points at cuts and static frames.
- So *keep both twins and trim at the next sync point* is usually better: one held frame, plus an invisible trim at a cut.
- It costs up to about a frame of A/V offset until the cut. The owner's tolerance: 1 frame is fine, 2 feels weird, 3 is noticeable.
- It only applies where a usable sync point is close (say within 2 s); keep rule 7 otherwise.
- None of the test files hits rule 7 with default settings. `smoke.mkv --no-rebuild` gives 300 visible substitutes to test on. `--keep-idr-pairs` already keeps both everywhere: max offset 3.8 frames, verify clean.

## Practicalities

- **Machine:** 32 threads and 31.7 GB of RAM. It once crashed under full load (likely marginal RAM), so keep the defaults: half the workers, below-normal priority, pools sized by `system::pool_size`.
  - Run **one heavy pipeline at a time.**
  - The owner also uses this PC. **Ask before starting a long run** (the full Amazon file takes about 35 min) and stop when asked.
- **Iteration targets:**
  - `work/mmco/clip.mkv` (5 min, Amazon). The full pipeline with `--verify` takes about 1.5 min.
  - `frame-dedup dev repair <scenario.json>` replays the repair stage alone from a recorded plan (drops, sigma, rebuilds). `work/port/oracle/full_scenario.json` is the full Amazon file. It has no per-period filter yet; adding one (for example, the worst capped periods such as 891 and 2006) is a cheap way to iterate on single periods.
- **Measuring:**
  - `--verify` reports per-frame and per-MB stats.
  - `--mb-report mb.json --mb-threshold N` lists every differing MB with its location and severity.
  - `review/` holds static HTML review pages. `.claude/launch.json` serves them with `python -m http.server`.
  - The owner reviews results **by eye**: give before/after crops of the worst MBs (original, output, difference) for every change.
- **Regression:** `pwsh tools/e2e.ps1 <names>` compares per-stream MD5s with reference outputs. Once repair changes on purpose, MD5s will differ on the Amazon files. The Canadian files must stay identical until step 3, and `verify.differing_outside_hard_neighbourhoods` must stay 0.
- **Owner's preferences:**
  - Bit-exact wherever possible.
  - Noise-level differences confined to frames that read a substitute are acceptable.
  - Visible artefacts and stutters are the worst outcome.
  - Keep the README current: it documents every design decision and result.

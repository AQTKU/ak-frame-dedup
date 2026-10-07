# Handoff: spend the freed bytes, cheaper patches from pre-deblock reconstruction

This brief is for a fresh session. Read `README.md` first, especially "Explicit mode". `docs/HANDOFF.md` and `docs/HANDOFF-anchor-rebuild.md` give the earlier design background.

## Where things stand (v1.2.0)

The tool handles three kinds of input.

- **The Canadian encodes** (`sample.mkv`, `foliage.mkv`, `smoke.mkv`) have non-reference B pictures and temporal direct prediction. Their hard drops are rare (26-93 per file) and are repaired exactly with I_PCM.
- **x264 B-pyramid encodes with MMCO and list modification** (`mmco-or-ref-list-modification.mkv`, Amazon) go through **explicit mode** (`frame_dedup/explicit.py`):
  - The headers of the kept pictures are rewritten so that every list entry and the DPB contents equal the originals, except that the dropped twin D is replaced by its substitute K.
  - Here D and K are two separate encodes of the same source frame and differ by encode noise everywhere: luma MAD 0.5-1.2, and almost no macroblock matches exactly. Exact repair would PCM nearly every block that reads D: 78,020 MBs (+30 MB) for 12 drops.
  - Explicit mode therefore **repairs by measurement** (`repair.run_task_explicit`):
    1. Decode the IDR period of the new stream.
    2. Compare it with the original over the visible area.
    3. I_PCM every MB that is off by `DRIFT_THRESHOLD` = 16 or more, in decode order up to the first reference picture. Patching that picture also fixes what is predicted from it.
    4. Repeat. At most `DRIFT_MAX_FRACTION` = 2% of any picture is patched.

### What the measured repair leaves (5-minute clip of the Amazon file)

| | |
|---|---|
| duplicates dropped / hard pairs | 1,798 / 154 |
| video stream: original → after drops → after repair | 355.8 MB → 331.7 MB (−24.1 MB) → 334.8 MB (+3.0 MB) |
| frames bit-exact / differing | 4,623 / 2,572 |
| differing frames: median frame MAD | 0.11 |
| differing frames: median / p90 / max pixel error | 9 / 15 / 41 |
| MBs off by ≥ 4 / ≥ 8 / ≥ 16 / ≥ 32 (luma, visible) | 1.21M / 176k / 14.3k / 48 |
| pictures still over the threshold (PCM budget exhausted) | 39 |

The repair spends **12% of the bytes the drops free**. The owner is interested in spending more of that room on more patches. The full-length results are in the last section.

## The ideas

### 1. Spend a byte budget instead of a fixed threshold

Replace the stopping rule of the measure-and-patch loop. The loop itself stays as it is.

- **Budget per IDR period of the new stream:** β × (bytes of the pictures dropped in that period), with β around 0.5 as a starting point. Unspent budget could carry over to later periods.
- **Spend worst-first** by a visibility-weighted score, not raw max difference:
  - **Propagation:** an error in a reference picture lives until the next IDR. Weight it by the number of pictures decoded after it in the period, or by the actual downstream error measured in the previous round.
  - **Masking:** drift in flat or static areas is far more visible than in texture or fast motion. Weight it by the inverse of the local variance of the original MB, or by the proxy's static mask.
  - **Luma over chroma.**
- **Floor:** stop below a difference of about 3-4 levels whatever the budget. Below that, the kept twin is as faithful to the source as the original was; it's just a different encode.
- **Report** budget, spend and leftover per period, and the distribution of remaining differences, so the owner can review the trade-off in `review/index.html`.

With I_PCM as the only currency (about 400 bytes per MB), even the full budget only takes the threshold from 16 to roughly 10-12 on busy content. Lowering it to 8 would cost about 70 MB per 5 minutes, about 3× the room. Idea 2 is what makes a low floor affordable.

### 2. Cheaper patches: residual corrections from our own pre-deblock reconstruction

Keep the MB's mode and motion and add a correction to its residual instead of replacing it with PCM. For an inter MB at QP q:

```
target  = original pre-deblock reconstruction of the MB
pred'   = its prediction in the new stream (it reads K where it read D)
R'      = target - pred' = original residual + (orig_pred - pred')
levels' = quantise(forward_transform(R'), q)
```

`original residual` is exactly representable: it is the original coefficients, dequantised. So only the drift term `orig_pred - pred'` is quantised. The leftover error is a fraction of the quantiser step: about ±1-3 levels at the QPs seen here, compared with up to 15 today. A correction MB should cost tens of bytes, compared with about 400 for PCM.

**The key fact that makes this feasible: no deblocking filter is needed.**
- References are stored after deblocking, and ffmpeg already gives us those frames in the measure loop, for both the original and the new stream.
- Pre-deblock reconstruction of picture N therefore needs only:
  - inter prediction from the ffmpeg frames: luma 6-tap quarter-pel, chroma eighth-pel bilinear, default / explicit / implicit weighting;
  - intra prediction from the pre-deblock pixels of picture N's own neighbouring MBs, which we reconstruct ourselves (4x4, 8x8 with reference filtering, 16x16, chroma);
  - inverse transforms and dequantisation: 4x4, 8x8, Intra16x16 luma DC Hadamard, chroma 2x2 DC; flat or custom scaling lists; chroma QP from `chroma_qp_offset` (−3 on the Amazon file) and the QP table.
- The measure loop already handles whatever deblocking does afterwards: decode, compare, patch again.

**A cheap, exact validation trick for the reconstruction:**
- Rewrite one picture's slice header with `disable_deblocking_filter_idc = 1`. `explicit.rewrite_slice` already rebuilds headers bit by bit; extend it with that field.
- Decode that segment with ffmpeg. Picture N then comes out *pre-deblock*, while every picture before it is unchanged.
- Our reconstruction of N must match it bit for bit.
- Do this for hundreds of pictures on all four test files (all MB types, both encoders) before relying on it.

Pieces already there:
- `h264cabac.py` parses and re-encodes every syntax element, coefficients included (`mb['coef']`, `cbp`, `t8`, `qpd`, intra modes `ipred` / `imodes`, `cmode`, `i16pred`).
- `h264mv.derive_motion` gives per-4x4 reference indices, pictures and motion vectors, temporal and spatial direct included. These were validated bit-exact by re-expressing motion.
- `patch.py` keeps the rest of a picture exact around a modified MB: skips that must become explicit, mvd re-solving, the QP chain, and direct blocks.

Pieces missing:
- **Explicit weight tables.** `h264hdr.parse_slice` reads `pred_weight_table` but discards the values. Record `luma_log2_weight_denom`, `chroma_log2_weight_denom` and the per-reference weights and offsets. Implicit weights follow from POCs (8.4.2.3).
- **The reconstruction itself,** in numpy, as a new module `recon.py`.
- **A forward transform and quantiser.** Use plain rounding first; trellis or RDO only if the byte cost matters.
- **Putting a correction into the CABAC records:**
  - set `cbp` and coefficients;
  - a skip MB becomes an explicit MB with the same motion; `to_explicit` already does this;
  - choose the transform size: keep `t8` unless 4x4 fits better;
  - keep the MB's QP if at all possible, because QP also feeds the deblocking strength.

Caveats to measure, not assume:
- **Deblocking strength.** bS = 2 at an edge when either 4x4 block has non-zero coefficients. A correction that adds coefficients where there were none changes the filtering across that edge, a second-order error. The measure loop sees it, and the next round may correct it.
- **Intra MBs** in affected pictures, and MBs whose error is too large for one correction at q, fall back to PCM. A second option is a correction at a lower QP via `mb_qp_delta`, which also changes the deblocking QP average at its edges.
- **Corrections in reference pictures** stop propagation, just as PCM does today. Correct reference pictures first, in decode order.
- **Exactness:** this approach is near-exact, not exact. That is fine per the owner: noise-level differences are acceptable, and visible substitutes are not.

### 3. Optional: use corrections on the Canadian files too

Their hard drops leave 1-8 levels around PCM patches from deblocking across PCM edges. With residual corrections, many of those MBs need no PCM at all. Run this only after ideas 1 and 2 work on the Amazon file, and compare with the v1.2.0 outputs (`work/*_out.mkv` and the reports).

## Practicalities

- **Machine:** 32 threads and 31.7 GB of RAM, with a history of crashing under full load. Keep the defaults: half the workers, below-normal priority, and pools sized by `system.pool_size`. Explicit repair workers peaked at 1.1 GB (budget `REPAIR_PEAK_EXPLICIT` = 1.5 GB, so 9 workers). Run one heavy pipeline at a time.
- **Speed:** the profile of one period is dominated by the Python CABAC parse and encode (about 0.8 s per picture each way) and `derive_motion` (about 0.25 s). Frame diffs are cached across rounds, and pictures whose inputs didn't change are not re-encoded (`patch_pictures(cache=...)`). The reconstruction will add numpy work per affected MB only. Don't reconstruct whole pictures except for validation.
- **Test material:**
  - The bitstream tests extract short clips from the MKVs in the repository root.
  - `tests/test_explicit.py` covers explicit mode.
  - `work/mmco/clip.mkv` (5 minutes) is a good iteration target: the full pipeline with `--verify` takes about 6 minutes.
  - Work files from this session are in `work/mmco/`, including `t_norepair.py`, a visible-area error histogram between two Annex-B streams.
- **Owner's preferences** (see the memory notes too):
  - Bit-exact wherever possible.
  - Noise-level differences confined to frames that read a substitute are acceptable.
  - Visible substitutes are the worst outcome.
  - The owner reviews results by eye; give them before/after crops of the worst MBs.

## Full-length run (v1.2.0, `mmco-or-ref-list-modification.mkv`)

### Plan (same in both runs)

| | |
|---|---|
| input | 150,499 pictures at 29.97 fps, 2,511 IDR periods, 84 min |
| cadence | 23.976, score 9.60, 10 phase changes (8 at black runs, 2 at hard cuts) |
| duplicates | 30,100 |
| drops | 30,098 → 120,401 frames out |
| free reference drops (bit-exact) | 382 |
| hard pairs | 2,716 |
| anchor rebuilds (fallbacks) | 423 (0) |
| substitutes | 13 invisible, **0 visible** |
| unsupported pairs | 0 |
| headers rewritten | 60,319; 423 IDRs demoted |
| A/V | video − audio gap change +0.076 s. The two phase changes at hard cuts near the end have no black run after them to absorb the correction (an existing limitation). |

### First run: two defects, since fixed

- **Status:** ok, 66 min (repair 46 min on 9 workers, verify 16 min).
- **Verify:** 120,401 frames compared, 73,476 bit-exact. Nothing differs outside the hard-drop neighbourhoods and rebuilt leading pictures.
- **Repair:** 105,731 PCM MBs, +39.8 MB on a 5.9 GB stream (0.7%).
- **Defect 1, round cap:** 1,722 pictures stayed over the threshold, worst 169.
  - The worst ones had **no patches and were not capped**.
  - The loop patched one reference picture per round and stopped after 12 rounds.
  - Anchor rebuilds **chain**: the IDR every 60 frames falls on a duplicate for a whole phase segment, so up to 18 consecutive IDR periods merge into one new period of up to 864 pictures. In such a period, 12 rounds don't reach the end.
  - **Fix:** after `DRIFT_SEQUENTIAL_ROUNDS` = 6, every round patches everything still over the threshold.
- **Defect 2, memory:** one repair worker peaked at **7.2 GB** (budget 1.5 GB).
  - It held every original frame of a merged period, a second copy for the diff cache, and every parsed picture (25-45 MB each: records, bit list, motion).
  - **Fix:**
    - the original is decoded again in lockstep each round;
    - only target frames of patched pictures are kept, and diffs are cached by hash;
    - parsed pictures live in an LRU (`PARSED_CAP` = 12);
    - parsed records keep only the header bits;
    - motion is cached only for reference pictures;
    - `patch_pictures` checks its cache *before* parsing, which matters with an LRU.
- **Effect on period 891, which held the worst picture:** worst 169 → 17, 9 rounds (converged), PCM 441 → 1,041 MBs, 118 s → 267 s, worker memory growth about 100 MB.

### Second run (with both fixes)

| | |
|---|---|
| status | ok, 94 min (repair 79 min on 9 workers, verify 14 min) |
| output | 5.59 GB MKV (input 6.04 GB); 120,401 frames at 23.976 |
| verify | 73,476 bit-exact, 46,925 differ, **0 outside** hard-drop neighbourhoods and rebuilt leading pictures, 0 decoder errors |
| repair | 149,483 PCM MBs, +56.2 MB (1.0% of the stream; the drops freed about 7% — see the clip numbers above) |
| left over the threshold | 291 pictures, **all at the 2% PCM cap** (65 of them rebuilt leading pictures); visible worst 56 |
| distribution of their worst MB | 16-23: 188, 24-31: 50, 32-39: 27, 40-47: 21, 48-56: 5 |
| verify's worst (uncropped, so it includes the 8 hidden padding rows) | 103 |

The 291 capped pictures are where a byte budget (idea 1) would spend first. Raising the cap alone would clear them at roughly 400 bytes per MB.

**Memory after the fixes:**
- The second run's workers still peaked at 3.5 GB.
- In an 864-picture merged period, measured with full motion fields, that was:
  - about 99 cached motion fields at 9.3 MB each;
  - 12 parsed pictures at about 40 MB each;
  - full target frames for every patched picture (3 MB each, hundreds).
- Two more fixes followed:
  - cached co-located motion is stored compactly (`h264mv.CompactMotion`, numpy-backed, 2.35 MB, equivalence checked);
  - only the PCM samples of the MBs being patched are kept, not whole frames.
- The real culprit turned out to be a leak: `derive_motion` stored every motion field on the picture dict it was given (`pic['motion'] = m`, never read). `patch_pictures` passes the task's own picture dicts, so each processed picture kept 9 MB forever. Removing it fixed most of the growth.
- **Result**, measured in a spawned worker as the pipeline runs it:
  - largest merged period (2006, 864 pictures): **1.0 GB** (first run: 7.2 GB);
  - period 891: 0.72 GB;
  - repair results identical (same PCM counts and rounds).
- `sample.mkv` is still packet-identical to v1.1 and all tests pass.
- The output in `work/mmco/full_out.mkv` comes from the second run. The later changes only affect memory, so a re-run gives the same repair.

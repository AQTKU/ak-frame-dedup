# ak-frame-dedup: build brief

## Goal

Build a start-to-finish command-line tool. It removes the duplicate frames that streaming services insert when converting film-rate content (23.976 / 24 / 25 fps) to 29.97 or 30 fps. It works on the H.264 bitstream directly and never re-encodes. The output is a constant-frame-rate MKV at the source rate, with the audio, subtitles and chapters carried over.

These are not telecine/pulldown: whole frames are repeated, e.g. one in every 5 for 23.976 to 29.97. Re-encoding is off the table because these files are already low bitrate.

## Owner's hard requirements

1. **Detect the input rate from the H.264 bitstream** (the SPS VUI `timing_info`: `time_scale / (2 * num_units_in_tick)`), never from container timestamps. Some inputs have deliberately mangled PTS. Derive display order from picture order count (POC), also never from PTS.
2. **Auto-detect the source rate** from {23.976, 24, 25} for an input of 29.97 or 30. The full set of conversions to consider:
   - 23.976 to 29.97 and 24 to 30: ratio exactly 4/5, so a fixed 5-frame cadence.
   - 25 to 30: ratio exactly 5/6, so a fixed 6-frame cadence.
   - 25 to 29.97: ratio 25 × 1.001 / 30. A 6-frame cadence that slips by one position about every 1200 output frames. The owner calls this "the 1001 catching up".
   - 24 to 29.97 (ratio 0.8008) and 23.976 to 30 (ratio 0.7992): 5-frame cadences that slip about every 1250 frames.
   - Model: output frame n shows source frame `floor(n * r + phi)`, with r = S/T. Frame n is a duplicate when that does not advance from frame n-1, i.e. when `frac((n-1)*r + phi) < 1 - r`.
3. **The output must be constant frame rate (CFR), never VFR.** If the file opens with a native-29.97 logo (sample.mkv does), apply the content cadence straight through it and accept some glitching there. The owner trims such intros separately. Trust that the real content has one consistent rate.
4. **Keep A/V sync.** Output frame k gets timestamp k / fps_out. Where the cadence phase changes at a chunk boundary, the number of dropped frames can be off by plus or minus one. Correct it inside the black run at that boundary: drop an extra black frame, or keep one black duplicate. Keep the cumulative kept-frame count tracking `round(t * fps_out)` to within half a frame across the whole file.
5. **Fail out cleanly** if there is no duplicate cadence at all. `nodupes.mkv` is native 29.97 studio content and must be rejected with a clear message, writing no output.
6. **Patch the output stream's SPS VUI timing** to the output rate (e.g. num_units_in_tick 1001 / time_scale 48000 for 23.976; 1 / 50 for 25; 1 / 48 for 24). Players that demux the raw stream must see the right rate. The fields are fixed-width 32-bit, so patch them in place in the RBSP and then re-escape. Do this for every in-band SPS.

## Owner's detection design (agreed, with refinements)

1. **Score duplicates per frame.** Decode a low-res luma proxy (480x270 gray) of every frame in display order. The current prototype does this with `ffmpeg -f h264 -i <annexb> -fps_mode passthrough -vf scale=480:270:flags=area,format=gray -f rawvideo -`; a raw Annex-B decode comes out in POC order. Compute per transition i (frame i vs i-1) the mean absolute difference and the max over 16x16 blocks of the block-mean absolute difference. Take each relative to a local median over ±6 frames. Turn both into a log-likelihood ratio (see `cadence.py`: `rel_diff`, `train_llr`, `emission`, `mask_static`) and sum them.
   - Evidence is zeroed where the picture is static (local median block-max < 1.0). There the differences are compression noise that follows the encoder's 4-frame I-B-B-B-P rhythm and fakes a phase.
   - The LLR tables were trained on sample.mkv, using its 1ms-retimed pairs as labels. Ship the trained model as a JSON file in the repo rather than retraining at run time. The training script can stay as a dev tool.
2. **Chunk the video at black runs of 0.5 s or longer** (low mean luma and low max luma). Real cadence breaks sit at blacks where ad breaks would have been. In sample.mkv the breaks are at a logo fade and the end blacks; in smoke.mkv all three are at pure black.
3. **For every candidate ratio and every chunk, score every phase phi exactly.** Each frame's evidence contributes to an arc of phi values of length 1-r. A sweep over the circle gives the exact best phi and its score in O(L log L), so no grid search is needed.
4. **Pick one global source rate** (the best total), then **choose per-chunk phases with dynamic programming**. Continuing the previous chunk's phase is free; changing it costs a penalty. This is the owner's "brute force at each chunk" idea done exactly, without 2^K combinations.
   - Also allow a phase change at a hard cut inside a chunk, at a much higher penalty. Breaks don't always sit at blacks: sample.mkv's first break is at a logo fade.
5. **Fail-out test:** compare the best cadence's evidence per predicted duplicate against what a random phase scores. Calibrate the threshold on the four test files.

## Existing code in this repo (validated; read these before writing anything)

- `h264hdr.py`: SPS/PPS/slice-header parser, Annex-B NAL iterator, `escape`/`unescape`, `set_bits`, `parse_stream(path)` returns (spss, ppss, pics in decode order). It records `frame_num_bit` and `poc_lsb_bit` for in-place patching. It does not parse the VUI yet; you must add that (aspect ratio, overscan, video signal type, chroma location, timing info, NAL/VCL HRD parameters, pic_struct, bitstream restriction).
- `h264cabac.py`: CABAC slice-data codec. One code path decodes or encodes each syntax element. `Coder` is the arithmetic engine, including I_PCM. `SliceCodec.run(coder, mbs=None)` decodes when mbs is None, otherwise encodes the given macroblock records. Intra 4x4/8x8 modes are stored as real modes (`imodes`) and re-derived on encode.
  - **Validated:** decode followed by re-encode is byte-identical on 74+ frames (I/P/B, including hard-pair neighbourhoods).
  - Scope: High profile, 4:2:0, 8-bit, progressive, one slice per picture, CABAC. All test files are like this; assert on anything else.
- `h264mv.py`:
  - `dpb_walk(spss, pics)` attaches `poc`, `frame_num` and initial ref lists (decode-order ids) to every picture, using sliding-window marking only. The streams have no MMCO, no reordering commands and no long-term refs; this is asserted.
  - `derive_motion(pic, mbs, W, H, pics, col_motion, target=None)` gives per-4x4 refidx, ref picture id and mv, including P_Skip and temporal direct. In `target` mode it solves the mvds that reproduce a given motion field.
  - **Validated:** every inter MB in 23 frames was rewritten into explicit 8x8 form and decoded bit-identical to the original.
- `roundtrip_test.py`: `decode_picture` / `encode_picture` helpers and the round-trip test CLI.
- `verify_motion.py`: the re-expression test, plus `reexpress()`, which `patch.py` uses.
- `dedup.py`: drops duplicate twins from an Annex-B stream and renumbers frame_num on following pictures up to the next IDR. It takes pairs from the 1ms container timestamps; replace that with the new detector. It has `--keep-second` and `--steal-poc` experiment flags, both proven worse; keep them off or delete them.
- `sniff.py`: `analyse_window` lists which blocks read the dropped twin D and the kept twin K. Its taint-propagation part over-predicts badly; only the direct-reader maps (`reads_D`) are used.
- `patch.py`: `patch_stream(es, patches{dec: [(mbx,mby)]}, yuv_of(dec)->uncropped yuv420p bytes, out_path)`. It replaces macroblocks with I_PCM copies of the original decoded pixels and keeps everything else in the picture exact:
  - P_Skip MBs whose inferred motion would change become explicit.
  - mvds are re-solved.
  - The QP chain is preserved. A PCM MB passes QP_pred through, so a following MB that doesn't code a qp delta gets an empty coded chroma DC so it can carry one.
  - B pictures whose co-located P got PCM MBs have the affected direct blocks rewritten as explicit Bi 8x8.
- `refine.py`: an iterative border-growing patch experiment. **It does not converge**, because deblocking at I_PCM edges keeps differing from the original. Do not use it in the tool.
- `cadence.py`: the current 5-phase Viterbi detector. Replace it with the general design above, but reuse the evidence functions.
- `cabac_tables.json`: CABAC init / rangeTabLPS / transIdxLPS tables, extracted from ffmpeg and verified.
- `review/*.html`: local review pages (served by `.claude/launch.json` "review" via `python -m http.server`).

## Facts established about these streams

All four test files come from the same encoder:

- H.264 High@4.0, 1920x1080 (1088 coded), CABAC, one slice per picture.
- Fixed I-B-B-B-P mini-GOP. **B pictures are never references.** Temporal direct mode (`direct_spatial=0`) and implicit weighted bipred (`weighted_bipred_idc=2`).
- `max_num_ref_frames=4`, `log2_max_frame_num=7`, POC type 0 with `log2_max_poc_lsb=7`, `transform_8x8=1`.
- No SEI in the stream, although the VUI has pic_struct_present and NAL HRD.

Duplicate pairs (n-1, n) mostly contain a non-reference B. **Dropping a non-reference twin is bit-exact for every other frame**: verified on all 31,203 remaining frames of sample.mkv.

"Hard" pairs are those where both twins are reference pictures. **Always drop the later twin** and renumber frame_num up to the next IDR; default ref-list construction then points former readers of D at K. Dropping the earlier twin instead destroys the 3 B-frames that use it as their forward anchor (errors of 80-245 levels), and dropping an IDR loses about 20 frames.

Never drop an IDR. If a pair's later twin is an IDR, or any other case you cannot handle:
- keep both twins,
- log it,
- correct the frame count in the next black run (requirement 4).

After a hard drop, only 1-30 MBs per picture read D. That happens in the first P after the cut and the B pictures around it, and only while D stays in a picture's ref lists. Check that with `dpb_walk` lists, which are cheap and header-only, so you parse only those pictures and their co-located P.

Patching exactly those MBs as I_PCM with the original pixels did this on sample.mkv:
- worst luma difference from 20 down to 5,
- MBs differing by 3 or more from 1,413 down to 360,
- cost: 145 MBs, +55 KB.

The remaining differences of 1-5 levels are deblocking and intra-edge effects at the PCM borders. That is accepted.

Hard-pair counts with the current detector: sample 33, foliage 93 (all followed by a non-IDR I), smoke 658 (576 of them are an I followed by a P twin, plus 20 P/I and 8 I/I pairs).

**Speed:** pure Python CABAC runs at about 1 s per 1080p picture (parse plus encode). Low-res proxy decode of a 40k-frame file takes about 25 s. A full-res compare of two streams takes about 2.5 min per 38k frames. The machine has 32 cores, so use `multiprocessing` for per-picture parsing, sniffing and patching.

## Pipeline to build

Make it a Python package plus CLI, e.g. `python -m frame_dedup in.mkv -o out.mkv [--report r.json] [--verify] [--work DIR] [--keep-work]`.

1. **Extract the stream:** `ffmpeg -i in.mkv -map 0:v:0 -c copy -bsf:v h264_mp4toannexb -f h264 work/in.h264`. Parse headers. `dpb_walk`. Read the input rate from the VUI.
2. **Display order:** from POC within each IDR period, in decode order. Check that the proxy decode's frame count equals the picture count.
3. **Evidence and chunks:** compute the evidence; detect black runs; build the chunks.
4. **Detect the cadence:** global source rate plus per-chunk phase (DP). Fail out if there is no cadence. Produce the duplicate list, then apply A/V-sync corrections in blacks.
5. **Choose drops:** the non-reference twin when available. If both twins are non-reference, either is bit-exact; prefer the second. Otherwise drop the later twin, as above. Never an IDR.
6. **Write the deduplicated stream:** drop, renumber frame_num, and record hard drops.
7. **Repair hard drops:** sniff the direct readers of each dropped reference twin, in parallel. Decode the needed original frames uncropped (`-apply_cropping none`, yuv420p, 1920x1088) to get the patch pixels. Pull them from a segment decode of the original stream, from the previous IDR, labelled by POC; see `refine.decode_segment`. Then `patch_stream`.
8. **Patch the SPS VUI timing** to the output rate.
9. **Mux:** `mkvmerge -o out.mkv --default-duration 0:<fps>p work/out.h264 --no-video in.mkv`, or equivalent. Keep audio, subtitles, chapters and track flags. For 23.976, 24 and 25 use 24000/1001p, 24p and 25p. Check that the output's video duration matches the audio to within one frame.
10. **`--verify`:** decode the output and the original, match frames by POC, and report:
    - frames bit-exact,
    - frames differing,
    - worst difference,
    - MBs differing by 3 or more.

    Everything except hard-drop neighbourhoods must be bit-exact.
11. **Report JSON:** detected rates, chunks and phases, phase changes, number of drops, hard pairs, PCM MBs and bytes added, sync corrections, unsupported pairs, verify stats.

## Test files (in the repo root, gitignored)

| File | Expected |
|---|---|
| `foliage.mkv` (2.0 GB, 79,390 frames, raw 29.97, normal PTS) | **Primary target: run end to end.** Should detect 23.976 with a single phase for the whole episode (the current detector finds none). Recipe-card stills and animated hand-lettering are the hard parts. |
| `nodupes.mkv` (1.0 GB, native 29.97) | **Must fail out cleanly.** |
| `sample.mkv` (1.0 GB, 38,962 frames) | Regression: should detect 23.976. PTS are mangled (duplicates retimed 1 ms apart); you may use these as labels for checking only. |
| `smoke.mkv` (2.0 GB, 79,430 frames) | Optional, if time allows: should detect 23.976 with phase changes near 806 s, 1362 s and 2364 s. PTS are mangled (duplicates share identical PTS), and those labels are wrong in places, e.g. the steam train at about 3 s. |

## Environment

- Windows 11. The Bash tool is Git Bash; PowerShell is also available.
- Python 3.12 with numpy 2.5 and Pillow 12.3. **No C compiler, no numba. Do not pip-install anything.**
- ffmpeg, ffprobe, mkvmerge and mkvextract are on PATH.
- About 138 GB of disk is free. Delete large intermediates when done unless `--keep-work`.
- The repo is git-initialised. Commit in logical steps. End every commit message with:

  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`

## Working rules

- Verify everything by decoding and comparing against the original. That has been the discipline throughout, and every claimed result above was measured that way.
- Keep the existing validated modules working. If you change `h264cabac.py`, `h264mv.py` or `patch.py`, rerun `roundtrip_test.py` and `verify_motion.py` on a few frames.
- Add a small test suite: CABAC round-trip on a few frames, synthetic cadence sequences for all six conversions recovering the right ratio and phase, and the fail-out case.
- Write a README covering usage, how it works, and limitations.

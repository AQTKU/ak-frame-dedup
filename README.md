# ak-frame-dedup

Removes the duplicate frames that streaming services insert when they convert
film-rate content (23.976 / 24 / 25 fps) to 29.97 or 30 fps, by editing the
H.264 bitstream directly. Nothing is re-encoded. The output is a
constant-frame-rate MKV at the source rate, with the audio, subtitles,
chapters, tags and track flags of the input.

These are not telecine or pulldown: whole frames are repeated, for example
one in every five for 23.976 to 29.97.

## Usage

```
frame-dedup in.mkv -o out.mkv [--report r.json] [--verify]
            [--mb-report mb.json [--mb-threshold N]]
            [--work DIR] [--keep-work] [--workers N]
            [--no-rebuild] [--keep-idr-pairs] [--dry-run]
```

| option | meaning |
|---|---|
| `-o out.mkv` | output file (written as `out.mkv.partial.mkv`, renamed on success) |
| `--report r.json` | JSON report: rates, chunks and phases, drops by kind, hard / unsupported pairs, substitutes, anchor rebuilds and their fallbacks, repair cost, sync corrections, durations, verify stats, runtime and peak memory |
| `--verify` | decode the output and the original and compare every kept frame by (IDR period, POC) |
| `--mb-report mb.json` | with `--verify`: every differing frame (output and input frame number and time, IDR period, POC, type, whether it is a rebuilt leading picture or in a hard-drop neighbourhood, worst difference) and its macroblocks at or above the threshold as `[mbx, mby, max_diff, luma_diff, visible]`, worst first, for building a viewer |
| `--mb-threshold N` | macroblock threshold of `--mb-report` (default 8) |
| `--work DIR` | intermediates (default `work/<input name>`). Only the files the tool creates are deleted afterwards; `run.log` stays as a trail |
| `--keep-work` | keep the intermediates (cached proxy features are reused on the next run) |
| `--workers N` | worker threads (default: half the logical CPUs) |
| `--no-rebuild` | never rebuild an anchor (drop rule 6 below); IDR-later pairs then use rule 7 |
| `--keep-idr-pairs` | see drop rule 7 below |
| `--dry-run` | detect and plan only |

Exit codes:
- 0: success.
- 1: any other error (the report is still written).
- 2: no duplicate cadence; no output is written.
- 3: the input is outside the supported scope.
- 4: `--verify` found differences outside the hard-drop neighbourhoods.

Requirements:
- `ffmpeg`, `ffprobe` and `mkvmerge` on PATH;
- to build: Rust 1.97 or later (`cargo build --release` gives
  `target/release/frame-dedup`). The likelihood tables are compiled in.

Tests: `cargo test --release`. The oracle tests in `tests/` compare with
cached outputs of the Python v1.2.0 implementation under `work/port/oracle/`
and skip when those are missing.

Dev tools (`frame-dedup dev <command>`):
- `train-model sample.mkv` retrains the likelihood tables
  (`src/cadence_model.json`).
- `roundtrip in.h264` is the CABAC round trip.
- `reexpress in.h264 <P dec idx> <count>` is the motion re-expression test.
- `headers`, `records`, `motion`, `features`, `detect` and others dump
  intermediate results as JSON.
- `tools/e2e.ps1` runs the release build on the test files and compares
  per-stream MD5s with reference outputs.

The tool is a Rust port of the Python implementation (v1.2.0), with
byte-identical output: see "Rust port" below.

## How it works

1. **Extract** the video track to Annex-B
   (`ffmpeg -c copy -bsf:v h264_mp4toannexb`).
   - Parse SPS (including the full VUI), PPS and slice headers
     (`h264hdr.rs`).
   - The **input rate** comes from the SPS VUI:
     `time_scale / (2 * num_units_in_tick)`.
   - **Display order** comes from POC within each IDR period. Container
     timestamps are never used.
2. **Evidence** (`evidence.rs`).
   - A 480x270 gray proxy of every frame is decoded from the Annex-B stream;
     a raw decode comes out in POC order. The proxy frame count is checked
     against the picture count.
   - Each transition gets two features: the mean absolute difference and
     the max 16x15-block mean difference.
   - Both are taken relative to a ±6-frame local median and turned into
     log-likelihood ratios, then summed. The ratios are trained on
     sample.mkv's 1 ms labels and shipped as
     `src/cadence_model.json`.
   - Evidence is zeroed where the picture is static, because there it
     follows the encoder's GOP rhythm.
3. **Chunks**: the video is split at black runs of 0.5 s or more (proxy
   mean < 20 and max < 40) and at hard cuts.
4. **Cadence** (`cadence.rs`).
   - Model: frame n shows source frame `floor(n*r + phi)`, with r = S/T.
     Frame n repeats frame n-1 iff `frac((n-1)r + phi) < 1 - r`.
   - With r = p/q in lowest terms, every arc endpoint is a multiple of 1/q,
     so phi only matters through `m = floor(phi*q)`. The score of every
     phase of a segment is an exact circular window sum over a q-bin
     histogram, in O(L + q). There is no grid search.
   - For each candidate ratio, dynamic programming chooses the segment
     phases. Continuing the phase is free; changing it costs 25 nats at a
     black run and 150 nats at a hard cut.
   - The best total picks **one global source rate**. The candidates are:

     | source → input | ratio | cadence |
     |---|---|---|
     | 23.976 → 29.97 and 24 → 30 | 4/5 | fixed 5-frame |
     | 25 → 30 | 5/6 | fixed 6-frame |
     | 25 → 29.97 | 1001/1200 | 6-frame, slips about every 1200 frames |
     | 24 → 29.97 | 1001/1250 | 5-frame, slips about every 1250 frames |
     | 23.976 → 30 | 800/1001 | 5-frame, slips about every 1250 frames |
5. **Fail-out**.
   - Score = (mean evidence of the predicted duplicates) − (mean evidence
     of all frames), both over non-static frames. The second term is what a
     random phase scores.
   - The tool fails out if the score is below 2.0, if fewer than 200
     predicted duplicates carry evidence, or if the best total is not
     positive.
   - Observed scores:

     | file | score |
     |---|---|
     | sample | 9.60 |
     | foliage | 10.20 |
     | smoke | 9.84 |
     | nodupes | 0.04 |
6. **Drops** (`plan.rs`). For a duplicate pair (n-1, n), the first rule
   that applies wins:
   1. drop the later twin if it is a non-reference picture;
   2. otherwise drop the earlier twin if that is non-reference;
   3. otherwise drop a reference twin whose removal changes no other
      picture's reference lists (*unreferenced*, e.g. a P picture that only
      an IDR follows), preferring the later twin. This is checked header-only
      per IDR period;
   4. otherwise drop a cheap picture (non-reference or unreferenced) within
      ±3 frames that is a visually lossless repeat of its predecessor (black
      or static: proxy MAD < 0.25 and block max < 1.0). This is an
      *invisible substitute*;
   5. otherwise drop the later twin as a **hard drop**, repaired in step 9,
      unless it is an IDR (an IDR is never dropped);
   6. if the later twin D is an IDR, drop the earlier twin K by an **anchor
      rebuild** (step 10): D is demoted to a non-IDR I picture that takes
      K's place. This needs, header-only, K to be a non-IDR reference
      picture ending the previous IDR period, only non-reference pictures
      between K and D in decode order, nothing else reading K, and the
      rewritten stream's reference lists to pass the check in step 10.
      `--no-rebuild` turns this rule off;
   7. if the rebuild is not possible, drop the cheap picture within ±3
      frames with the least motion instead. This *visible substitute* loses
      one real frame but keeps A/V sync. `--keep-idr-pairs` keeps both twins
      instead, as the brief specified;
   8. if there is no candidate, keep both twins (*unsupported*) and correct
      the count at the next black run.

   Rules 1-4 leave every other frame bit-exact. Rule 6 changes only the
   non-reference B pictures that used K as their forward anchor.
7. **A/V sync**.
   - Kept frame n becomes output frame k at k/fps_out; its offset is
     k − n·r output frames.
   - At every black run, one extra black frame is dropped (a non-reference
     one) or one black duplicate is kept, so that the mean offset of the
     stretch after the run is within ±0.5 frame.
   - Phase changes at hard cuts, and unsupported pairs, are corrected at
     the next black run.
8. **Write** (`stream.write_stream`).
   - Pictures are dropped and frame_num is renumbered up to the next IDR.
   - Every in-band SPS gets its VUI timing patched in place:
     num_units_in_tick / time_scale = 1001/48000 (23.976), 1/48 (24) or
     1/50 (25). The fields are fixed-width u(32), so the RBSP is patched and
     then re-escaped.
9. **Repair hard drops** (`repair.rs` + `patch.rs`).
   - The default reference lists of the deduplicated stream are recomputed
     from headers alone. Pictures whose lists change are found, plus B
     pictures whose co-located picture is one of them.
   - Only those pictures are parsed with the CABAC codec, and their motion
     is derived twice from the same macroblock records: in the original
     reference context and in the new one.
   - A 4x4 block whose reference picture or motion vector differs is left
     alone when its prediction is provably unchanged. That needs equal mvs,
     original decoded pixels of the old and new reference identical over the
     block's footprint, and for bi-prediction identical L0 and L1 inputs,
     because implicit weights follow the POCs.
   - Every other macroblock with such a block is replaced by an I_PCM copy
     of the original decoded pixels. The rest of each picture is preserved
     exactly: skips become explicit where needed, mvds are re-solved, the QP
     chain is kept, and direct blocks become explicit bi-prediction.
   - Work runs in parallel, one task per IDR period.
10. **Anchor rebuilds** (`rebuild.rs`), a third pass over the repaired
    stream. For a pair whose earlier twin K (a P) is the forward anchor of
    the B pictures before it and whose later twin D is an IDR:

    ```
    display:  P0  B1 B2 B3  K | D(IDR)  b b b  P ...
    decode:   P0 ... K  B1 B2 B3  D  P  b b b ...
    ```

    - **Demote D.** `nal_unit_type` 5 becomes 1 (`nal_ref_idc` kept), the
      slice header loses `idr_pic_id`, and the IDR form of
      `dec_ref_pic_marking` becomes `adaptive_ref_pic_marking_mode_flag = 0`.
      The header is rebuilt bit by bit and padded with
      `cabac_alignment_one_bit`s, and the original CABAC slice-data bytes are
      appended unchanged (they depend only on slice QP and slice type). The
      result, I2, decodes bit-identical to D.
    - **Move it into K's slot.** K is dropped and I2 is written in K's
      decode position, behind D's SPS / PPS and a recovery point SEI. I2
      carries K's `frame_num` and K's POC, so B1-B3 find their anchor at the
      same list positions and POC distances.
    - **Merge the IDR periods.** Every POC of D's former period is shifted
      by POC(K) − POC(D), and `frame_num` continues from K's (it is
      re-derived as PrevRefFrameNum + 1 for every picture; the inputs have no
      `frame_num` gaps). POC differences inside the period are unchanged, so
      its direct modes and implicit weights are too. Both fields are
      fixed-width and patched in place. Chains of consecutive rebuilds merge
      several periods.
    - **List check.** The decoded picture buffer is no longer flushed at D,
      so older pictures linger in it. The reference lists of the rewritten
      stream are simulated from headers (`dpb_walk`): every entry the
      original could use must still point at the same picture. Lingering
      pictures only ever extend a list beyond the entries the original had.
      The check runs per pair while planning (a failure falls back to rule
      7), again over the whole stream before writing, and the written stream
      is re-parsed and compared with the plan.
    - **Repair B1-B3.** They are non-reference pictures, so nothing they
      contain propagates. Their temporal-direct blocks took motion from K's
      motion field; I2 is all intra, so every direct block whose motion would
      change is rewritten as explicit prediction with the original motion,
      using the coarsest partitioning that fits (16x16, 16x8, 8x16, else
      8x8), and the mvds are re-solved. Their reads of K now read I2, a second
      encode of the same frame. The pair's segment is decoded and every
      macroblock that differs from the original by 16 levels or more
      (`PCM_THRESHOLD`, max over Y/U/V) becomes I_PCM, worst first, repeating
      until no new macroblock crosses the threshold. At most 2% of a picture
      (`PCM_MAX_FRACTION`, 163 MBs at 1080p) is patched, so fast-motion pairs,
      where the IDR twin is coded very differently from K, do not turn into
      whole PCM pictures. Smaller differences are encode noise and are
      accepted.
    - **Seeking.** I2 is no longer an IDR. A recovery point SEI in front of
      it (`recovery_frame_cnt = 0`, `exact_match_flag = 1`,
      `broken_link_flag = 0`) keeps it a random-access point, i.e. an open
      GOP. B1-B3 are its leading pictures and also read P0, so they cannot
      be decoded after a seek straight to I2; decoders skip them.
    - Work runs in parallel, one task per pair.
11. **Mux** with mkvmerge:
    `--default-duration 0:24000/1001p|24p|25p out.h264 --no-video in.mkv`,
    plus the original video track's language, name, flags and display
    dimensions, and the segment title.
12. **Duration check**: the output video's duration must match the input
    video's within one output frame, so the input's own video-to-audio gap
    is preserved. Both gaps are reported.
13. **Verify**.
    - Original and output are decoded uncropped at full resolution. Every
      output picture is matched with its original by the original's (IDR
      period, POC); the pipeline knows which original each output picture
      came from, including I2 (it is D) and the shifted POCs of merged
      periods. Decoding is split into groups of output IDR periods, each
      covering whole original periods.
    - It reports bit-exact frames, differing frames, the worst difference
      and MBs differing by 3 or more, over Y, U and V and also luma only.
    - Everything outside hard-drop neighbourhoods and the leading B
      pictures of rebuilt pairs must be bit-exact. A hard-drop neighbourhood
      runs from the dropped twin to the end of its IDR period. The leading B
      pictures are reported separately (`verify.rebuild_leading_b`).

### Explicit mode (MMCO / reference list modification)

Some encoders, such as x264 with B-pyramid and weighted P prediction
(mmco-or-ref-list-modification.mkv, an Amazon encode), manage references
explicitly:
- reference B pictures unmark older pictures with MMCO 1;
- every P picture reorders its list with ref_pic_list_modification, and
  lists the same picture two or three times so that each entry can carry
  different explicit weights.

Renumbering frame_num after a dropped reference breaks such a stream. The
picNum differences in later headers count the dropped picture, and its own
MMCOs never run. When the input uses either feature anywhere, the tool
switches to explicit mode (`explicit.rs`):

- **Header planning.** For every kept picture, the decoder must see exactly
  its original reference lists, entry by entry, and after marking exactly
  its original set of short-term references. A dropped twin D is replaced
  in both by its substitute σ(D), the other twin K. The planner simulates
  the DPB of the new stream and writes:
  - ref_pic_list_modification commands: the original ones if they still
    produce the list, none if the default list already does, else the
    shortest explicit prefix;
  - MMCO 1 operations: sliding window if it gives the target set, else
    explicit unmarking.

  frame_num is renumbered without gaps. The plan is re-simulated with
  `dpb_walk`, and the written stream is re-parsed and compared with it.
  Reference indices keep their meaning, so explicit weight tables, spatial
  direct reference indices and every block that does not read D stay
  valid. Planning an unmodified stream reproduces every header bit for bit.
- **Header rewriting.** Slice headers are re-serialised bit by bit:
  frame_num, POC lsb, ref_pic_list_modification, dec_ref_pic_marking, and
  IDR demotion. The CABAC slice data is byte-aligned and is appended
  unchanged.
- **Drop rules.** K must be decoded before D, so the twin later in decode
  order is dropped. It is *free* (bit-exact) when no list names it. The
  earlier twin is dropped instead only if nothing reads it. When the later
  twin is an IDR, the anchor rebuild is planned the same way: D is demoted
  into K's slot as I2, its period merges into K's with shifted POCs, and
  explicit MMCOs unmark K's period at the first reference picture after
  the leading pictures. Leading reference B pictures are fine here, unlike
  in `rebuild.rs`.
- **Spatial direct.** `derive_motion` derives spatial direct (MinPositive
  reference indices, colZero). Re-expressing every inter MB of 30 pictures
  as explicit motion decodes bit-identically. B pictures whose co-located
  picture became K are re-encoded with their original motion (direct
  blocks made explicit).
- **Measured repair.** This encoder codes the duplicate twin with its own
  residual, so D and K differ by encode noise almost everywhere (luma MAD
  0.5-1.2). Exact repair would PCM nearly every block that reads D: 78,020
  MBs (+30 MB) for 12 drops. Instead, each IDR period of the new stream that
  reads a substitute is decoded and compared with the original over the
  visible (cropped) area. Macroblocks off by `DRIFT_THRESHOLD` (16) or more
  become I_PCM copies of the original, in decode order up to the first
  reference picture, since that also fixes what is predicted from it. The
  period is then decoded again, until nothing crosses the threshold. At most
  `DRIFT_MAX_FRACTION` (2%, 163 MBs at 1080p) of a picture is patched.
  After `DRIFT_SEQUENTIAL_ROUNDS` (6) rounds, every picture still over the
  threshold is patched in the same round. This matters because anchor
  rebuilds chain: with an IDR every 60 frames, a whole phase segment of
  IDRs falls on duplicates, and up to 18 periods merge into one.
- **Verify** treats every picture of D's IDR period decoded after D as its
  neighbourhood (here D can be the display-earlier twin).

### Modules

| module | role |
|---|---|
| `h264hdr.rs`, `bits.rs` | Annex-B NAL iterator, SPS (incl. full VUI) / PPS / slice header parser, escape / unescape, bit reader / writer, in-place SPS timing patch |
| `cabac.rs`, `records.rs`, `picture.rs` | CABAC slice-data codec (decode and encode through one code path, I_PCM included); macroblock records; byte-exact round trip |
| `mv.rs` | `dpb_walk` (POC, sliding-window ref lists) and `derive_motion` (P_Skip, temporal and spatial direct, mvd solving) |
| `reexpress.rs` | explicit re-expression of skip / direct macroblocks; motion re-expression test |
| `patch.rs` | I_PCM macroblock patcher that keeps the rest of the picture exact (`patch_pictures`) |
| `rebuild.rs` | anchor rebuild: IDR demotion, period merging, list check, B1-B3 repair, recovery point SEI |
| `explicit.rs` | explicit mode: header planning (ref_pic_list_modification + MMCO 1), slice header rewriting, explicit anchor rebuilds |
| `evidence.rs`, `cadence.rs`, `plan.rs` | detection and drop planning |
| `stream.rs`, `repair.rs`, `decode.rs`, `verify.rs`, `mux.rs`, `pipeline.rs`, `main.rs` | the pipeline |
| `system.rs` | priority, RAM-aware pool sizing, peak-memory probe, logging |
| `dev.rs` | `frame-dedup dev` commands |

### Resource use

The tool is throttled by default. A full-load run with 32 workers and 64
ffmpeg decoders once crashed a machine with marginal RAM.
- `--workers` defaults to half the logical CPUs (16 on a 32-thread
  machine). Workers are threads of one process; they share the
  memory-mapped streams and the parsed headers.
- Memory-heavy pools are sized so that workers × budget stays under 40% of
  physical RAM, with the budgets measured for the Python implementation:
  0.8 GB per repair worker, 1.5 GB per explicit-mode repair worker (9
  workers on 31.7 GB) and 0.5 GB per verify worker. The threads need much
  less; the process peak on the full Amazon file was 6.7 GB, including the
  pages of the two 6 GB streams it maps.
- An explicit-mode repair task holds at most `PARSED_CAP` parsed pictures
  and the co-located motion fields of reference pictures. It decodes the
  original again each round instead of keeping it, and it keeps only the
  PCM samples of the MBs it patches.
- Each worker runs at most one full-resolution ffmpeg decoder at a time
  (explicit repair: the new and original decodes in lockstep), with 2
  threads. Verify hashes the original, then compares the output, then
  re-decodes only for frames that differ. It uses half the workers,
  because each keeps a decoder busy.
- The proxy decode is a single ffmpeg with up to 8 threads.
- The process and every ffmpeg / ffprobe / mkvmerge child run at
  below-normal priority (`SetPriorityClass`, `BELOW_NORMAL_PRIORITY_CLASS`).
- Logging is unbuffered and also goes to `<work>/run.log`.

## Rust port

Version 2.0 is a port of the Python implementation (v1.2.0) to Rust, so
that Python is no longer needed. Every module was checked against the
Python one on real data, then the whole tool end to end:
- Output MKVs are identical per stream (video, audio, subtitles; MD5) to
  v1.2.0's on sample, foliage, smoke, the 5-minute Amazon clip and the full
  Amazon file, and nodupes fails out identically. The reports are equal
  except for timings, worker counts, memory figures and the tool version.
- Intermediate checks included: every header of the full Amazon stream;
  macroblock records and motion fields of every picture of four clips; the
  stage-1 stream of the full Amazon run byte for byte; 119 recorded
  `patch_pictures` calls; the repaired and rebuilt streams of all four test
  files byte for byte, with 1 and 4 workers.
- The proxy features match numpy bit for bit, including its float32
  pairwise summation order, so detection cannot drift.

Wall-clock times with `--verify`, fresh work directories (proxy decode
included unless noted):

| file | Python v1.2.0 | Rust 2.0 |
|---|---|---|
| sample (8 / 4 workers) | 189 s | 113 s |
| 5-minute Amazon clip (8 / 4 workers) | 240 s | 93 s |
| foliage (8 workers) | 737 s | 208 s |
| smoke (8 workers) | 829 s | 339 s |
| full Amazon file (16 workers) | 94 min (proxy features cached) | 31 min |
| full Amazon: repair stage | 79 min | 20 min |
| full Amazon: verify stage | 14 min | 8 min |

The CABAC codec runs about 40× faster than the Python and motion
derivation about 70× faster. What remains is mostly ffmpeg: the measured
repair and the anchor rebuild decode every affected IDR period several
times, and verify decodes everything twice.
## Results

Measured on the four test files on a 32-thread / 31.7 GB machine with
default settings:
- Every run used `--verify`.
- Times are wall-clock. The sample run reused cached proxy features, which
  cost about 43 s fresh; the smoke run includes its 100 s proxy decode.
- foliage and sample have no IDR-later pairs, so the anchor rebuild never
  runs on them: their outputs are packet-for-packet identical to version
  1.0's (per-stream MD5 over video, audio and subtitles).
- Differences are max absolute over Y/U/V, with luma-only in brackets.

| | foliage.mkv | sample.mkv | smoke.mkv | nodupes.mkv |
|---|---|---|---|---|
| input rate (SPS VUI) | 29.97 | 29.97 | 29.97 | 29.97 |
| detected source / output rate | 23.976 | 23.976 | 23.976 | none: exit 2, score 0.04 |
| frames in → out | 79,390 → 63,512 | 38,962 → 31,169 | 79,430 → 63,544 | 38,902, no output |
| chunks / phase changes | 5 / 0 | 4 / 0 | 9 / 3 (805.9 s, 1361.7 s, 2363.5 s, all at blacks) | |
| drops | 15,878 | 7,793 | 15,886 | |
| hard pairs (repaired) | 93 | 26 | 53 | |
| unreferenced reference twins dropped | 0 | 1 | 302 | |
| substitutes (invisible / visible) | 0 / 0 | 0 / 0 | 2 / 0 (1.0: 2 / 300) | |
| anchor rebuilds (fallbacks) | 0 | 0 | 300 (0) | |
| unsupported pairs | 0 | 0 | 0 | |
| sync corrections | 0 | 0 | 3 | |
| hard-drop repair: PCM MBs / bytes added | 84 / +31.7 KB | 140 / +53.1 KB | 177 / +66.3 KB | |
| rebuild: leading B pictures / PCM MBs | | | 592 / 19,205 (40 pictures at the 2% cap) | |
| rebuild: bytes added (output size vs 1.0) | | | +12.3 MB (+0.58 MB: K is a P, the old substitutes were Bs) | |
| verify: bit-exact / differing frames | 63,286 / 226 | 30,920 / 249 | 62,707 / 837 | |
| hard-drop neighbourhoods: differing frames, worst difference | 226, 7 (luma 6) | 249, 5 (luma 5) | 245, 8 (luma 8) | |
| hard-drop neighbourhoods: MBs differing by 3 or more | 465 (luma 268) | 631 (luma 352) | 270 (luma 131) | |
| rebuilt leading B pictures: differing, worst difference | | | 592 of 592; worst 56, median per picture 15; 552 below 16 | |
| rebuilt leading B pictures: MBs differing by 3 or more | | | 1,805,319 (median 2,791 of 8,160 per picture: encode noise) | |
| differing anywhere else | 0 | 0 | 0 | |
| video − audio duration, in → out | +0.019 → +0.020 s | −0.088 → −0.089 s | +0.052 → +0.011 s | |
| A/V mean offset / max offset | −0.35 / 0.8 frames | −0.35 / 0.8 frames | +0.24 / 1.4 frames | |
| runtime | 349 s | 172 s | 579 s (rebuild 180 s, verify 267 s) | 50 s |
| peak RSS: main / worker | 2.1 GB / 0.25 GB | 1.0 GB / 0.26 GB | 2.4 GB / 0.25 GB | 1.0 GB |

System load during these runs: CPU peaked at 80-84% (mean 51-68%), and
available RAM never dropped below 13.3 GB of 31.7 GB.

### mmco-or-ref-list-modification.mkv (explicit mode, v1.2.0)

An 84-minute Amazon x264 encode: B-pyramid with MMCO, ref list modification
and weighted P prediction, spatial direct, and an IDR every 60 frames. It
was run with `--verify` and default settings.

| | |
|---|---|
| input rate / detected source rate | 29.97 / 23.976, score 9.60 |
| frames in → out | 150,499 → 120,401 |
| chunks / phase changes | 9 / 10 (8 at black runs, 2 at hard cuts) |
| drops | 30,098, of which 88% are non-reference B pictures (bit-exact) |
| unreferenced reference twins dropped (bit-exact) | 382 |
| hard pairs (measured repair) | 2,716 |
| anchor rebuilds (fallbacks) | 423 (0). They chain: up to 18 IDR periods merge into one. |
| substitutes (invisible / visible) | 13 / 0 |
| slice headers rewritten | 60,319 |
| repair | 149,483 PCM MBs, +56.2 MB (1.0%) |
| pictures left over the drift threshold | 291, all at the 2% PCM cap; visible worst 56 |
| verify: bit-exact / differing frames | 73,476 / 46,925. The differing frames are noise-level drift, all inside hard-drop neighbourhoods and rebuilt leading pictures. |
| differing anywhere else / decoder errors | 0 / 0 |
| output size | 5.59 GB (input 6.04 GB) |
| video − audio duration change | +0.076 s. Two phase changes at hard cuts in the last 3 minutes have no black run after them. |
| runtime | 94 min (repair 79 min on 9 workers, verify 14 min) |

On a 5-minute clip of the same file (154 hard pairs), the frames that
differ have a median frame MAD of 0.11. Their median worst pixel is 9, and
the 99.9th-percentile pixel error has a median of 3. Only 48 MBs are off by
32 or more.

Seeking on smoke.mkv's output: it has 883 keyframes, one per original IDR
period (583 IDRs and 300 rebuilt I2s; version 1.0 also had 883, and
mkvmerge does not flag plain non-IDR I pictures, so the recovery point SEI is
what makes I2 a keyframe). At five rebuilt pairs spread over the file
(24 s to 2360 s), `ffmpeg -ss` straight to I2 returns frames identical to a
continuous decode from the previous real IDR, and a seek 60 ms earlier, into
the leading B pictures, lands on the previous keyframe and is exact too. A
raw decode starting at I2's byte offset also reproduces every frame from I2
on (the Python v1.2.0 test suite).

Raw-stream check: the output's SPS signals 1001/48000, and ffprobe reports
`tbr 47.95` for it, just as it reports 59.94 for the 29.97 input. mkvmerge,
given only the raw output stream with no `--default-duration`, infers
23.976 fps.

## Changes from the build brief, and why

- **Sniffing covers more than reads of D.** P slices use two active
  references, so dropping D changes `[D, K]` into `[K, older]`. A block that
  used index 1 now reads a different picture, not only blocks that read D.
  Direct blocks and implicit bi-prediction weights also change with the
  reference POCs. The tool therefore compares motion derived in the old and
  new reference contexts for every affected picture; header-only lists pick
  those pictures. `sniff.analyse_window` and its taint prediction are not
  used.
- **IDR on the later twin (smoke.mkv).** In smoke.mkv's phase-4 chunks the
  encoder's IDR, every 90 frames, falls exactly on the later twin of a
  duplicate, 602 times.
  - The brief's rule (keep both, fix the count at the next black run)
    drifted the video **24 s** against the audio: there are only 8 black
    runs to absorb it.
  - In about half of these pairs the earlier twin is a P that nothing reads,
    because the IDR flushes the DPB. Rule 3 drops it bit-exactly.
  - In the other half the P is the forward anchor of the B-frames displayed
    before it. Version 1.0 dropped the least-motion nearby B-frame instead
    (a *visible substitute*, 300 on smoke.mkv). Each one caused two
    artefacts, a skipped frame and a frame held too long, i.e. more judder.
  - The anchor rebuild (rule 6, step 10) now drops K itself, the actual
    duplicate. An earlier objection to demoting the IDR was that it would
    remove random-access points and that every reader of the IDR would need
    patching. Neither holds: a recovery point SEI keeps I2 a random-access
    point (mkvmerge flags it as a keyframe, and decoding from it is exact),
    and nothing reads I2 differently from how it read D. Only the leading B
    pictures that read K change, and they are not references.
- **Pixel-aware sniffing and invisible substitutes.** A predicted pair
  (P, I) at the start of a black run, where the phase changes, is not a
  real duplicate. A pure ref-change test there PCM-patched four entire
  all-skip black pictures (+12.7 MB). Rule 4 now resolves such a pair by
  dropping a black frame. The pixel test also trims a few MBs elsewhere,
  for example 147 → 140 on sample.mkv.
- **sample.mkv has no phase break.** The evidence puts sample.mkv in one
  phase from the first content frame to the end, including the end credits,
  where the file's 1 ms labels are offset. The native 29.97 logo (frames
  0-66) is run through the content cadence, as requirement 3 asks. That is
  why the tool counts 26 hard pairs (plus one unreferenced twin) on sample
  where the label-based prototype counted 33.
- **What A/V sync can guarantee.** Inside a constant-phase stretch, the
  offset k − n·r cycles over a window about 0.6-0.8 frame wide. "Within
  half a frame" is therefore enforced on the mean offset of each stretch;
  the pointwise maximum (0.8 frame, 33 ms; 1.4 frames briefly around smoke's
  substitutes) is reported.
- **Duration check.** The inputs already end their video up to 2 frames
  away from their audio. The check therefore requires the video duration to
  match the input's within one frame, so the gap is preserved, rather than
  matching the audio absolutely. Both numbers are in the report.
- **Black threshold.** The gray proxy comes out of swscale at full range,
  so black reads 0-2. A black frame is proxy mean < 20 and max < 40.
- **Verify** counts luma and chroma differences, and also reports
  luma-only figures. The luma-only figures match the prototype's
  measurements (sample: worst 5, 352 MBs ≥ 3, against the prototype's 5
  and 360).
- **Resource defaults** (see above) were added after a crash under full
  load.
- `refine.py`, `dedup.py` and `drop_test.py` were removed; they remain in
  git history.

## Limitations

- Scope: High/Main profile, 4:2:0, 8-bit, progressive, CABAC, one slice
  per picture, POC type 0, one SPS id, no long-term references. MMCO 1 and
  short-term ref-list modification are supported (explicit mode); other
  MMCO operations are not. Anything else exits with code 3. Only 29.97 and
  30 fps inputs are processed.
- Explicit mode repairs by measurement, not exactly: frames that read a
  dropped twin keep encode-noise-level differences (below
  `DRIFT_THRESHOLD`, or above it where the per-picture PCM budget ran out)
  until the next IDR. See "Explicit mode" above.
- Hard drops leave differences of 1-8 levels around the I_PCM patches.
  They come from deblocking across PCM edges, and from intra prediction and
  later references that read post-deblock PCM pixels. They cannot be
  removed without the tool's own pre-deblock reconstruction.
- An anchor rebuild changes the B pictures displayed just before the pair:
  they now read I2, a second encode of K's frame. Differences are
  encode-noise level, except in fast motion, where the IDR twin can be coded
  very differently from K. There the per-picture PCM budget (2% of the MBs)
  leaves some MBs at up to about 40 levels (see Results).
- After a rebuild, I2 is an open-GOP random-access point rather than an
  IDR. A seek to it skips its leading B pictures, as with any open GOP.
  A player that seeks only to IDRs lands on the previous real IDR.
- A visible substitute (rule 7) is only used when a rebuild is not possible.
  It drops one real frame next to a duplicate that had to be kept: a
  one-frame stutter. `--keep-idr-pairs` keeps both twins instead and accepts
  A/V drift.
- A file with no black run of 0.5 s or more cannot correct the sync offset
  after a mid-chunk phase change or an unsupported pair. The offset is
  reported.
- Native-rate intros (logos) are forced through the content cadence, which
  can drop real frames there. This is by design: the output is always CFR.
- The likelihood tables were trained on one file from this encoder. Other
  encoders may need retraining with `frame-dedup dev train-model` on a labelled
  file.

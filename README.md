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
            [--no-corrections] [--drift-threshold N] [--pcm-threshold N]
            [--drift-budget F]
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
| `--no-corrections` | repair drift with I_PCM only, as 2.0 did (see "Residual corrections") |
| `--drift-threshold N` | measured repair: correct macroblocks off by N or more (default 6; 16 with `--no-corrections`) |
| `--pcm-threshold N` | measured repair: I_PCM for macroblocks off by N or more that no correction fixes (default 8); below it they are left |
| `--drift-budget F` | measured repair: may add up to F times the bytes dropped (default 1; 0: no budget). Each IDR period starts with F times its own dropped bytes; periods that run out are repaired again with a share of what the others left; where even that runs out, repair goes on only from `--pcm-threshold` |

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
and skip when those are missing. The Python implementation and its oracle
tooling were removed in 2.0.0; they remain in git history (commit
`571a346`).

Dev tools (`frame-dedup dev <command>`):
- `train-model sample.mkv` retrains the likelihood tables
  (`src/cadence_model.json`).
- `roundtrip in.h264` is the CABAC round trip.
- `recon in.h264 [--periods A-B]` checks the inter reconstruction of
  `recon.rs` against ffmpeg decodes without deblocking (see "Residual
  corrections").
- `reexpress in.h264 <P dec idx> <count>` is the motion re-expression test.
- `headers`, `records`, `motion`, `features`, `detect` and others dump
  intermediate results as JSON.
- `tools/e2e.ps1` runs the release build on the test files and compares
  per-stream MD5s with reference outputs.

The tool is a Rust port of the Python implementation (v1.2.0): 2.0 had
byte-identical output (see "Rust port" below). 2.1 repairs drift with
residual corrections (see "Residual corrections"); `--no-corrections` gives
2.0's output. 2.2 pools the byte budget across IDR periods, repairs the
rebuilt leading B pictures from 8 levels, and makes the measured repair
faster.

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
      the count at the next sync point (step 7).

   Rules 1-4 leave every other frame bit-exact. Rule 6 changes only the
   non-reference B pictures that used K as their forward anchor.
7. **A/V sync**.
   - Kept frame n becomes output frame k at k/fps_out; its offset is
     k − n·r output frames.
   - Inside a stretch of constant phase the offset cycles around a constant
     mean. Phase changes at hard cuts and unsupported pairs move that mean
     (phase changes at black runs are corrected inside the run).
   - The mean is pulled back within ±0.5 frame at the first **sync point**
     after it moves, where one frame more or less goes unseen:
     - a **black run**: one extra black frame is dropped (a non-reference
       one), or one black duplicate is kept;
     - a **hard cut**: one frame next to the cut is dropped, nearest first:
       the last frame of the old shot or the first of the new one (a pure
       trim, as an editor would make), then the second-last or second frame.
       Or the dropped duplicate nearest before the cut (within 6 frames) is
       kept;
     - a **static frame**, a visually lossless repeat of its predecessor
       (the rule-4 test) that is not a duplicate twin: it is dropped, or a
       dropped duplicate there is kept when the next frame repeats too.
   - Only cheap pictures (non-reference, or a reference nothing reads) are
     dropped, never an IDR, a substitute twin or a rebuild picture, so every
     other frame stays bit-exact.
   - Each correction is in the report's `sync.corrections` (`at`: black,
     cut or static).
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
      A frame is bit-exact only if the whole coded frame is; the per-MB
      statistics (and `--mb-report`) count only the visible area. Until 2.0
      they included the 8 cropped rows below 1080, which made the Amazon
      clip's worst difference 76 instead of 37.
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

### Residual corrections (2.1)

Version 2.0 repaired drift with I_PCM only. An I_PCM macroblock costs about
370 bytes, so the measured repair could afford a threshold of 16 and a 2%
per-picture cap. It spent 11% of the room the drops freed and left
macroblocks off by up to 100 levels. Since 2.1 the measured repair first
*corrects the residual* of a drifting macroblock and keeps its mode, motion
and QP; I_PCM is the fallback. A correction costs about 30-40 bytes.
`--no-corrections` restores 2.0's behaviour exactly.

- **Reconstruction** (`recon.rs`). An inter macroblock decodes to
  clip(prediction + residual) before deblocking. The prediction is computed
  from frames ffmpeg has already decoded, which is exactly what motion
  compensation reads: quarter-sample luma (6-tap), eighth-sample chroma,
  default, explicit (P) and implicit (B) weighted prediction, all partition
  sizes, direct included (motion from `mv::derive_motion`). The residual
  needs dequantisation (flat scaling lists) and the 4x4, 8x8 and chroma DC
  inverse transforms. No intra prediction and no deblocking filter are
  needed.
  - `frame-dedup dev recon <file.h264>` checks it: it decodes IDR periods
    with ffmpeg's deblocking switched off (`-skip_loop_filter all`, so every
    picture and every reference is the pre-deblocking reconstruction) and
    rebuilds every inter macroblock. It is bit-exact on about 40 million
    inter macroblocks of all four test files (P and B, weighted, implicit,
    temporal and spatial direct, skip, 4x4 and 8x8 residuals). With
    deblocking left on, 74% mismatch, so the check is not vacuous.
- **The correction** (`recon::correct`). The levels of the macroblock are
  changed so that its reconstruction over the new prediction comes as close
  as the quantiser allows to a target. The change is a least-squares
  projection of (target − current reconstruction) on each transform block's
  basis (the integer transforms are orthogonal), repeated a few times, then
  a greedy ±1 search on the levels of the block holding the worst sample.
  - If the worst error is still over the limit (half the threshold, at least
    3), the QP is lowered by 6, 12, then 18 through `mb_qp_delta`. At QP−6k
    the original levels times 2^k dequantise to exactly the same values
    (`recon::tests::qp_minus_6_doubles_exactly`), so the original residual
    is kept exactly and only the correction is quantised more finely. The
    B pictures of the Amazon file run at QP 25-35, where one level is
    several sample values; without this most corrections stayed 5-27 off.
  - Motion and macroblock type are kept. A P_Skip becomes P_L0_16x16 (its
    mvd is re-solved), a B_Skip becomes B_Direct_16x16. A macroblock that
    had no luma residual gets the 4x4 transform, which also keeps the set of
    deblocked edges.
  - Corrections go through `patch_pictures` like I_PCM: an inter
    replacement record with its own QP (the QP chain is re-solved) and a
    version for the patch cache. Unlike I_PCM it keeps its motion, so B
    pictures that use it as co-located macroblock are unaffected.
- **The target** is the original's pre-deblocking reconstruction (the
  original motion over the original decoded references, plus the original
  residual: exact), plus the difference between what deblocking did to the
  macroblock in the original and what it does in the current decode. The
  second term makes up for deblocking that differs, for example next to an
  I_PCM macroblock.
- **The prediction** reads the references *as they will be after this
  round*: macroblocks fixed in a reference picture this round are taken as
  the original's. Without that, a correction in a chain of references was
  made against a reference that changed in the same round, flipped the sign
  of the error, and left macroblocks 79 levels off.
- **The loop** (`repair::run_task_explicit`). Every round decodes the IDR
  period, compares it with the original and fixes every picture:
  - macroblocks off by `--drift-threshold` (6) or more are corrected;
  - intra macroblocks, and macroblocks no correction fixes (error over the
    limit, three tries, or a correction that would not change), get I_PCM
    from `--pcm-threshold` (8) on; below it they are left as they are;
  - the last round makes no corrections, only I_PCM, since nothing measures
    it again; up to 20 rounds (most periods need 3-6);
  - a pooled byte budget (`--drift-budget`, 1.0): the repair may add as
    many bytes as the drops freed. Each period starts with its own dropped
    bytes; one that runs out stops at once, and when all periods are done,
    those that stopped are repaired again with an equal share of what the
    others left (up to two such passes; the shares only depend on finished
    passes, so the output does not depend on worker scheduling). Where even
    that runs out, the last pass repairs only from the I_PCM threshold on.
    Most periods free far more than they need; a few (fast motion, a long
    period with few drops) need many times their own share;
  - reference frames are kept while a picture that reads them still has to
    be handled (pictures arrive in display order, references in decode
    order).
- **Rebuilt pairs** (exact mode, `rebuild.rs`). The leading B pictures that
  now read I2 instead of K get the same corrections from 8 levels on, and
  I_PCM from 8 where no correction fixes a macroblock (2.1: from 16; no
  per-picture cap with corrections). They are not references, so nothing
  propagates.
- **Speed** (2.2). Each round used to parse the period's pictures and decode
  the original again. A task now keeps the parsed original pictures (24,
  the first 16 for good: plain LRU over a period longer than its capacity
  evicts every picture before it is read again) and, for tasks of up to 64
  original pictures, the decoded original frames. The corrections keep what
  they need of each original macroblock (record, QP, motion, pre-deblocking
  reconstruction). The refinement scores only the samples a level change
  touches. Long periods are started first. None of this changes the
  output.

What did not work, measured on the 5-minute Amazon clip at threshold 6 /
I_PCM 8 unless noted (+11.8 MB, worst 12, as reference):

| attempt | result |
|---|---|
| I_PCM only, thresholds 12 / 8 / 4, no cap | +11.9 / +37 / +368 MB (the drops free 24 MB) |
| corrections at the macroblock's own QP only | 496 of 15k usable; errors 5-27 |
| corrections against the current decode, all pictures each round | worst 79 (stale corrections in reference chains) |
| predicted references only for first-time fixes | oscillation, worst 154 |
| a stricter threshold (T−2) for reference pictures | +149 MB at 6: deblocking-floor I_PCM everywhere |
| intra macroblocks re-coded as inter + residual | +29.7 MB: they never settle under their own deblocking |
| the Canadian files through explicit mode (`--force-explicit`) | sample: +17-19 MB, 216k macroblocks differ (exact mode: 53 KB, 44k) |
| a correction clean-up after the exact repair (threshold 2) | sample: worst 5 → 20; ±1 errors spread through P chains |

I_PCM has a weakness the corrections avoid: it is an intra macroblock, so
its edges with inter neighbours get bS 4, H.264's strong filter. That filter
rewrites three samples deep and pulls the third towards the fourth; next to
the dark edge of white text it moved a flat white sample by 38 levels on the
full Amazon file. Corrections keep the macroblock inter (bS 0-2, normal
filter, at most a few levels).

### Modules

| module | role |
|---|---|
| `h264hdr.rs`, `bits.rs` | Annex-B NAL iterator, SPS (incl. full VUI) / PPS / slice header parser, escape / unescape, bit reader / writer, in-place SPS timing patch |
| `cabac.rs`, `records.rs`, `picture.rs` | CABAC slice-data codec (decode and encode through one code path, I_PCM included); macroblock records; byte-exact round trip |
| `mv.rs` | `dpb_walk` (POC, sliding-window ref lists) and `derive_motion` (P_Skip, temporal and spatial direct, mvd solving) |
| `reexpress.rs` | explicit re-expression of skip / direct macroblocks; motion re-expression test |
| `patch.rs` | macroblock patcher (I_PCM or corrected inter records) that keeps the rest of the picture exact (`patch_pictures`) |
| `recon.rs` | inter prediction, dequantisation and inverse transforms of single macroblocks; residual corrections (`correct`) |
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
  physical RAM: 0.8 GB per repair worker, 1.5 GB per explicit-mode repair
  worker, with or without corrections (9 workers on 31.7 GB), and 0.5 GB
  per verify worker. With corrections the budget is measured: about 1 GB
  of private memory per worker on the 5-minute Amazon clip, 1.4 GB on the
  full file (12.7 GB peak with 9 workers). The report gives the peak
  private memory (`resources.main_peak_private_mb`) next to the peak
  working set, which also counts the pages of the two memory-mapped
  streams (6 GB each for the full Amazon file).
- An explicit-mode repair task holds a bounded number of parsed pictures
  (`PARSED_CAP`, 24 with corrections; 24 more parsed original pictures) and
  the co-located motion fields of reference pictures. It keeps the decoded
  original frames only when the task has at most 64 original pictures,
  and otherwise decodes the original again each round. It keeps only the
  samples of the MBs it patches. A parsed 1080p picture holds about 11 MB of
  records and 2.6 MB of motion.
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

### Version 2.2: pooled budget, faster repair

Default settings, `--verify`, 32-thread / 31.7 GB machine; counts as for
2.1 below. sample.mkv and foliage.mkv are identical per stream to 2.1 (and
1.2.0), nodupes.mkv exits with code 2, and `--no-corrections` still gives
2.0's output (checked on the clip).

Amazon, 5-minute clip (the drops free 24.1 MB):

| | 2.1 | 2.2 |
|---|---|---|
| repair: bytes added | +9.6 MB (40% of the freed bytes) | +11.8 MB (49%) |
| periods left over budget | 9 of 88 | 0 (9 repaired again with pooled shares) |
| MBs off by ≥ 4 / 6 / 8 / 12 / 16 | 851,204 / 43,886 / 3 / 0 / 0 | 826,114 / 6,397 / 1 / 1 / 0 |
| worst difference | 9 | 12 (one I_PCM macroblock) |
| runtime (repair) | 185 s (139 s) | 175 s (136 s) |

Amazon, full file (the drops free 501 MB of 5,898 MB of video):

| | 2.1 | 2.2 |
|---|---|---|
| repair: bytes added | +121.9 MB (24%) | +153.1 MB (31%) |
| output file | 5,660 MB | 5,692 MB (input: 6,040 MB) |
| periods left over budget | 87 of 1,403 | 0 (87 repaired again) |
| MBs off by ≥ 6 / 8 / 12 / 16 / 32 | 283,823 / 10 / 5 / 5 / 5 | 45,254 / 11 / 8 / 6 / 3 |
| worst difference | 38 | 38 (the same I_PCM macroblock) |
| differing outside hard-drop neighbourhoods | 0 | 0 |
| repair workers | 6 | 9 |
| runtime (repair, verify) | 53.5 min (42, 8) | 52.0 min (41, 8) |
| peak private memory | | 12.7 GB |

smoke.mkv, the leading B pictures of its 300 anchor rebuilds:

| | 2.1 | 2.2 |
|---|---|---|
| bytes added | +16.8 MB | +20.5 MB (output still 131 MB smaller than the input) |
| MBs off by ≥ 8 / 12 / 16 / 24 / 32 | 4,150 / 1,383 / 14 / 4 / 2 | 78 / 31 / 8 / 1 / 0 |
| worst difference | 35 | 28 |
| runtime (rebuild) | 384 s (126 s) | 388 s (131 s) |

Where the time goes (the full file's repair, summed over workers): writing
patched pictures (parsing, re-deriving motion, CABAC encoding) about 40%,
the corrections about 25% (of which parsing original pictures is half),
waiting for the two decoders about 25%. The caches help the clip most:
on the full file, periods with more than 32 rewritten pictures take 61% of
the repair time, and the merged periods of 200-400 pictures (the longest
tasks) are longer than the caches. On the clip the repair stays far from
CPU-bound (35-45% of the machine with 9 workers), but each worker gets
slower as workers are added (about 35% from 6 to 9), which points at
contention in memory allocation: a parsed picture is tens of thousands of
small allocations.

### Version 2.1: residual corrections

Default settings, `--verify`, 32-thread / 31.7 GB machine. Difference
counts are luma, visible area, per macroblock. The 2.0 counts of the
full file come from 2.0's `--mb-report`, which still included the cropped
rows (it inflates the top of the tail slightly, not its size).

sample.mkv and foliage.mkv have no anchor rebuilds and no explicit
reference control, so 2.1 leaves them untouched: their outputs are
identical per stream to 2.0's (and to 1.2.0's). nodupes.mkv still exits
with code 2.

Amazon, 5-minute clip (the drops free 24.1 MB):

| | 2.0 | 2.1 |
|---|---|---|
| repair: I_PCM / corrected MBs | 9,584 / 0 | 15,797 / 461,212 |
| repair: bytes added | +3.6 MB (15% of the freed bytes) | +9.6 MB (40%) |
| periods that hit the byte budget | | 9 of 88 |
| MBs off by ≥ 4 / 6 / 8 / 12 / 16 / 32 | 1,212,629 / 425,432 / 176,269 / 52,958 / 15,736 / 45 | 851,204 / 43,886 / 3 / 0 / 0 / 0 |
| worst difference | 37 | 9 |
| runtime (repair) | 93 s (55 s) | 185 s (139 s) |

Amazon, full file (the drops free 501 MB of 5,898 MB of video):

| | 2.0 | 2.1 |
|---|---|---|
| repair: I_PCM / corrected MBs | 149,483 / 0 | 129,136 / 6,817,950 |
| repair: bytes added | +56.2 MB (11%) | +121.9 MB (24%) |
| output video | 5,453 MB | 5,518 MB |
| periods that hit the byte budget | | 87 of 1,403 |
| MBs off by ≥ 8 / 12 / 16 / 24 / 32 | 2,076,015 / 437,192 / 112,718 / 20,871 / 3,109 | 10 / 5 / 5 / 5 / 5 |
| worst difference | 103 | 38 (one I_PCM macroblock, see "Residual corrections") |
| differing outside hard-drop neighbourhoods | 0 | 0 |
| runtime (repair, verify) | 33.5 min (21, 8) | 53.5 min (42, 8) |
| peak RSS (main process) | 6.7 GB | 11.5 GB |

smoke.mkv, the 592 leading B pictures of its 300 anchor rebuilds (all other
frames are as in 2.0):

| | 2.0 | 2.1 |
|---|---|---|
| I_PCM / corrected MBs | 19,205 / 0 | 6,360 / 289,604 |
| bytes added | +12.3 MB | +16.8 MB |
| MBs off by ≥ 8 / 12 / 16 / 24 / 32 | 276,812 / 67,477 / 12,800 / 2,163 / 960 | 4,150 / 1,383 / 14 / 4 / 2 |
| worst difference | 56 | 35 |

Review pages (`review/clip-2.0-worst`, `clip-2.1-worst`, `smoke-2.0-worst`,
`smoke-2.1-worst`, served by `.claude/launch.json`) show 96x96 crops of the
worst macroblocks of each version: original, both outputs, and their
differences ×8.

### Earlier results (1.x / 2.0)

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
| frames in → out | 150,499 → 120,400 |
| chunks / phase changes | 9 / 10 (8 at black runs, 2 at hard cuts) |
| drops | 30,099, of which 88% are non-reference B pictures (bit-exact); one is a sync trim, the first frame of a shot after the phase change at a cut |
| unreferenced reference twins dropped (bit-exact) | 382 |
| hard pairs (measured repair) | 2,716 |
| anchor rebuilds (fallbacks) | 423 (0). They chain: up to 18 IDR periods merge into one. |
| substitutes (invisible / visible) | 13 / 0 |
| slice headers rewritten | 60,319 |
| repair | 149,483 PCM MBs, +56.2 MB (1.0%) |
| pictures left over the drift threshold | 291, all at the 2% PCM cap; visible worst 56 |
| verify: bit-exact / differing frames | 73,476 / 46,924. The differing frames are noise-level drift, all inside hard-drop neighbourhoods and rebuilt leading pictures. |
| differing anywhere else / decoder errors | 0 / 0 |
| output size | 5.59 GB (input 6.04 GB) |
| video − audio duration change | +0.034 s (0.076 s before cuts became sync points: the two phase changes at hard cuts in the last 3 minutes have no black run after them). Sync offset: max 1.4 frames, final 0.6. |
| runtime | 94 min in Python v1.2.0 (repair 79 min on 9 workers, verify 14 min); 33 min in Rust 2.0 |

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
  git history. So does the whole Python implementation, removed in 2.0.0.

## Limitations

- Scope: High/Main profile, 4:2:0, 8-bit, progressive, CABAC, one slice
  per picture, POC type 0, one SPS id, no long-term references. MMCO 1 and
  short-term ref-list modification are supported (explicit mode); other
  MMCO operations are not. Anything else exits with code 3. Only 29.97 and
  30 fps inputs are processed.
- Explicit mode repairs by measurement, not exactly: frames that read a
  dropped twin keep encode-noise-level differences until the next IDR:
  below `--drift-threshold` (6), between it and `--pcm-threshold` (8) where
  no correction fixes a macroblock or a period's byte budget ran out, and
  rarely above (an I_PCM macroblock next to a sharp edge, see "Residual
  corrections"). With `--no-corrections`, as in 2.0: below 16, or above it
  where the 2% per-picture PCM cap ran out.
- A correction costs the time of reconstructing macroblocks and of more
  repair rounds: the full Amazon file takes 52 minutes instead of 2.0's
  34 (2.1: 54).
- Hard drops in exact mode leave differences of 1-8 levels around the I_PCM
  patches. They come from deblocking across PCM edges, and from intra
  prediction and later references that read post-deblock PCM pixels.
  Corrections do not help there: they are accurate to ±1-3 before
  deblocking, not exact, and below a threshold of about 3 those errors
  spread through P chains (tried: worst 5 became 20 on sample.mkv). Removing
  them would take the tool's own deblocking filter and intra prediction.
- An anchor rebuild changes the B pictures displayed just before the pair:
  they now read I2, a second encode of K's frame. Differences are
  encode-noise level; corrections and I_PCM bring macroblocks off by 8 or
  more back (on smoke.mkv 78 macroblocks stay at 8 or more, 8 at 16 or
  more, worst 28). With
  `--no-corrections` the 2% PCM cap leaves fast-motion pictures at up to
  about 56 levels.
- After a rebuild, I2 is an open-GOP random-access point rather than an
  IDR. A seek to it skips its leading B pictures, as with any open GOP.
  A player that seeks only to IDRs lands on the previous real IDR.
- A visible substitute (rule 7) is only used when a rebuild is not possible.
  It drops one real frame next to a duplicate that had to be kept: a
  one-frame stutter. `--keep-idr-pairs` keeps both twins instead and accepts
  A/V drift.
- The sync offset after a mid-chunk phase change or an unsupported pair
  lasts until the next sync point where a cheap picture is available (a
  black run, a static frame, or a cut with a non-reference picture next to
  it). The offsets are reported.
- Native-rate intros (logos) are forced through the content cadence, which
  can drop real frames there. This is by design: the output is always CFR.
- The likelihood tables were trained on one file from this encoder. Other
  encoders may need retraining with `frame-dedup dev train-model` on a labelled
  file.

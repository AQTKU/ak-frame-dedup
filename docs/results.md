# Results

The measurements below come from a handful of test files, called sample A to
E here:

| alias | what it is |
|---|---|
| sample A | 22 minutes, 29.97 fps from 23.976. Opens with a native 29.97 logo. Its frame timestamps label the duplicates, so the likelihood tables were trained on it. |
| sample B | 44 minutes, 29.97 fps from 23.976. |
| sample C | 44 minutes, 29.97 fps from 23.976. The encoder's IDR, every 90 frames, often lands on the later twin of a duplicate, so it exercises the anchor rebuild. |
| sample D | 22 minutes of native 29.97 content with no duplicates: the tool must refuse it. |
| sample E | 84 minutes, an x264 encode with B-pyramid, MMCO, reference list modification and weighted prediction (explicit mode). The *sample E clip* is a 5-minute excerpt of it. |
| sample F | 45 minutes, mixed content: shots at native 29.97 between 23.976 ones (detected as 24 fps, 153 phase changes). The encoder starts every shot with an I picture and uses two forward references. |

Samples A-C are plain (exact mode) encodes; sample E needs explicit mode.

## Versions

The tool is a Rust port of the Python implementation (v1.2.0): 2.0 had
byte-identical output (see "Rust port" below). 2.1 repairs drift with
residual corrections (see "Residual corrections" in [how-it-works.md](how-it-works.md)); `--no-corrections` gives
2.0's output. 2.2 pools the byte budget across IDR periods, repairs the
rebuilt leading B pictures from 8 levels, and makes the measured repair
faster; 2.2.1 uses the mimalloc allocator (same output, the full sample E
file in 41 minutes instead of 52). 2.2.2 never predicts a duplicate on the
first frame after a cut and refuses hard drops that would leave a reference
list short (output identical on samples A-C and E).

## Version 2.2.2: mixed-rate content

Sample F made 2.2.1 fail with `list index out of range`. In its native 29.97
shots the cadence phase is arbitrary, and 97 predicted duplicates fell on the
first frame of a shot, i.e. on the encoder's scene-cut I picture: 77 of the
96 hard pairs were such P + I pairs, and one IDR + I pair left the next P
with one reference where its header makes two active. Results with default
settings and `--verify`:

| | 2.2.1 | 2.2.1 `--force-explicit` | 2.2.2 |
|---|---|---|---|
| status | error | ok | ok |
| hard pairs | 96 | 96 | 21 |
| output size (input 2.02 GB) | | 2.78 GB | 1.89 GB |
| repair | (1.08 M PCM MBs before the error) | 2.15 M PCM MBs | 25,195 PCM MBs, +9.7 MB |
| max A/V offset | 52.3 ms | | 52.3 ms |
| runtime | | 13.2 min | 5.0 min |

Verify finds no differences outside the hard-drop neighbourhoods and rebuilt
leading B pictures. With only the reference-list check (the cut rule turned
off) the IDR + I pair is kept instead of failing, but the repair adds 787 MB.
Samples A, B, C, the sample E clip and the full sample E file are identical
per stream to 2.2.1.


## Version 2.2: pooled budget, faster repair

Default settings, `--verify`, 32-thread / 31.7 GB machine; counts as for
2.1 below. sample A and sample B are identical per stream to 2.1 (and
1.2.0), sample D exits with code 2, and `--no-corrections` still gives
2.0's output (checked on the clip).

Sample E clip (the drops free 24.1 MB):

| | 2.1 | 2.2 |
|---|---|---|
| repair: bytes added | +9.6 MB (40% of the freed bytes) | +11.8 MB (49%) |
| periods left over budget | 9 of 88 | 0 (9 repaired again with pooled shares) |
| MBs off by ≥ 4 / 6 / 8 / 12 / 16 | 851,204 / 43,886 / 3 / 0 / 0 | 826,114 / 6,397 / 1 / 1 / 0 |
| worst difference | 9 | 12 (one I_PCM macroblock) |
| runtime (repair) | 185 s (139 s) | 175 s (136 s); 2.2.1: 148 s (108 s) |

Sample E, full file (the drops free 501 MB of 5,898 MB of video):

| | 2.1 | 2.2 |
|---|---|---|
| repair: bytes added | +121.9 MB (24%) | +153.1 MB (31%) |
| output file | 5,660 MB | 5,692 MB (input: 6,040 MB) |
| periods left over budget | 87 of 1,403 | 0 (87 repaired again) |
| MBs off by ≥ 6 / 8 / 12 / 16 / 32 | 283,823 / 10 / 5 / 5 / 5 | 45,254 / 11 / 8 / 6 / 3 |
| worst difference | 38 | 38 (the same I_PCM macroblock) |
| differing outside hard-drop neighbourhoods | 0 | 0 |
| repair workers | 6 | 9 |
| runtime (repair, verify) | 53.5 min (42, 8) | 52.0 min (41, 8); 2.2.1: 40.8 min (30, 8) |
| peak private memory | | 12.7 GB; 2.2.1: 12.4 GB |

sample C, the leading B pictures of its 300 anchor rebuilds:

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
small allocations. 2.2.1 switches the allocator to mimalloc, which cuts the
repair from 41 to 30 minutes on the full file (136 to 108 s on the clip)
with identical output.

## Version 2.1: residual corrections

Default settings, `--verify`, 32-thread / 31.7 GB machine. Difference
counts are luma, visible area, per macroblock. The 2.0 counts of the
full file come from 2.0's `--mb-report`, which still included the cropped
rows (it inflates the top of the tail slightly, not its size).

sample A and sample B have no anchor rebuilds and no explicit
reference control, so 2.1 leaves them untouched: their outputs are
identical per stream to 2.0's (and to 1.2.0's). sample D still exits
with code 2.

Sample E clip (the drops free 24.1 MB):

| | 2.0 | 2.1 |
|---|---|---|
| repair: I_PCM / corrected MBs | 9,584 / 0 | 15,797 / 461,212 |
| repair: bytes added | +3.6 MB (15% of the freed bytes) | +9.6 MB (40%) |
| periods that hit the byte budget | | 9 of 88 |
| MBs off by ≥ 4 / 6 / 8 / 12 / 16 / 32 | 1,212,629 / 425,432 / 176,269 / 52,958 / 15,736 / 45 | 851,204 / 43,886 / 3 / 0 / 0 / 0 |
| worst difference | 37 | 9 |
| runtime (repair) | 93 s (55 s) | 185 s (139 s) |

Sample E, full file (the drops free 501 MB of 5,898 MB of video):

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

sample C, the 592 leading B pictures of its 300 anchor rebuilds (all other
frames are as in 2.0):

| | 2.0 | 2.1 |
|---|---|---|
| I_PCM / corrected MBs | 19,205 / 0 | 6,360 / 289,604 |
| bytes added | +12.3 MB | +16.8 MB |
| MBs off by ≥ 8 / 12 / 16 / 24 / 32 | 276,812 / 67,477 / 12,800 / 2,163 / 960 | 4,150 / 1,383 / 14 / 4 / 2 |
| worst difference | 56 | 35 |

## Earlier results (1.x / 2.0)

Measured on the four test files on a 32-thread / 31.7 GB machine with
default settings:
- Every run used `--verify`.
- Times are wall-clock. The sample A run reused cached proxy features, which
  cost about 43 s fresh; the sample C run includes its 100 s proxy decode.
- sample B and sample have no IDR-later pairs, so the anchor rebuild never
  runs on them: their outputs are packet-for-packet identical to version
  1.0's (per-stream MD5 over video, audio and subtitles).
- Differences are max absolute over Y/U/V, with luma-only in brackets.

| | sample B | sample A | sample C | sample D |
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

## Sample E (explicit mode, v1.2.0)

An 84-minute x264 encode: B-pyramid with MMCO, ref list modification
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

Seeking on sample C's output: it has 883 keyframes, one per original IDR
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

## Rust port

Version 2.0 is a port of the Python implementation (v1.2.0) to Rust, so
that Python is no longer needed. Every module was checked against the
Python one on real data, then the whole tool end to end:
- Output MKVs are identical per stream (video, audio, subtitles; MD5) to
  v1.2.0's on sample, sample B, sample C, the sample E clip and the full
  Amazon file, and sample D fails out identically. The reports are equal
  except for timings, worker counts, memory figures and the tool version.
- Intermediate checks included: every header of the full sample E stream;
  macroblock records and motion fields of every picture of four clips; the
  stage-1 stream of the full sample E run byte for byte; 119 recorded
  `patch_pictures` calls; the repaired and rebuilt streams of all four test
  files byte for byte, with 1 and 4 workers.
- The proxy features match numpy bit for bit, including its float32
  pairwise summation order, so detection cannot drift.

Wall-clock times with `--verify`, fresh work directories (proxy decode
included unless noted):

| file | Python v1.2.0 | Rust 2.0 |
|---|---|---|
| sample A (8 / 4 workers) | 189 s | 113 s |
| sample E clip (8 / 4 workers) | 240 s | 93 s |
| sample B (8 workers) | 737 s | 208 s |
| sample C (8 workers) | 829 s | 339 s |
| full sample E file (16 workers) | 94 min (proxy features cached) | 31 min |
| full sample E: repair stage | 79 min | 20 min |
| full sample E: verify stage | 14 min | 8 min |

The CABAC codec runs about 40× faster than the Python and motion
derivation about 70× faster. What remains is mostly ffmpeg: the measured
repair and the anchor rebuild decode every affected IDR period several

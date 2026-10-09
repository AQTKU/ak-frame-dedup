# Design notes

The tool was built from a brief with fixed requirements: detect the rates
from the H.264 stream itself and never from container timestamps, always
write constant frame rate, keep A/V sync, refuse files with no duplicate
cadence, and patch the stream's own timing to the output rate. This page
lists where the implementation departs from that brief, and why. Test files
are called sample A to E; [results.md](results.md) describes them.

- **Sniffing covers more than reads of D.** P slices use two active
  references, so dropping D changes `[D, K]` into `[K, older]`. A block that
  used index 1 now reads a different picture, not only blocks that read D.
  Direct blocks and implicit bi-prediction weights also change with the
  reference POCs. The tool therefore compares motion derived in the old and
  new reference contexts for every affected picture; header-only lists pick
  those pictures. `sniff.analyse_window` and its taint prediction are not
  used.
- **IDR on the later twin (sample C).** In sample C's phase-4 chunks the
  encoder's IDR, every 90 frames, falls exactly on the later twin of a
  duplicate, 602 times.
  - The brief's rule (keep both, fix the count at the next black run)
    drifted the video **24 s** against the audio: there are only 8 black
    runs to absorb it.
  - In about half of these pairs the earlier twin is a P that nothing reads,
    because the IDR flushes the DPB. Rule 3 drops it bit-exactly.
  - In the other half the P is the forward anchor of the B-frames displayed
    before it. Version 1.0 dropped the least-motion nearby B-frame instead
    (a *visible substitute*, 300 on sample C). Each one caused two
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
  for example 147 → 140 on sample A.
- **sample A has no phase break.** The evidence puts sample A in one
  phase from the first content frame to the end, including the end credits,
  where the file's 1 ms labels are offset. The native 29.97 logo (frames
  0-66) is run through the content cadence, as requirement 3 asks. That is
  why the tool counts 26 hard pairs (plus one unreferenced twin) on sample A
  where the label-based prototype counted 33.
- **What A/V sync can guarantee.** Inside a constant-phase stretch, the
  offset k − n·r cycles over a window about 0.6-0.8 frame wide. "Within
  half a frame" is therefore enforced on the mean offset of each stretch;
  the pointwise maximum (0.8 frame, 33 ms; 1.4 frames briefly around sample C's
  substitutes) is reported.
- **Duration check.** The inputs already end their video up to 2 frames
  away from their audio. The check therefore requires the video duration to
  match the input's within one frame, so the gap is preserved, rather than
  matching the audio absolutely. Both numbers are in the report.
- **Black threshold.** The gray proxy comes out of swscale at full range,
  so black reads 0-2. A black frame is proxy mean < 20 and max < 40.
- **Verify** counts luma and chroma differences, and also reports
  luma-only figures. The luma-only figures match the prototype's
  measurements (sample A: worst 5, 352 MBs ≥ 3, against the prototype's 5
  and 360).
- **Resource defaults** (see above) were added after a crash under full
  load.
- `refine.py`, `dedup.py` and `drop_test.py` were removed; they remain in
  git history. So does the whole Python implementation, removed in 2.0.0.

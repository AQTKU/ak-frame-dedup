# frame-dedup

In professional broadcasting, video is often delivered at 29.97 FPS. When the
original video is shot at 23.976, 24, or 25 FPS, the most common method to
convert to 29.97 is repeating whole frames: one in every 5 or 6. This adds a
noticeable and unpleasant stutter.

The most common solution to this problem is re-encoding the video, which takes
time, and reduces quality. `frame-dedup` removes repeated frames from an H.264
MKV without re-encoding the video. It edits the compressed video data directly.
The output is a constant-frame-rate MKV at the original film rate, with the
audio, subtitles, chapters and track settings of the input carried over.

## What to expect

The majority of frames in any given video should remain bit-for-bit identical
to the input. The rest may differ slightly.

The reason for frames differing: H.264 video does not store every frame as a
complete picture. Frames are often stored taking parts from future or past
frames. When this tool removes a duplicate other frames were built from, those
frames must be patched to point somewhere else, and the patching isn't always
perfect.

| Kind of file | Frames identical to the input | Frames that differ (levels out of 255) |
|---|---|---|
| Typical encode (no B-pyramid) | About 99% | About 0.4-1.3%, mostly off by at most 5-8 levels, in the frames right after a patched one |
| x264 encode with B-pyramid and explicit reference control | About 61% | About 39%. Half of them have no pixel off by more than 5; 10 frames out of 120,400 have a pixel off by 8 or more (worst 38, in one small block) |

B-pyramid encodes are harder. The encoder codes each duplicate slightly differently
from the frame it repeats, so every frame built from the copy inherits a little
noise until the next keyframe. The tool measures the drift and corrects it block
by block.

The output file should be smaller than the input. A sample file at 6.04 GB was
reduced to 5.69 GB. This tool uses the space saved by removing frames as room
to store frame corrections. 

Full measurements are in [docs/results.md](docs/results.md).

## Using it

1. Download for Windows or Linux from the [Releases](../../releases) page,
   or [build it](#building) yourself.
2. Grab the following, and put them on your `PATH`:
   - `ffmpeg` and `ffprobe` from [FFmpeg](https://ffmpeg.org/download.html);
   - `mkvmerge` from [MKVToolNix](https://mkvtoolnix.download/).

   On Windows, `winget install Gyan.FFmpeg MoritzBunkus.MKVToolNix` installs
   both; check that `mkvmerge --version` works in a new terminal, and if not,
   add the MKVToolNix folder to your `PATH`. On Debian or Ubuntu:
   `sudo apt install ffmpeg mkvtoolnix`.
3. Run it on a video:

   ```
   frame-dedup "my-file.mkv" -o "deduped-file.mkv"
   ```

   It prints what it found (the frame rates, how many frames it will drop)
   and what it is doing. An easy case can take a few minutes. A video that
   needs explicit mode (see [How it works](#how-it-works)) can take much longer.
   A sample on a 16-core CPU took roughly half the runtime of the video.

If you like, you can add `--verify` and the tool decodes both files, compares
every frame, then reports how many are identical.

Intermediate files go to `work/<input name>` in the current folder and are
removed afterwards.

If the file has no repeated frames, the tool says so and writes nothing.

### Options

| Option | Meaning |
|---|---|
| `-o out.mkv` | Output file (required unless `--dry-run`) |
| `--dry-run` | Detect and plan only: shows the rates and drops, writes no video |
| `--verify` | Decode the output and the input and compare every kept frame |
| `--report r.json` | Write a detailed JSON report: rates, drops, repairs, A/V sync, verify results, runtime, memory |
| `--workers N` | Worker threads (default: half your logical CPUs) |
| `--work DIR` | Folder for intermediate files (default `work/<input name>`) |
| `--keep-work` | Keep the intermediate files; a re-run of the same file then skips the slow first analysis |

Advanced options, for tuning the repair and debugging:

| Option | Meaning |
|---|---|
| `--mb-report mb.json` | With `--verify`: list every differing frame and its worst blocks |
| `--mb-threshold N` | The smallest difference `--mb-report` lists (default 8) |
| `--drift-threshold N` | Explicit mode: correct blocks off by N or more (default 6) |
| `--pcm-threshold N` | Explicit mode: replace blocks no correction fixes with raw pixels from N on (default 8) |
| `--drift-budget F` | Explicit mode: the repair may add up to F times the bytes the drops saved (default 1; 0 for no limit) |
| `--no-corrections` | Repair with raw pixel blocks only, as version 2.0 did |
| `--no-rebuild` | Never turn a keyframe into a regular frame to drop the duplicate before it (see [How it works](#how-it-works)) |
| `--keep-idr-pairs` | When that is not possible, keep both copies rather than drop a nearby frame |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Any other error (the report is still written) |
| 2 | No repeated-frame pattern found; no output written |
| 3 | The input uses H.264 features the tool does not support |
| 4 | `--verify` found differences where there should be none |

### What it supports

- H.264 video in an MKV, at 29.97 or 30 fps, made from 23.976, 24 or 25 fps
  content (every combination, including 25 to 29.97).
- 8-bit 4:2:0, progressive, CABAC entropy coding, one slice per picture.
  A few rarer stream features (long-term references, more than one SPS) are not
  supported; see [Limitations](docs/how-it-works.md#limitations).
- Not supported: HEVC / H.265, AV1, interlaced video, CAVLC, multiple
  slices per picture. The tool exits with code 3 on these rather than guess.

It assumes the whole video has one source rate. A short intro made natively at
29.97 (like a studio logo) is pushed through the same pattern and may result
in real dropped motion. The output is constant frame rate video.

## Building

You need [Rust](https://rustup.rs/) (a recent stable version).

```
git clone <this repository>
cd ak-frame-dedup
cargo build --release
```

The program is then at `target/release/frame-dedup` (`frame-dedup.exe` on
Windows). The detection model is compiled in, so that one file is all you
need, plus the three helper programs above. More in
[docs/development.md](docs/development.md).

## How it works

1. The tool extracts the H.264 video data, its bitstream, and reads its headers
   to find the frame rate and frame order.
2. To find duplicates, a small greyscale preview of every frame is generated to
   measure how much each frame differs from the one before. Repeats follow a
   strict rhythm (for 23.976 to 29.97, one in five), so it searches for a target
   rate that best explains the positions of the found duplicate frames (either
   23.976, 24, or 25). The video is split into chunks at black screens and hard
   cuts, where an edit or ad break may have shifted the rhythm, and each chunk
   gets its own position. If no rhythm stands out clearly, the tool stops.
3. The tool selects which of the identical frames to drop.
   - Most frames in an H.264 stream are never used as a reference by other
     frames. If either copy is such a frame, it's simply removed, and every
     other frame stays bit-for-bit identical.
   - If both copies are references, the tool drops one and repairs the frames
     that were built from it. It works out exactly which blocks now come out
     differently and rewrites only those, either with a small correction to the
     stored detail or, failing that, as raw pixels.
   - If the second copy is a keyframe, the keyframe is turned into a regular
     frame that takes the first copy's place.
4. Removing frames at a regular rate keeps the audio and video aligned on its
   own. Where the rhythm shifts, audio and video can drift apart by a frame;
   the tool makes that up somewhere unnoticeable (a black screen, a still shot,
   or a hard cut) by dropping or keeping one extra frame.
5. At the end, the remaining frames are renumbered, the frame-rate field in
   the bitstream is set to the new rate, and mkvmerge puts the video back
   together with the original audio, subtitles, and chapters.

Some encoders, such as x264 with B-pyramid, manage their references
explicitly: each frame lists exactly which earlier frames it uses, and which
ones to forget. For those, the tool rewrites those lists frame by frame
(*explicit mode*) and repairs by measurement: it decodes the result, compares
it with the original, corrects the blocks that drifted, and repeats until
they settle. This is the slow, slightly lossy case from the table above.

The full technical description is in
[docs/how-it-works.md](docs/how-it-works.md).

## Resource use

By default, this tool uses half as many workers as you have logical CPUs, sizes
memory-heavy stages to stay under 40% of RAM, and on Windows it runs everything at
below-normal priority.

## Further reading

- [docs/how-it-works.md](docs/how-it-works.md): every stage in detail,
  explicit mode, residual corrections, the source modules, and the
  limitations.
- [docs/results.md](docs/results.md): measurements for each version on the
  test files.
- [docs/design-notes.md](docs/design-notes.md): design decisions, and why.
- [docs/development.md](docs/development.md): building, tests, dev tools and
  releases.

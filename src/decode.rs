//! Full-resolution decoding of Annex-B byte ranges with ffmpeg.
//! Port of `frame_dedup/decode.py`.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::thread::JoinHandle;

use anyhow::{Result, anyhow, bail};

use crate::h264hdr::{Picture, Sps};
use crate::system::{self, FFMPEG_THREADS};

/// Bytes of one uncropped yuv420p frame.
pub fn frame_size(sps: &Sps) -> usize {
    sps.mb_w * 16 * sps.mb_h * 16 * 3 / 2
}

/// Copy the byte ranges of `path` to the decoder's stdin, then close it.
/// Errors (a decoder that exited early) just end the feed, as in the Python.
fn feed(mut stdin: ChildStdin, path: PathBuf, ranges: Vec<(usize, usize)>) {
    let _ = (|| -> std::io::Result<()> {
        let mut f = File::open(&path)?;
        let mut chunk = vec![0u8; 1 << 22];
        for (a, b) in ranges {
            f.seek(SeekFrom::Start(a as u64))?;
            let mut left = b.saturating_sub(a);
            while left > 0 {
                let n = f.read(&mut chunk[..left.min(1 << 22)])?;
                if n == 0 {
                    break;
                }
                stdin.write_all(&chunk[..n])?;
                left -= n;
            }
        }
        Ok(())
    })();
    // stdin is dropped (closed) here
}

/// Decoded frames of the concatenated byte ranges of an Annex-B file (they
/// must start at an IDR access unit): uncropped yuv420p frames in output
/// order. Iterating yields `Err` (once, at the end) on decoder errors: a
/// non-zero exit or anything on stderr. Dropping the iterator early kills the
/// decoder and joins the helper threads.
pub struct FrameIter {
    child: Child,
    stdout: Option<ChildStdout>,
    feeder: Option<JoinHandle<()>>,
    stderr: Option<JoinHandle<Vec<u8>>>,
    fs: usize,
    path: String,
    done: bool,
}

/// Start decoding. `threads` is ffmpeg's `-threads`; use `FFMPEG_THREADS`.
pub fn iter_frames(path: &Path, ranges: &[(usize, usize)], fs: usize, threads: usize) -> Result<FrameIter> {
    iter_frames_with(path, ranges, fs, threads, &[])
}

/// [`iter_frames`] with extra decoder options (before `-i`), e.g.
/// `["-skip_loop_filter", "all"]`.
pub fn iter_frames_with(path: &Path, ranges: &[(usize, usize)], fs: usize, threads: usize, opts: &[&str]) -> Result<FrameIter> {
    let mut child = system::command("ffmpeg")
        .args(["-v", "error", "-threads"])
        .arg(threads.to_string())
        .args(opts)
        .args(["-apply_cropping", "none", "-f", "h264", "-i", "-", "-fps_mode", "passthrough", "-pix_fmt", "yuv420p"])
        .args(["-f", "rawvideo", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("could not start ffmpeg: {e}"))?;
    let stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let (p, r) = (path.to_path_buf(), ranges.to_vec());
    let feeder = std::thread::spawn(move || feed(stdin, p, r));
    let stderr = std::thread::spawn(move || {
        let mut v = Vec::new();
        let _ = stderr.read_to_end(&mut v);
        v
    });
    Ok(FrameIter {
        child,
        stdout: Some(stdout),
        feeder: Some(feeder),
        stderr: Some(stderr),
        fs,
        path: path.display().to_string(),
        done: false,
    })
}

impl FrameIter {
    /// Wait for the decoder and join the helper threads; returns the stderr text.
    fn shutdown(&mut self) -> (Option<i32>, bool, String) {
        self.stdout = None; // close our end; a still-running ffmpeg dies on the broken pipe
        let status = self.child.wait().ok();
        if let Some(t) = self.feeder.take() {
            let _ = t.join();
        }
        let err = self.stderr.take().map(|t| t.join().unwrap_or_default()).unwrap_or_default();
        let ok = status.map(|s| s.success()).unwrap_or(false);
        let code = status.and_then(|s| s.code());
        (code, ok, String::from_utf8_lossy(&err).trim().to_string())
    }

    /// Read the next frame into `buf` (resized to the frame size). `Ok(false)`
    /// after the last frame; a decoder error is returned instead of that.
    pub fn next_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        if self.done {
            return Ok(false);
        }
        buf.resize(self.fs, 0);
        let out = self.stdout.as_mut().expect("open until done");
        let mut got = 0;
        while got < self.fs {
            match out.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        if got == self.fs {
            return Ok(true);
        }
        // A short read ends the stream (a partial last frame is discarded, as in the Python).
        self.done = true;
        let (_, ok, msg) = self.shutdown();
        if !ok || !msg.is_empty() {
            let msg: String = msg.chars().take(500).collect();
            bail!("decoder error on {}: {}", self.path, msg);
        }
        Ok(false)
    }
}

impl Iterator for FrameIter {
    type Item = Result<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = Vec::new();
        match self.next_frame(&mut buf) {
            Ok(true) => Some(Ok(buf)),
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

impl Drop for FrameIter {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            self.shutdown();
        }
    }
}

/// Decode pictures `[a, b)` (decode order; `a` must be an IDR, all in one IDR
/// period) and return `{poc: frame}` for the POCs in `want`.
pub fn decode_pocs(
    path: &Path,
    pics: &[Picture],
    a: usize,
    b: usize,
    want: &BTreeSet<i32>,
    fs: usize,
    threads: usize,
) -> Result<HashMap<i32, Vec<u8>>> {
    let mut pocs: Vec<i32> = (a..b).map(|d| pics[d].poc).collect();
    pocs.sort_unstable();
    let mut out = HashMap::new();
    let it = iter_frames(path, &[(pics[a].au_start, pics[b - 1].au_end)], fs, threads)?;
    for (i, fr) in it.enumerate() {
        let fr = fr?;
        // Python: pocs[i] raises IndexError when the decoder yields more frames than pictures.
        let poc = *pocs.get(i).ok_or_else(|| anyhow!("list index out of range"))?;
        if want.contains(&poc) {
            out.insert(poc, fr);
        }
    }
    let missing: Vec<i32> = want.iter().copied().filter(|p| !out.contains_key(p)).take(5).collect();
    if !missing.is_empty() {
        bail!("segment decode {a}..{b} did not produce POCs {missing:?}");
    }
    Ok(out)
}

/// MD5 (RFC 1321), only for the `dev framemd5` check against Python's hashlib.
pub fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14,
        20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6,
        10, 15, 21,
    ];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let mut h: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64).wrapping_mul(8)).to_le_bytes());
    for chunk in msg.chunks_exact(64) {
        let m: Vec<u32> = chunk.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect();
        let [mut a, mut b, mut c, mut d] = h;
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let t = d;
            d = c;
            c = b;
            b = b.wrapping_add(a.wrapping_add(f).wrapping_add(k[i]).wrapping_add(m[g]).rotate_left(S[i]));
            a = t;
        }
        h = [h[0].wrapping_add(a), h[1].wrapping_add(b), h[2].wrapping_add(c), h[3].wrapping_add(d)];
    }
    h.iter().flat_map(|w| w.to_le_bytes()).map(|b| format!("{b:02x}")).collect()
}

/// `frame-dedup dev framemd5 <file.h264> [start end] [--take N]`: the MD5 of every
/// decoded frame of the byte range (default the whole file), one `index md5` line
/// each. The frame size is taken from the first frame's size, so it needs
/// `--size WxH` (luma pixels). With `--take N` only N frames are read and the
/// iterator is then dropped early.
pub fn dev_framemd5(args: &[String]) -> Result<()> {
    let mut pos = Vec::new();
    let (mut take, mut size) = (usize::MAX, (1920usize, 1088usize));
    let mut nohash = false; // time the decode alone: print only the frame count
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--nohash" => nohash = true,
            "--take" => take = it.next().ok_or_else(|| anyhow!("--take needs a value"))?.parse()?,
            "--size" => {
                let s = it.next().ok_or_else(|| anyhow!("--size needs WxH"))?;
                let (w, h) = s.split_once('x').ok_or_else(|| anyhow!("--size needs WxH"))?;
                size = (w.parse()?, h.parse()?);
            }
            _ => pos.push(a.clone()),
        }
    }
    let path = PathBuf::from(pos.first().ok_or_else(|| anyhow!("usage: dev framemd5 <file.h264> [start end] [--take N] [--size WxH]"))?);
    let (a, b) = if pos.len() >= 3 {
        (pos[1].parse()?, pos[2].parse()?)
    } else {
        (0, std::fs::metadata(&path)?.len() as usize)
    };
    let fs = size.0 * size.1 * 3 / 2;
    let mut frames = iter_frames(&path, &[(a, b)], fs, FFMPEG_THREADS)?;
    let mut buf = Vec::new();
    let mut n = 0;
    while n < take && frames.next_frame(&mut buf)? {
        if !nohash {
            println!("{n} {}", md5_hex(&buf));
        }
        n += 1;
    }
    if nohash {
        println!("{n} frames");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn md5_known() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(md5_hex(&[b'a'; 1000]), "cabe45dcc9ae5b66ba86600cca6b8ba8");
    }

    /// Reads two frames of a real clip (skipped if the test data is missing), then drops
    /// the iterator: must return promptly with ffmpeg and the helper threads gone.
    #[test]
    fn early_drop_does_not_hang() {
        let p = Path::new(r"C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port\clips\sample.h264");
        let Ok(meta) = std::fs::metadata(p) else { return };
        let t = std::time::Instant::now();
        let mut it = iter_frames(p, &[(0, meta.len() as usize)], 1920 * 1088 * 3 / 2, 2).unwrap();
        let mut buf = Vec::new();
        assert!(it.next_frame(&mut buf).unwrap() && it.next_frame(&mut buf).unwrap());
        assert_eq!(buf.len(), 1920 * 1088 * 3 / 2);
        drop(it);
        assert!(t.elapsed().as_secs() < 20);
    }

    #[test]
    fn garbage_is_an_error_without_hanging() {
        let p = std::env::temp_dir().join(format!("fd_garbage_{}.h264", std::process::id()));
        std::fs::write(&p, vec![0x55u8; 300_000]).unwrap();
        let r: Result<Vec<_>> = iter_frames(&p, &[(0, 300_000)], 3 << 20, 2).unwrap().collect();
        let _ = std::fs::remove_file(&p);
        // Either ffmpeg is missing (spawn error is handled above) or the garbage is reported.
        let e = r.unwrap_err().to_string();
        assert!(e.starts_with("decoder error on "), "{e}");
    }
}

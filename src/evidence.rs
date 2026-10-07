//! Per-frame duplicate evidence from a low-resolution luma proxy.
//! Port of `frame_dedup/evidence.py`.
//!
//! A raw Annex-B decode comes out in display order (IDR period, then POC), so
//! row i of every array here is display frame i.
//!
//! Features per transition i (frame i against frame i-1):
//!   * mad  : mean absolute difference of the 480x270 proxies,
//!   * bmax : max over 16x15 blocks of the block-mean absolute difference.
//! Per frame: mean and max luma of the proxy (for black detection).
//!
//! Floating point mirrors numpy bit for bit (see the `np` module): the Python
//! computes `mad` as a float32 `mean` over 129,600 elements (numpy's pairwise
//! summation in float32), and several report numbers via `round(x, n)`.

use std::io::Read;
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

pub const PW: usize = 480;
pub const PH: usize = 270;
/// Proxy block size (30 x 18 blocks).
pub const BW: usize = 16;
pub const BH: usize = 15;

/// The cadence model, embedded so the binary is self-contained (retrain with
/// `frame-dedup dev train-model`).
const MODEL_JSON: &str = include_str!("cadence_model.json");

/// numpy-compatible numerics.
pub mod np {
    /// numpy's `pairwise_sum` for float64 (loops_utils.h.src): blocks of 8
    /// accumulators up to 128 elements, split in halves (rounded down to a
    /// multiple of 8) above that.
    pub fn pairwise_f64(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = -0.0;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0f64; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise_f64(&a[..n2]) + pairwise_f64(&a[n2..])
        }
    }

    /// The same in float32.
    pub fn pairwise_f32(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            let mut res = -0.0f32;
            for &v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0f32; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - (n % 8) {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise_f32(&a[..n2]) + pairwise_f32(&a[n2..])
        }
    }

    /// `np.sum(a)` / `np.add.reduce(a)` of a contiguous float64 array (the
    /// reduction starts from the identity 0.0, then adds the pairwise sum).
    pub fn sum_f64(a: &[f64]) -> f64 {
        0.0 + pairwise_f64(a)
    }

    /// `a.mean()` of a contiguous float64 array.
    pub fn mean_f64(a: &[f64]) -> f64 {
        sum_f64(a) / a.len() as f64
    }

    /// `np.nanmean(a)` of a float64 array (NaN becomes 0 in the sum; the
    /// count excludes it). NaN if every element is NaN.
    pub fn nanmean_f64(a: &[f64]) -> f64 {
        let cnt = a.iter().filter(|v| !v.is_nan()).count();
        let z: Vec<f64> = a.iter().map(|&v| if v.is_nan() { 0.0 } else { v }).collect();
        sum_f64(&z) / cnt as f64
    }

    /// `np.nanmax(a)`: NaN only if every element is NaN.
    pub fn nanmax_f64(a: &[f64]) -> f64 {
        let mut m = f64::NAN;
        for &v in a {
            if !v.is_nan() && (m.is_nan() || v > m) {
                m = v;
            }
        }
        m
    }

    /// Python's `round(x, nd)` for a float: the decimal string correctly
    /// rounded from the exact binary value (ties to even), read back as the
    /// nearest double.
    pub fn round(x: f64, nd: usize) -> f64 {
        if !x.is_finite() {
            return x;
        }
        format!("{x:.nd$}").parse().unwrap_or(x)
    }

    /// `np.linspace(start, stop, num)` (endpoint included).
    pub fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
        let div = (num - 1) as f64;
        let delta = stop - start;
        let step = delta / div;
        let mut y: Vec<f64> = (0..num).map(|i| i as f64 * step + start).collect();
        if num > 1 {
            y[num - 1] = stop;
        }
        y
    }
}

// --------------------------------------------------------------------------
// proxy decode

/// The four per-frame arrays of the proxy decode (all float64, like numpy's).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Features {
    pub mad: Vec<f64>,
    pub bmax: Vec<f64>,
    pub mean: Vec<f64>,
    pub max: Vec<f64>,
}

/// Decode the Annex-B stream to the gray proxy and return the feature arrays.
/// `threads` is passed to ffmpeg (the Python default is 8).
pub fn proxy_features(es_path: &str, threads: usize) -> Result<Features> {
    let mut cmd = crate::system::command("ffmpeg");
    cmd.args(["-v", "error", "-threads", &threads.to_string(), "-f", "h264", "-i", es_path, "-fps_mode",
              "passthrough", "-vf", &format!("scale={PW}:{PH}:flags=area,format=gray"), "-f", "rawvideo", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn().context("cannot start ffmpeg")?;
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();
    let err_thread = std::thread::spawn(move || {
        let mut s = Vec::new();
        let _ = stderr.read_to_end(&mut s);
        String::from_utf8_lossy(&s).into_owned()
    });

    let f = features_from_reader(&mut stdout)?;
    let status = child.wait()?;
    let err = err_thread.join().unwrap_or_default();
    if !status.success() {
        bail!("proxy decode failed: {}", err.chars().take(500).collect::<String>());
    }
    if f.mad.is_empty() {
        bail!("proxy decode failed: no frames");
    }
    Ok(f)
}

/// Features of a stream of raw 480x270 gray frames (a trailing partial frame is dropped).
pub fn features_from_reader(stdout: &mut dyn Read) -> Result<Features> {
    let fs = PW * PH;
    let mut f = Features::default();
    let mut cur = vec![0u8; fs];
    let mut prev = vec![0u8; fs];
    let mut have_prev = false;
    let mut d32 = vec![0f32; fs];
    loop {
        // Python reads whole batches and drops a trailing partial frame
        match stdout.read_exact(&mut cur) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let sum: u32 = cur.iter().map(|&v| v as u32).sum();
        f.mean.push(sum as f64 / fs as f64); // exact integer sum, one rounding
        f.max.push(*cur.iter().max().unwrap() as f64);
        if !have_prev {
            f.mad.push(0.0);
            f.bmax.push(0.0);
        } else {
            let (mad, bm) = frame_diff(&cur, &prev, &mut d32);
            f.mad.push(mad);
            f.bmax.push(bm);
        }
        std::mem::swap(&mut cur, &mut prev);
        have_prev = true;
    }
    Ok(f)
}

/// (mad, bmax) of one frame against its predecessor, as the Python computes
/// them: float32 `d.mean(axis=(1, 2))` and the max over block means.
pub fn frame_diff(cur: &[u8], prev: &[u8], scratch: &mut [f32]) -> (f64, f64) {
    let fs = PW * PH;
    let (bx_n, by_n) = (PW / BW, PH / BH);
    let mut bsum = vec![0u32; bx_n * by_n];
    for y in 0..PH {
        let by = y / BH;
        let c = &cur[y * PW..(y + 1) * PW];
        let p = &prev[y * PW..(y + 1) * PW];
        for bx in 0..bx_n {
            let mut s = 0u32;
            for x in bx * BW..(bx + 1) * BW {
                s += c[x].abs_diff(p[x]) as u32;
            }
            bsum[by * bx_n + bx] += s;
        }
    }
    let total: u32 = bsum.iter().sum();
    // Integer-valued float32 sums are exact while they stay below 2^24, in
    // any order; above that numpy's pairwise order decides the rounding.
    let s32: f32 = if total < (1 << 24) {
        total as f32
    } else {
        for i in 0..fs {
            scratch[i] = cur[i].abs_diff(prev[i]) as f32;
        }
        0.0 + np::pairwise_f32(&scratch[..fs])
    };
    // float32 sum / int64 count is evaluated in float64, then stored as float32
    let mad = ((s32 as f64) / fs as f64) as f32 as f64;
    // block sums (<= 240 integers) are exact in float32; only the division rounds
    let maxsum = *bsum.iter().max().unwrap();
    let bm = ((maxsum as f64) / (BW * BH) as f64) as f32 as f64;
    (mad, bm)
}

// --------------------------------------------------------------------------
// relative differences and log-likelihood ratios

/// Median of x[max(1, i-half) : i+half+1] for every i (index 0 excluded,
/// since transition 0 has no predecessor). `np.nanmedian` over NaN-padded
/// windows; an empty window gives 0.
pub fn local_median(x: &[f64], half: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    let mut buf: Vec<f64> = Vec::with_capacity(2 * half + 1);
    for i in 0..n {
        buf.clear();
        let lo = i.saturating_sub(half).max(1);
        let hi = (i + half).min(n.saturating_sub(1));
        if lo <= hi {
            buf.extend(x[lo..=hi].iter().copied().filter(|v| !v.is_nan()));
        }
        let c = buf.len();
        if c == 0 {
            continue; // nan_to_num(nan, 0.0)
        }
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (l, h) = (if c % 2 == 1 { c / 2 } else { c / 2 - 1 }, c / 2);
        out[i] = (buf[l] + buf[h]) / 2.0; // masked mean: (low + high) * 1. / 2
    }
    out
}

pub fn rel_diff(x: &[f64]) -> Vec<f64> {
    rel_diff_with(x, 6, 0.05)
}

pub fn rel_diff_with(x: &[f64], half: usize, eps: f64) -> Vec<f64> {
    let med = local_median(x, half);
    let mut r: Vec<f64> = x.iter().zip(&med).map(|(&v, &m)| v / (m + eps)).collect();
    if !r.is_empty() {
        r[0] = 1.0;
    }
    r
}

/// np.histogram(a, bins) counts for increasing bin edges (last bin closed).
fn histogram(a: &[f64], bins: &[f64]) -> Vec<u64> {
    let nb = bins.len() - 1;
    let mut h = vec![0u64; nb];
    for &v in a {
        if v.is_nan() || v < bins[0] || v > bins[nb] {
            continue;
        }
        // index of the last edge <= v, with v == last edge in the last bin
        let k = bins.partition_point(|&b| b <= v);
        let idx = if k > nb { nb - 1 } else { k - 1 };
        h[idx] += 1;
    }
    h
}

/// `np.linspace(-4, 2, 61)`, the default bins of `train_llr`.
pub fn default_bins() -> Vec<f64> {
    np::linspace(-4.0, 2.0, 61)
}

/// Log-likelihood ratio log P(r|repeat)/P(r|not), binned on log(r).
/// Returns (bins, llr).
pub fn train_llr(r: &[f64], repeats: &[usize], bins: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let lr: Vec<f64> = r.iter().map(|&v| v.max(1e-3).ln()).collect();
    let mut rep = vec![false; r.len()];
    for &i in repeats {
        rep[i] = true;
    }
    rep[0] = false;
    let a1: Vec<f64> = (0..r.len()).filter(|&i| rep[i]).map(|i| lr[i]).collect();
    // lr[~rep][1:]: index 0 is never a repeat, so it is the first element dropped
    let a0: Vec<f64> = (1..r.len()).filter(|&i| !rep[i]).map(|i| lr[i]).collect();
    let h1 = histogram(&a1, bins);
    let h0 = histogram(&a0, bins);
    let (s1, s0): (u64, u64) = (h1.iter().sum(), h0.iter().sum());
    let nb = h1.len() as f64;
    let mut llr: Vec<f64> = (0..h1.len())
        .map(|k| {
            let p1 = (h1[k] + 1) as f64 / (s1 as f64 + nb);
            let p0 = (h0[k] + 1) as f64 / (s0 as f64 + nb);
            (p1 / p0).ln()
        })
        .collect();
    // smooth (np.convolve(np.pad(llr, 2, 'edge'), ones(5)/5, 'valid')) ...
    let mut padded = vec![llr[0]; 2];
    padded.extend_from_slice(&llr);
    padded.extend_from_slice(&[llr[llr.len() - 1]; 2]);
    let k = 1.0 / 5.0;
    llr = (0..llr.len())
        .map(|i| {
            let mut s = 0.0;
            for j in 0..5 {
                s += padded[i + j] * k;
            }
            s
        })
        .collect();
    // ... and force monotone (smaller r => more likely a repeat)
    let mut acc = f64::NEG_INFINITY;
    for v in llr.iter_mut().rev() {
        acc = acc.max(*v);
        *v = acc;
    }
    for v in llr.iter_mut() {
        *v = v.clamp(-8.0, 8.0);
    }
    (bins.to_vec(), llr)
}

/// A binned log-likelihood-ratio table for one feature.
#[derive(Clone, Debug, Deserialize)]
pub struct FeatureModel {
    pub bins: Vec<f64>,
    pub llr: Vec<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub mad: FeatureModel,
    pub bmax: FeatureModel,
}

#[derive(Deserialize)]
struct ModelFile {
    trained_on: Option<String>,
    features: Model,
}

/// The emission (per-frame LLR) of the relative differences `r` under a feature model.
pub fn emission(r: &[f64], m: &FeatureModel) -> Vec<f64> {
    let top = m.llr.len() as isize - 1;
    let mut e: Vec<f64> = r
        .iter()
        .map(|&v| {
            // np.digitize(x, bins) = searchsorted(bins, x, side='right')
            let k = m.bins.partition_point(|&b| b <= v.max(1e-3).ln()) as isize - 1;
            m.llr[k.clamp(0, top) as usize]
        })
        .collect();
    if !e.is_empty() {
        e[0] = 0.0;
    }
    e
}

/// True where nothing in the picture moves (local median of the per-frame max
/// block difference below 1.0): there the differences are compression noise
/// that follows the encoder's GOP rhythm.
pub fn static_mask(bmax: &[f64]) -> Vec<bool> {
    static_mask_with(bmax, 1.0, 6)
}

pub fn static_mask_with(bmax: &[f64], floor: f64, half: usize) -> Vec<bool> {
    local_median(bmax, half).into_iter().map(|m| m < floor).collect()
}

/// The embedded cadence model and its `trained_on` metadata.
pub fn load_model() -> Result<(Model, Option<String>)> {
    parse_model(MODEL_JSON)
}

/// Read a model file (same format as `cadence_model.json`).
pub fn load_model_from(path: &str) -> Result<(Model, Option<String>)> {
    parse_model(&std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?)
}

fn parse_model(text: &str) -> Result<(Model, Option<String>)> {
    let m: ModelFile = serde_json::from_str(text).context("parsing the cadence model")?;
    Ok((m.features, m.trained_on))
}

/// Per-frame duplicate log-likelihood ratio (positive: frame i repeats i-1).
pub fn evidence(feat: &Features, model: &Model) -> Vec<f64> {
    let a = emission(&rel_diff(&feat.mad), &model.mad);
    let b = emission(&rel_diff(&feat.bmax), &model.bmax);
    let mut e: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    for (v, s) in e.iter_mut().zip(static_mask(&feat.bmax)) {
        if s {
            *v = 0.0;
        }
    }
    if !e.is_empty() {
        e[0] = 0.0;
    }
    e
}

/// [start, end) display-index runs of black frames lasting >= 0.5 s.
pub fn black_runs(feat: &Features, fps: f64) -> Vec<(usize, usize)> {
    black_runs_with(feat, fps, 0.5, 20.0, 40.0)
}

pub fn black_runs_with(feat: &Features, fps: f64, min_seconds: f64, mean_max: f64, max_max: f64) -> Vec<(usize, usize)> {
    let n = feat.mean.len();
    let blk: Vec<bool> = (0..n).map(|i| feat.mean[i] < mean_max && feat.max[i] < max_max).collect();
    let min_len = (min_seconds * fps).ceil() as usize;
    let (mut runs, mut i) = (Vec::new(), 0);
    while i < n {
        if blk[i] {
            let mut j = i;
            while j < n && blk[j] {
                j += 1;
            }
            if j - i >= min_len {
                runs.push((i, j));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    runs
}

// --------------------------------------------------------------------------
// dev command

/// `dev features-raw <file.gray>`: features of raw 480x270 gray frames.
pub fn dev_features_raw(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { bail!("usage: dev features-raw <file.gray>") };
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    println!("{}", serde_json::to_string(&features_from_reader(&mut r)?)?);
    Ok(())
}

/// `dev train-llr <features.json> <repeats.json>`: `train_llr` on both features
/// with the given repeat indices (a JSON list); prints {mad: [bins, llr], bmax: [...]}.
pub fn dev_train_llr(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("usage: dev train-llr <features.json> <repeats.json>");
    }
    let f: Features = serde_json::from_str(&std::fs::read_to_string(&args[0])?)?;
    let reps: Vec<usize> = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let bins = default_bins();
    let (b1, l1) = train_llr(&rel_diff(&f.mad), &reps, &bins);
    let (b2, l2) = train_llr(&rel_diff(&f.bmax), &reps, &bins);
    println!("{}", serde_json::json!({"mad": [b1, l1], "bmax": [b2, l2]}));
    Ok(())
}

/// `dev train-model <labelled.mkv> [--work DIR] [--out FILE]`: retrain the
/// likelihood tables shipped as `cadence_model.json` (Python:
/// `tools/train_model.py`). Labels come from sample.mkv, whose duplicates were
/// retimed 1 ms after their twin: in PTS order, frame i is a repeat when
/// pts[i] - pts[i-1] <= 5 ms (PTS order equals display order for that file).
pub fn dev_train_model(args: &[String]) -> Result<()> {
    #[derive(Serialize)]
    struct Table {
        bins: Vec<f64>,
        llr: Vec<f64>,
    }
    #[derive(Serialize)]
    struct Tables {
        mad: Table,
        bmax: Table,
    }
    #[derive(Serialize)]
    struct ModelFile {
        trained_on: String,
        n_frames: usize,
        n_repeats: usize,
        features: Tables,
    }
    let mut it = args.iter();
    let (mut mkv, mut work, mut out) = (None, "work/train".to_string(), "src/cadence_model.json".to_string());
    while let Some(a) = it.next() {
        match a.as_str() {
            "--work" => work = it.next().context("--work needs a value")?.clone(),
            "--out" => out = it.next().context("--out needs a value")?.clone(),
            _ => mkv = Some(a.clone()),
        }
    }
    let Some(mkv) = mkv else { bail!("usage: dev train-model <labelled.mkv> [--work DIR] [--out FILE]") };
    std::fs::create_dir_all(&work)?;
    let es = std::path::Path::new(&work).join("train.h264");
    let feat_path = std::path::Path::new(&work).join("train_feat.json");
    let feat: Features = if feat_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&feat_path)?)?
    } else {
        crate::stream::extract_annexb(std::path::Path::new(&mkv), &es)?;
        let f = proxy_features(&es.to_string_lossy(), 8)?;
        std::fs::write(&feat_path, serde_json::to_string(&f)?)?;
        f
    };
    let o = crate::system::command("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time", "-of", "csv=p=0", &mkv])
        .output()
        .context("cannot run ffprobe")?;
    let mut pts: Vec<f64> = String::from_utf8_lossy(&o.stdout)
        .split_whitespace()
        .filter(|x| *x != "N/A")
        .map(|x| x.parse::<f64>())
        .collect::<Result<_, _>>()?;
    pts.sort_by(f64::total_cmp);
    let labels: Vec<usize> = (1..pts.len()).filter(|&i| pts[i] - pts[i - 1] <= 0.005).collect();
    if pts.len() != feat.mad.len() {
        bail!("{} packets but {} proxy frames", pts.len(), feat.mad.len());
    }
    let bins = default_bins();
    let table = |x: &[f64]| {
        let (b, l) = train_llr(&rel_diff(x), &labels, &bins);
        Table { bins: b.iter().map(|&v| np::round(v, 6)).collect(), llr: l.iter().map(|&v| np::round(v, 6)).collect() }
    };
    let model = ModelFile {
        trained_on: std::path::Path::new(&mkv).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or(mkv.clone()),
        n_frames: pts.len(),
        n_repeats: labels.len(),
        features: Tables { mad: table(&feat.mad), bmax: table(&feat.bmax) },
    };
    let mut buf = Vec::new();
    let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
    serde::Serialize::serialize(&model, &mut serde_json::Serializer::with_formatter(&mut buf, fmt))?;
    std::fs::write(&out, buf)?;
    println!("{} frames, {} labelled repeats -> {out}", pts.len(), labels.len());
    Ok(())
}

/// `dev features <file.h264> [threads]`: the proxy features as JSON.
pub fn dev_features(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { bail!("usage: dev features <file.h264> [threads]") };
    let threads = args.get(1).map(|s| s.parse()).transpose()?.unwrap_or(2);
    let f = proxy_features(path, threads)?;
    println!("{}", serde_json::to_string(&f)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_matches_naive_for_small_integers() {
        let a: Vec<f64> = (0..1000).map(|i| (i % 17) as f64).collect();
        assert_eq!(np::sum_f64(&a), a.iter().sum::<f64>());
        let b: Vec<f32> = (0..129600).map(|i| (i % 5) as f32).collect();
        assert_eq!(np::pairwise_f32(&b) as f64, b.iter().map(|&v| v as f64).sum::<f64>());
    }

    #[test]
    fn pairwise_f32_rounds_above_2_24() {
        // 129600 * 255 = 33,048,000: partial sums beyond 2^24 lose bits, and
        // the pairwise order decides which (numpy gives 33048000 here as the
        // blocks are summed exactly up to 128 elements, then halved).
        let a = vec![255f32; 129600];
        assert_eq!(np::pairwise_f32(&a), 33_048_000.0);
    }

    #[test]
    fn local_median_basics() {
        let x = [100.0, 1.0, 5.0, 3.0, 4.0];
        let m = local_median(&x, 1);
        // i=0: window {x1} -> 1; i=1: {x1,x2} -> 3; i=2: {1,5,3} -> 3; i=3: {5,3,4} -> 4; i=4: {3,4} -> 3.5
        assert_eq!(m, vec![1.0, 3.0, 3.0, 4.0, 3.5]);
    }

    #[test]
    fn python_round() {
        assert_eq!(np::round(0.0625, 3), 0.062); // exact tie -> even
        assert_eq!(np::round(0.1875, 3), 0.188);
        assert_eq!(np::round(2.675, 2), 2.67); // binary value is below the tie
        assert_eq!(np::round(-0.0004, 3), -0.0);
    }

    #[test]
    fn model_parses_with_exact_floats() {
        // serde_json must be built with float_roundtrip (Cargo.toml): the model's
        // 17-digit floats have to parse to the same doubles as Python's
        for s in ["2.2250738585072011e-308", "0.30000000000000004", "-1.2345678901234567", "8.41e21"] {
            let v: f64 = serde_json::from_str(s).unwrap();
            assert_eq!(v.to_bits(), s.parse::<f64>().unwrap().to_bits(), "{s}");
        }
        let (m, t) = load_model().unwrap();
        assert_eq!(m.mad.bins.len(), m.mad.llr.len() + 1);
        assert_eq!(t.as_deref(), Some("sample.mkv"));
    }
}

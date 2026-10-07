//! Decode the output and the original and compare every kept frame.
//! Port of `frame_dedup/verify.py`.
//!
//! The pixel helpers (they also include `repair.visible_maxdiff`), the verify
//! job and driver, and the macroblock export (`--mb-report`).
//!
//! Frames are matched by (IDR period, POC) of the original: dropping pictures
//! never touches POC and never drops an IDR, so each output picture has exactly
//! one original. Anchor rebuilds merge IDR periods and shift POCs; for them the
//! caller passes the original of every output picture. The work is split into
//! groups of output IDR periods (each covering whole original periods) decoded
//! by worker threads, one decoder per worker at a time.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use xxhash_rust::xxh3::xxh3_128;

use crate::cadence::Ratio;
use crate::decode::{frame_size, iter_frames};
use crate::h264hdr::{Picture, Sps, Spss};
use crate::mux::py_round;
use crate::system::{self, FFMPEG_THREADS, Logger, VERIFY_PEAK};

/// Max absolute difference over the bytes `a` and `b` (equal length).
#[inline]
fn max_absdiff(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).fold(0u8, |m, (&x, &y)| m.max(x.abs_diff(y)))
}

/// Same for exactly `N` bytes (lets the compiler unroll / vectorise).
#[inline]
fn max_absdiff_n<const N: usize>(a: &[u8; N], b: &[u8; N]) -> u8 {
    let mut m = 0u8;
    for i in 0..N {
        m = m.max(a[i].abs_diff(b[i]));
    }
    m
}

/// Per-block max of |a - b| over a plane `w` wide in `BS` x `BS` blocks,
/// restricted to rows `y0..y1` and columns `x0..x1` (everything outside counts
/// as zero difference). Writes `max` into `out[(h / BS) * (w / BS)]`.
fn plane_blocks<const BS: usize>(a: &[u8], b: &[u8], w: usize, vis: (usize, usize, usize, usize), out: &mut [u8]) {
    let bw = w / BS;
    let (x0, x1, y0, y1) = vis;
    for y in y0..y1 {
        let ra = &a[y * w..(y + 1) * w];
        let rb = &b[y * w..(y + 1) * w];
        let orow = &mut out[(y / BS) * bw..(y / BS + 1) * bw];
        if x0 == 0 && x1 == w {
            for ((o, ca), cb) in orow.iter_mut().zip(ra.chunks_exact(BS)).zip(rb.chunks_exact(BS)) {
                let m = max_absdiff_n::<BS>(ca.try_into().unwrap(), cb.try_into().unwrap());
                *o = (*o).max(m);
            }
        } else {
            for (bx, o) in orow.iter_mut().enumerate() {
                let lo = (bx * BS).max(x0);
                let hi = ((bx + 1) * BS).min(x1);
                if lo < hi {
                    *o = (*o).max(max_absdiff(&ra[lo..hi], &rb[lo..hi]));
                }
            }
        }
    }
}

fn check(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize) -> (usize, usize) {
    let (w, h) = (mb_w * 16, mb_h * 16);
    assert!(a.len() == w * h * 3 / 2 && b.len() == a.len(), "frame size mismatch");
    (w, h)
}

/// Per-macroblock max absolute difference over Y, U and V, plus the luma-only
/// map. Both are row-major `mb_h` x `mb_w` arrays. Mirrors `verify.mb_maxdiff`.
pub fn mb_maxdiff(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize) -> (Vec<u8>, Vec<u8>) {
    let (w, h) = check(a, b, mb_w, mb_h);
    let n = mb_w * mb_h;
    let mut ly = vec![0u8; n];
    plane_blocks::<16>(&a[..w * h], &b[..w * h], w, (0, w, 0, h), &mut ly);
    let mut all = ly.clone();
    let (cw, ch) = (w / 2, h / 2);
    let (u0, v0) = (w * h, w * h * 5 / 4);
    for (o, e) in [(u0, v0), (v0, a.len())] {
        plane_blocks::<8>(&a[o..e], &b[o..e], cw, (0, cw, 0, ch), &mut all);
    }
    (all, ly)
}

/// Per-MB max absolute difference over Y, U and V counting only the pixels
/// inside the SPS cropping window. Mirrors `repair.visible_maxdiff`.
pub fn visible_maxdiff(a: &[u8], b: &[u8], sps: &Sps) -> Vec<u8> {
    let (mb_w, mb_h) = (sps.mb_w, sps.mb_h);
    let (w, h) = check(a, b, mb_w, mb_h);
    // Python doubles the SPS crop offsets (4:2:0): left, right, top, bottom in luma pixels
    let [cl, cr, ct, cb] = sps.crop.map(|c| 2 * c as usize);
    let mut out = vec![0u8; mb_w * mb_h];
    let vis = |s: usize, pw: usize, ph: usize| (cl / s, pw - cr / s, ct / s, ph - cb / s);
    plane_blocks::<16>(&a[..w * h], &b[..w * h], w, vis(1, w, h), &mut out);
    let (cw, ch) = (w / 2, h / 2);
    let (u0, v0) = (w * h, w * h * 5 / 4);
    for (o, e) in [(u0, v0), (v0, a.len())] {
        plane_blocks::<8>(&a[o..e], &b[o..e], cw, vis(2, cw, ch), &mut out);
    }
    out
}

// ---------------------------------------------------------------- verify

/// (IDR period, POC) of an original picture.
pub type Key = (usize, i32);

/// Pictures per verify group: output IDR periods are added to a group until
/// it holds at least this many (Python: `group=300`).
pub const GROUP: usize = 300;

/// One entry of `differing_frames` (the same keys, in the same order, as the
/// Python dict). `display` is the original's display index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DiffFrame {
    pub display: usize,
    pub period: usize,
    pub poc: i32,
    pub worst: u32,
    pub mbs_ge3: usize,
    pub mbs_diff: usize,
    pub worst_luma: u32,
    pub mbs_ge3_luma: usize,
}

/// A macroblock at or above the `--mb-report` threshold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MbHit {
    pub x: u16,
    pub y: u16,
    /// Max |difference| over Y, U and V.
    pub max: u8,
    /// Max |difference| over Y.
    pub luma: u8,
    /// The macroblock overlaps the SPS crop window.
    pub visible: bool,
}

/// A differing frame with its macroblocks at or above the threshold, worst
/// first, handed to the `--mb-report` sink.
#[derive(Clone, Debug)]
pub struct MbFrame {
    pub frame: DiffFrame,
    /// Index of the frame in the output's display order.
    pub out_display: usize,
    pub hits: Vec<MbHit>,
}

/// Macroblock export: `sink` is called on the calling thread, once per
/// differing frame, in output display order (groups are released in order).
pub struct MbOptions<'a> {
    pub threshold: u8,
    pub sink: &'a mut dyn FnMut(&MbFrame) -> Result<()>,
}

/// The report's `verify` section as `verify()` returns it (Python's dict, in
/// the same key order). `worker_peak_rss_mb` is the process-wide peak: the
/// workers are threads.
#[derive(Clone, Debug, Serialize)]
pub struct VerifyResult {
    pub frames_output: usize,
    pub frames_compared: usize,
    pub bit_exact: usize,
    pub differing: usize,
    pub worst_difference: u32,
    pub mbs_ge3: usize,
    pub mbs_differing: usize,
    pub worst_luma: u32,
    pub mbs_ge3_luma: usize,
    pub errors: Vec<String>,
    pub workers: usize,
    pub worker_peak_rss_mb: u64,
    pub differing_frames: Vec<DiffFrame>,
}

/// Whether the macroblock (x, y) overlaps the visible (cropped) area. `crop`
/// is left, right, top, bottom in luma pixels.
fn mb_visible(x: usize, y: usize, mb_w: usize, mb_h: usize, crop: [usize; 4]) -> bool {
    let [cl, cr, ct, cb] = crop;
    let (w, h) = (mb_w * 16, mb_h * 16);
    x * 16 < w.saturating_sub(cr) && x * 16 + 16 > cl && y * 16 < h.saturating_sub(cb) && y * 16 + 16 > ct
}

/// The macroblocks with `all >= thr`, worst first (then larger luma
/// difference, then raster order).
pub fn mb_hits(all: &[u8], luma: &[u8], mb_w: usize, mb_h: usize, thr: u8, crop: [usize; 4]) -> Vec<MbHit> {
    let mut v = Vec::new();
    for y in 0..mb_h {
        for x in 0..mb_w {
            let i = y * mb_w + x;
            if all[i] >= thr {
                v.push(MbHit { x: x as u16, y: y as u16, max: all[i], luma: luma[i], visible: mb_visible(x, y, mb_w, mb_h, crop) });
            }
        }
    }
    v.sort_by(|a, b| b.max.cmp(&a.max).then(b.luma.cmp(&a.luma)).then(a.y.cmp(&b.y)).then(a.x.cmp(&b.x)));
    v
}

struct Job {
    orange: (usize, usize),
    nrange: (usize, usize),
    okeys: Vec<Key>,
    nkeys: Vec<Key>,
    disp_of: HashMap<Key, usize>,
}

#[derive(Default)]
struct JobResult {
    compared: usize,
    exact: usize,
    differing: Vec<(DiffFrame, Vec<MbHit>)>,
    errors: Vec<String>,
}

enum JobError {
    Runtime(String),
    Key(Key),
}

impl std::fmt::Display for JobError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JobError::Runtime(m) => write!(f, "RuntimeError: {m}"),
            JobError::Key(k) => write!(f, "KeyError: ({}, {})", k.0, k.1),
        }
    }
}

fn rt(e: anyhow::Error) -> JobError {
    JobError::Runtime(format!("{e}"))
}

struct Ctx<'a> {
    orig_es: &'a Path,
    out_es: &'a Path,
    sps: &'a Sps,
    fs: usize,
    crop: [usize; 4],
    threshold: Option<u8>,
}

/// Compare one group of IDR periods. Only one ffmpeg decoder runs at a time:
/// the original is decoded and hashed, then the output is decoded and compared
/// by hash; frames that differ are decoded from the original again (kept only
/// for those POCs) to measure per-macroblock differences. Mirrors `_job`.
fn run_job(c: &Ctx, job: &Job) -> JobResult {
    let mut res = JobResult::default();
    if let Err(e) = job_body(c, job, &mut res) {
        res.errors.push(e.to_string());
    }
    res
}

fn job_body(c: &Ctx, job: &Job, res: &mut JobResult) -> std::result::Result<(), JobError> {
    let (mb_w, mb_h) = (c.sps.mb_w, c.sps.mb_h);
    let nset: HashSet<Key> = job.nkeys.iter().copied().collect();
    let mut ohash: HashMap<Key, u128> = HashMap::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut n = 0usize;
    let mut it = iter_frames(c.orig_es, &[job.orange], c.fs, FFMPEG_THREADS).map_err(rt)?;
    while it.next_frame(&mut buf).map_err(rt)? {
        n += 1;
        if n > job.okeys.len() {
            return Err(JobError::Runtime("original group decoded too many frames".into()));
        }
        if nset.contains(&job.okeys[n - 1]) {
            ohash.insert(job.okeys[n - 1], xxh3_128(&buf));
        }
    }
    drop(it);
    if n != job.okeys.len() {
        return Err(JobError::Runtime(format!("original group decoded {n} frames, expected {}", job.okeys.len())));
    }
    let mut bad: HashMap<Key, Vec<u8>> = HashMap::new();
    n = 0;
    let mut it = iter_frames(c.out_es, &[job.nrange], c.fs, FFMPEG_THREADS).map_err(rt)?;
    while it.next_frame(&mut buf).map_err(rt)? {
        n += 1;
        if n > job.nkeys.len() {
            return Err(JobError::Runtime("output group decoded too many frames".into()));
        }
        let key = job.nkeys[n - 1];
        res.compared += 1;
        let want = *ohash.get(&key).ok_or(JobError::Key(key))?;
        if xxh3_128(&buf) == want {
            res.exact += 1;
        } else {
            bad.insert(key, std::mem::take(&mut buf));
        }
    }
    drop(it);
    if n != job.nkeys.len() {
        return Err(JobError::Runtime(format!("output group decoded {n} frames, expected {}", job.nkeys.len())));
    }
    if !bad.is_empty() {
        let mut it = iter_frames(c.orig_es, &[job.orange], c.fs, FFMPEG_THREADS).map_err(rt)?;
        let mut i = 0;
        while !bad.is_empty() && i < job.okeys.len() {
            if !it.next_frame(&mut buf).map_err(rt)? {
                break;
            }
            let key = job.okeys[i];
            i += 1;
            if let Some(fnew) = bad.remove(&key) {
                let (m, ly) = mb_maxdiff(&buf, &fnew, mb_w, mb_h);
                let display = *job.disp_of.get(&key).ok_or(JobError::Key(key))?;
                let df = DiffFrame {
                    display,
                    period: key.0,
                    poc: key.1,
                    worst: *m.iter().max().unwrap_or(&0) as u32,
                    mbs_ge3: m.iter().filter(|&&v| v >= 3).count(),
                    mbs_diff: m.iter().filter(|&&v| v > 0).count(),
                    worst_luma: *ly.iter().max().unwrap_or(&0) as u32,
                    mbs_ge3_luma: ly.iter().filter(|&&v| v >= 3).count(),
                };
                let hits = match c.threshold {
                    Some(t) => mb_hits(&m, &ly, mb_w, mb_h, t, c.crop),
                    None => Vec::new(),
                };
                res.differing.push((df, hits));
            }
        }
    }
    Ok(())
}

/// Decode the output and the original and compare every output frame.
///
/// `orig_of_out`: original decode index of every output picture (output
/// decode order). Without it, output pictures are matched to originals by
/// (IDR period, POC), which holds when pictures were only dropped. Anchor
/// rebuilds merge IDR periods and shift POCs, so the pipeline passes it.
/// `workers` is the pipeline's worker count; verify uses half of it, because
/// each worker keeps a 2-thread ffmpeg decoder busy.
#[allow(clippy::too_many_arguments)]
pub fn verify(
    orig_es: &Path,
    out_es: &Path,
    spss: &Spss,
    orig_pics: &[Picture],
    out_pics: &[Picture],
    workers: usize,
    log: &Logger,
    orig_of_out: Option<&[usize]>,
    mut mb: Option<MbOptions>,
) -> Result<VerifyResult> {
    let sps = spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let fs = frame_size(sps);

    fn periods(pics: &[Picture]) -> BTreeMap<usize, Vec<usize>> {
        let mut out: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for p in pics {
            out.entry(p.period).or_default().push(p.dec);
        }
        out
    }
    let (po, pn) = (periods(orig_pics), periods(out_pics));
    let key_of: HashMap<usize, Key> = match orig_of_out {
        None => out_pics.iter().map(|q| (q.dec, (q.period, q.poc))).collect(),
        Some(o) => {
            if o.len() != out_pics.len() {
                bail!("AssertionError: orig_of_out has {} entries for {} output pictures", o.len(), out_pics.len());
            }
            out_pics.iter().zip(o).map(|(q, &o)| (q.dec, (orig_pics[o].period, orig_pics[o].poc))).collect()
        }
    };
    // each output IDR period covers whole original IDR periods
    let cover: BTreeMap<usize, Vec<usize>> = pn
        .iter()
        .map(|(&g, v)| {
            let mut c: Vec<usize> = v.iter().map(|d| key_of[d].0).collect();
            c.sort_unstable();
            c.dedup();
            (g, c)
        })
        .collect();
    let flat: Vec<usize> = cover.values().flatten().copied().collect();
    let all_orig: Vec<usize> = po.keys().copied().collect();
    if flat != all_orig || cover.values().any(|c| c.iter().enumerate().any(|(i, &v)| v != c[0] + i)) {
        bail!("output IDR periods do not map onto whole original IDR periods");
    }
    let mut jobs: Vec<Job> = Vec::new();
    let mut cur: Vec<usize> = Vec::new();
    let flush = |cur: &mut Vec<usize>, jobs: &mut Vec<Job>| {
        if cur.is_empty() {
            return;
        }
        let og: Vec<usize> = cur.iter().flat_map(|g| cover[g].iter().copied()).collect();
        let (a, b) = (og[0], *og.last().unwrap());
        let mut okeys: Vec<Key> = og.iter().flat_map(|g| po[g].iter().map(|&d| (orig_pics[d].period, orig_pics[d].poc))).collect();
        okeys.sort_unstable();
        let mut nkeys: Vec<Key> = Vec::new();
        for g in cur.iter() {
            let mut qs: Vec<usize> = pn[g].clone();
            qs.sort_by_key(|&d| out_pics[d].poc); // stable, like sorted(key=poc)
            nkeys.extend(qs.iter().map(|d| key_of[d]));
        }
        let disp_of: HashMap<Key, usize> =
            og.iter().flat_map(|g| po[g].iter().map(|&d| ((orig_pics[d].period, orig_pics[d].poc), orig_pics[d].disp))).collect();
        let first = |m: &BTreeMap<usize, Vec<usize>>, pics: &[Picture], g: usize| pics[m[&g][0]].au_start;
        let last = |m: &BTreeMap<usize, Vec<usize>>, pics: &[Picture], g: usize| pics[*m[&g].last().unwrap()].au_end;
        jobs.push(Job {
            orange: (first(&po, orig_pics, a), last(&po, orig_pics, b)),
            nrange: (first(&pn, out_pics, cur[0]), last(&pn, out_pics, *cur.last().unwrap())),
            okeys,
            nkeys,
            disp_of,
        });
        cur.clear();
    };
    let mut n = 0;
    for (&g, v) in &pn {
        cur.push(g);
        n += v.len();
        if n >= GROUP {
            flush(&mut cur, &mut jobs);
            n = 0;
        }
    }
    flush(&mut cur, &mut jobs);
    let out_disp_of: HashMap<Key, usize> = out_pics.iter().map(|q| (key_of[&q.dec], q.disp)).collect();

    let crop = sps.crop.map(|c| 2 * c as usize);
    let threshold = mb.as_ref().map(|m| m.threshold);
    let ctx = Ctx { orig_es, out_es, sps, fs, crop, threshold };
    let nproc = system::pool_size((workers / 2).max(1), VERIFY_PEAK, Some(jobs.len()));
    log.log(&format!("  verify: {} groups on {} workers (one decoder each)", jobs.len(), nproc));

    let (mut compared, mut exact) = (0usize, 0usize);
    let mut diff: Vec<DiffFrame> = Vec::new();
    let mut errors_by_job: Vec<Vec<String>> = vec![Vec::new(); jobs.len()];
    let stop = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| -> Result<()> {
        let (tx, rx) = mpsc::channel::<(usize, JobResult)>();
        let (jobs_r, next_r, stop_r, ctx_r) = (&jobs, &next, &stop, &ctx);
        for _ in 0..nproc {
            let tx = tx.clone();
            s.spawn(move || {
                loop {
                    let i = next_r.fetch_add(1, Ordering::SeqCst);
                    if i >= jobs_r.len() || stop_r.load(Ordering::SeqCst) {
                        break;
                    }
                    let r = run_job(ctx_r, &jobs_r[i]);
                    if tx.send((i, r)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending: BTreeMap<usize, JobResult> = BTreeMap::new();
        let mut emit = 0usize;
        let mut done = 0usize;
        let mut n_diff = 0usize; // differing frames of all finished groups
        for (i, r) in rx {
            done += 1;
            compared += r.compared;
            exact += r.exact;
            n_diff += r.differing.len();
            errors_by_job[i] = r.errors.clone();
            pending.insert(i, r);
            // release finished groups in order, so the export is in display order
            while let Some(r) = pending.remove(&emit) {
                emit += 1;
                for (df, hits) in r.differing {
                    if let Some(m) = mb.as_mut() {
                        let key = (df.period, df.poc);
                        let out_display = out_disp_of.get(&key).copied().unwrap_or(usize::MAX);
                        let f = MbFrame { frame: df.clone(), out_display, hits };
                        if let Err(e) = (m.sink)(&f) {
                            stop.store(true, Ordering::SeqCst);
                            return Err(e);
                        }
                    }
                    diff.push(df);
                }
            }
            if done % 10 == 0 || done == jobs.len() {
                log.log(&format!("  verify: {done}/{} groups, {compared} frames compared, {n_diff} differ", jobs.len()));
            }
        }
        Ok(())
    })?;
    diff.sort_by_key(|d| d.display); // stable
    let peak = system::peak_rss() as f64 / (1u64 << 20) as f64;
    Ok(VerifyResult {
        frames_output: out_pics.len(),
        frames_compared: compared,
        bit_exact: exact,
        differing: diff.len(),
        worst_difference: diff.iter().map(|d| d.worst).max().unwrap_or(0),
        mbs_ge3: diff.iter().map(|d| d.mbs_ge3).sum(),
        mbs_differing: diff.iter().map(|d| d.mbs_diff).sum(),
        worst_luma: diff.iter().map(|d| d.worst_luma).max().unwrap_or(0),
        mbs_ge3_luma: diff.iter().map(|d| d.mbs_ge3_luma).sum(),
        errors: errors_by_job.into_iter().flatten().collect(),
        workers: nproc,
        worker_peak_rss_mb: peak.round_ties_even() as u64,
        differing_frames: diff,
    })
}

// ---------------------------------------------------------------- --mb-report

/// What the `--mb-report` header says about the run.
pub struct MbReportMeta {
    pub input: String,
    pub output: Option<String>,
    pub fps_in: Ratio,
    pub fps_out: Ratio,
    pub mb_w: usize,
    pub mb_h: usize,
    /// left, right, top, bottom in luma pixels.
    pub crop: [usize; 4],
    pub threshold: u8,
}

/// Per-frame facts only the pipeline knows.
pub struct MbFrameInfo<'a> {
    /// The original's picture type ('I', 'P', 'B').
    pub typ: &'a str,
    pub idr: bool,
    /// A leading B picture of a rebuilt pair (reads I2 where it read K).
    pub rebuilt_leading_b: bool,
    /// After a dropped reference twin, up to the end of its IDR period.
    pub hard_drop_neighbourhood: bool,
}

#[derive(Serialize)]
struct Fps {
    exact: String,
    float: f64,
}

#[derive(Serialize)]
struct MbReportHeader {
    tool: &'static str,
    version: &'static str,
    input: String,
    output: Option<String>,
    input_fps: Fps,
    output_fps: Fps,
    frame_size_mbs: [usize; 2],
    frame_size_px: [usize; 2],
    /// left, right, top, bottom in luma pixels
    crop: [usize; 4],
    visible_size_px: [usize; 2],
    threshold: u8,
    /// the fields of every entry of `mbs`; `visible` is 1 when the macroblock
    /// overlaps the crop window
    mb_fields: [&'static str; 5],
}

#[derive(Serialize)]
struct MbFrameEntry<'a> {
    out_frame: usize,
    out_time_s: f64,
    in_frame: usize,
    in_time_s: f64,
    in_period: usize,
    in_poc: i32,
    #[serde(rename = "type")]
    typ: &'a str,
    idr: bool,
    rebuilt_leading_b: bool,
    hard_drop_neighbourhood: bool,
    worst: u32,
    worst_luma: u32,
    mbs_diff: usize,
    mbs_ge_threshold: usize,
    mbs: Vec<(u16, u16, u8, u8, u8)>,
}

/// Streams the `--mb-report` JSON file: a header, then one line per
/// differing frame as it is released by `verify`, then a count.
pub struct MbReportWriter {
    w: BufWriter<File>,
    path: PathBuf,
    fps_in: Ratio,
    fps_out: Ratio,
    n: usize,
}

impl MbReportWriter {
    pub fn create(path: &Path, meta: &MbReportMeta) -> Result<MbReportWriter> {
        let f = File::create(path).map_err(|e| anyhow!("cannot create {}: {e}", path.display()))?;
        let mut w = BufWriter::new(f);
        let (mw, mh) = (meta.mb_w, meta.mb_h);
        let [cl, cr, ct, cb] = meta.crop;
        let h = MbReportHeader {
            tool: "frame-dedup",
            version: crate::VERSION,
            input: meta.input.clone(),
            output: meta.output.clone(),
            input_fps: Fps { exact: meta.fps_in.to_string(), float: meta.fps_in.to_f64() },
            output_fps: Fps { exact: meta.fps_out.to_string(), float: meta.fps_out.to_f64() },
            frame_size_mbs: [mw, mh],
            frame_size_px: [mw * 16, mh * 16],
            crop: meta.crop,
            visible_size_px: [(mw * 16).saturating_sub(cl + cr), (mh * 16).saturating_sub(ct + cb)],
            threshold: meta.threshold,
            mb_fields: ["mbx", "mby", "max_diff", "luma_diff", "visible"],
        };
        let mut text = Vec::new();
        {
            let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
            let mut ser = serde_json::Serializer::with_formatter(&mut text, fmt);
            h.serialize(&mut ser)?;
        }
        // drop the closing brace: "frames" and "frames_listed" follow
        while text.last().is_some_and(|&c| c == b'}' || c == b'\n') {
            text.pop();
        }
        w.write_all(&text)?;
        w.write_all(b",\n \"frames\": [")?;
        Ok(MbReportWriter { w, path: path.to_path_buf(), fps_in: meta.fps_in, fps_out: meta.fps_out, n: 0 })
    }

    /// Append one frame (hits are already sorted worst first).
    pub fn frame(&mut self, f: &MbFrame, info: &MbFrameInfo) -> Result<()> {
        let t = |frame: usize, fps: Ratio| py_round(frame as f64 * fps.den as f64 / fps.num as f64, 6);
        let e = MbFrameEntry {
            out_frame: f.out_display,
            out_time_s: t(f.out_display, self.fps_out),
            in_frame: f.frame.display,
            in_time_s: t(f.frame.display, self.fps_in),
            in_period: f.frame.period,
            in_poc: f.frame.poc,
            typ: info.typ,
            idr: info.idr,
            rebuilt_leading_b: info.rebuilt_leading_b,
            hard_drop_neighbourhood: info.hard_drop_neighbourhood,
            worst: f.frame.worst,
            worst_luma: f.frame.worst_luma,
            mbs_diff: f.frame.mbs_diff,
            mbs_ge_threshold: f.hits.len(),
            mbs: f.hits.iter().map(|h| (h.x, h.y, h.max, h.luma, h.visible as u8)).collect(),
        };
        self.w.write_all(if self.n == 0 { b"\n  " } else { b",\n  " })?;
        serde_json::to_writer(&mut self.w, &e)?;
        self.n += 1;
        Ok(())
    }

    /// Close the file; returns the number of frames listed.
    pub fn finish(mut self) -> Result<usize> {
        write!(self.w, "\n ],\n \"frames_listed\": {}\n}}\n", self.n)?;
        self.w.flush().map_err(|e| anyhow!("cannot write {}: {e}", self.path.display()))?;
        Ok(self.n)
    }
}

/// `frame-dedup dev maxdiff <file.h264> <mb_w> <mb_h> <cl> <cr> <ct> <cb> <i> <j>`: decode frames
/// `0..=max(i, j)` of the file (from its start, which must be an IDR) and print the
/// canonical JSON of the macroblock difference maps between frames `i` and `j`.
pub fn dev_maxdiff(args: &[String]) -> anyhow::Result<()> {
    use anyhow::anyhow;
    let n = |i: usize| -> anyhow::Result<usize> { Ok(args.get(i).ok_or_else(|| anyhow!("missing argument {i}"))?.parse()?) };
    let path = std::path::PathBuf::from(args.first().ok_or_else(|| anyhow!("missing file"))?);
    let (mb_w, mb_h) = (n(1)?, n(2)?);
    let crop = [n(3)? as u32, n(4)? as u32, n(5)? as u32, n(6)? as u32];
    let (i, j) = (n(7)?, n(8)?);
    let fs = mb_w * mb_h * 384;
    let len = std::fs::metadata(&path)?.len() as usize;
    let mut frames = Vec::new();
    for (k, f) in crate::decode::iter_frames(&path, &[(0, len)], fs, crate::system::FFMPEG_THREADS)?.enumerate() {
        frames.push(f?);
        if k == i.max(j) {
            break;
        }
    }
    let (a, b) = (frames.get(i).ok_or_else(|| anyhow!("no frame {i}"))?, frames.get(j).ok_or_else(|| anyhow!("no frame {j}"))?);
    let sps = Sps { mb_w, mb_h, crop, ..test_sps_base() };
    let (m, ly) = mb_maxdiff(a, b, mb_w, mb_h);
    let vis = visible_maxdiff(a, b, &sps);
    println!("{}", serde_json::json!({"mb": m, "luma": ly, "visible": vis}));
    Ok(())
}

/// `frame-dedup dev verify <orig.h264> <out.h264> <inputs.json> [--workers N] [--mb-report FILE]
/// [--mb-threshold N]`: run `verify` on two Annex-B streams and print the result as JSON.
/// `inputs.json` (written by `python -m port.oracle verify-inputs`) holds `orig_of_out` (the
/// original decode index of every output picture), and for the export `fps_in` / `fps_out`
/// as "num/den".
pub fn dev_verify(args: &[String]) -> anyhow::Result<()> {
    let mut pos = Vec::new();
    let (mut workers, mut mb_path, mut thr) = (8usize, None::<PathBuf>, 8u8);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--workers" => workers = it.next().ok_or_else(|| anyhow!("--workers needs a value"))?.parse()?,
            "--mb-report" => mb_path = Some(PathBuf::from(it.next().ok_or_else(|| anyhow!("--mb-report needs a value"))?)),
            "--mb-threshold" => thr = it.next().ok_or_else(|| anyhow!("--mb-threshold needs a value"))?.parse()?,
            _ => pos.push(a.clone()),
        }
    }
    let [orig, out, inputs] = &pos[..] else { bail!("usage: dev verify <orig.h264> <out.h264> <inputs.json> [options]") };
    let (orig, out) = (PathBuf::from(orig), PathBuf::from(out));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(inputs)?)?;
    let orig_of_out: Option<Vec<usize>> = match v.get("orig_of_out") {
        Some(x) if !x.is_null() => Some(serde_json::from_value(x.clone())?),
        _ => None,
    };
    let ratio = |k: &str, dflt: Ratio| -> anyhow::Result<Ratio> {
        let Some(s) = v.get(k).and_then(|x| x.as_str()) else { return Ok(dflt) };
        let (n, d) = s.split_once('/').unwrap_or((s, "1"));
        Ok(Ratio::new(n.parse()?, d.parse()?))
    };
    let (fps_in, fps_out) = (ratio("fps_in", Ratio::new(30000, 1001))?, ratio("fps_out", Ratio::new(24000, 1001))?);
    let (spss, _, opics) = crate::stream::load(&orig)?;
    let (_, _, npics) = crate::stream::load(&out)?;
    let sps = &spss[&0];
    let log = Logger::new(None)?;
    let by_disp: BTreeMap<usize, &Picture> = opics.iter().map(|p| (p.disp, p)).collect();
    let mut writer = match &mb_path {
        Some(p) => Some(MbReportWriter::create(
            p,
            &MbReportMeta {
                input: orig.display().to_string(),
                output: Some(out.display().to_string()),
                fps_in,
                fps_out,
                mb_w: sps.mb_w,
                mb_h: sps.mb_h,
                crop: sps.crop.map(|c| 2 * c as usize),
                threshold: thr,
            },
        )?),
        None => None,
    };
    let mut sink = |f: &MbFrame| -> Result<()> {
        let p = by_disp[&f.frame.display];
        let info = MbFrameInfo { typ: p.typ().as_str(), idr: p.idr, rebuilt_leading_b: false, hard_drop_neighbourhood: false };
        writer.as_mut().unwrap().frame(f, &info)
    };
    let mb = mb_path.as_ref().map(|_| MbOptions { threshold: thr, sink: &mut sink });
    let res = verify(&orig, &out, &spss, &opics, &npics, workers, &log, orig_of_out.as_deref(), mb)?;
    if let Some(w) = writer {
        w.finish()?;
    }
    println!("{}", serde_json::to_string(&res)?);
    Ok(())
}

fn test_sps_base() -> Sps {
    Sps {
        profile_idc: 100,
        level_idc: 40,
        id: 0,
        chroma_format_idc: 1,
        separate_colour_plane: false,
        log2_max_frame_num: 4,
        poc_type: 0,
        log2_max_poc_lsb: 4,
        delta_pic_order_always_zero: false,
        max_num_ref_frames: 1,
        gaps_allowed: false,
        mb_w: 0,
        mb_h: 0,
        frame_mbs_only: true,
        mbaff: false,
        direct_8x8_inference: true,
        crop: [0; 4],
        vui: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(mb_w: usize, mb_h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        (0..mb_w * mb_h * 384)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect()
    }

    /// Straightforward reference: the numpy algorithm, element by element.
    fn naive(a: &[u8], b: &[u8], mb_w: usize, mb_h: usize, crop: [usize; 4]) -> (Vec<u8>, Vec<u8>) {
        let (w, h) = (mb_w * 16, mb_h * 16);
        let mut all = vec![0u8; mb_w * mb_h];
        let mut luma = all.clone();
        let planes = [(0, w, h, 1usize, 16usize), (w * h, w / 2, h / 2, 2, 8), (w * h * 5 / 4, w / 2, h / 2, 2, 8)];
        for (pi, (off, pw, ph, s, bs)) in planes.into_iter().enumerate() {
            for y in 0..ph {
                for x in 0..pw {
                    if y < crop[2] / s || y >= ph - crop[3] / s || x < crop[0] / s || x >= pw - crop[1] / s {
                        continue;
                    }
                    let d = a[off + y * pw + x].abs_diff(b[off + y * pw + x]);
                    let i = (y / bs) * mb_w + x / bs;
                    all[i] = all[i].max(d);
                    if pi == 0 {
                        luma[i] = luma[i].max(d);
                    }
                }
            }
        }
        (all, luma)
    }

    /// `cargo test --profile fastrel -- --ignored --nocapture timing`
    #[test]
    #[ignore]
    fn timing() {
        let (mw, mh) = (120, 68);
        let (a, b) = (frame(mw, mh, 1), frame(mw, mh, 2));
        let sps = test_sps(mw, mh, [0, 0, 0, 4]);
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(mb_maxdiff(&a, &b, mw, mh));
        }
        println!("mb_maxdiff: {:?} per 1080p frame", t.elapsed() / 100);
        let t = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(visible_maxdiff(&a, &b, &sps));
        }
        println!("visible_maxdiff: {:?} per 1080p frame", t.elapsed() / 100);
    }

    #[test]
    fn matches_naive() {
        let (mw, mh) = (5, 4);
        let (a, b) = (frame(mw, mh, 1), frame(mw, mh, 2));
        let (all, ly) = mb_maxdiff(&a, &b, mw, mh);
        assert_eq!((all, ly), naive(&a, &b, mw, mh, [0; 4]));
        for crop in [[0, 0, 0, 4], [1, 2, 3, 4], [0, 3, 0, 0], [8, 0, 8, 0]] {
            let sps = test_sps(mw, mh, crop.map(|c| c as u32));
            let want = naive(&a, &b, mw, mh, crop.map(|c| 2 * c)).0;
            assert_eq!(visible_maxdiff(&a, &b, &sps), want, "crop {crop:?}");
        }
    }

    #[test]
    fn hits_sorted_worst_first_and_visibility() {
        let all = [9, 0, 8, 20, 8, 7];
        let luma = [9, 0, 1, 5, 8, 7];
        let h = mb_hits(&all, &luma, 3, 2, 8, [0, 0, 0, 0]);
        let got: Vec<(u16, u16, u8)> = h.iter().map(|m| (m.x, m.y, m.max)).collect();
        assert_eq!(got, vec![(0, 1, 20), (0, 0, 9), (1, 1, 8), (2, 0, 8)]);
        assert!(h.iter().all(|m| m.visible));
        // a 16-pixel crop at the right and bottom hides the last column and row; a partial crop does not
        let h = mb_hits(&all, &luma, 3, 2, 8, [0, 16, 0, 16]);
        let vis: Vec<(u16, u16, bool)> = h.iter().map(|m| (m.x, m.y, m.visible)).collect();
        assert_eq!(vis, vec![(0, 1, false), (0, 0, true), (1, 1, false), (2, 0, false)]);
        assert!(mb_hits(&all, &luma, 3, 2, 8, [0, 0, 0, 8]).iter().all(|m| m.visible));
    }

    #[test]
    fn mb_report_file_is_valid_json() {
        let path = std::env::temp_dir().join(format!("fd_mbreport_test_{}.json", std::process::id()));
        let meta = MbReportMeta {
            input: "in.mkv".into(),
            output: None,
            fps_in: Ratio::new(30000, 1001),
            fps_out: Ratio::new(24000, 1001),
            mb_w: 120,
            mb_h: 68,
            crop: [0, 0, 0, 8],
            threshold: 8,
        };
        let mut w = MbReportWriter::create(&path, &meta).unwrap();
        let df = DiffFrame { display: 10, period: 1, poc: 4, worst: 12, mbs_ge3: 3, mbs_diff: 5, worst_luma: 11, mbs_ge3_luma: 2 };
        let hits = vec![MbHit { x: 3, y: 4, max: 12, luma: 11, visible: true }];
        let info = MbFrameInfo { typ: "B", idr: false, rebuilt_leading_b: false, hard_drop_neighbourhood: true };
        w.frame(&MbFrame { frame: df.clone(), out_display: 8, hits: hits.clone() }, &info).unwrap();
        w.frame(&MbFrame { frame: df, out_display: 9, hits: Vec::new() }, &info).unwrap();
        assert_eq!(w.finish().unwrap(), 2);
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(v["frames_listed"], 2);
        assert_eq!(v["frames"][0]["mbs"], serde_json::json!([[3, 4, 12, 11, 1]]));
        assert_eq!(v["frames"][0]["out_frame"], 8);
        assert_eq!(v["frames"][0]["hard_drop_neighbourhood"], true);
        assert_eq!(v["frames"][1]["mbs"], serde_json::json!([]));
        assert_eq!(v["input_fps"]["exact"], "30000/1001");
        assert_eq!(v["visible_size_px"], serde_json::json!([1920, 1080]));
        assert_eq!(v["frames"][0]["out_time_s"], 0.333667);
    }

    fn test_sps(mb_w: usize, mb_h: usize, crop: [u32; 4]) -> Sps {
        Sps { mb_w, mb_h, crop, ..test_sps_base() }
    }
}

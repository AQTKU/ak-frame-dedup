//! Repair of hard drops: exact I_PCM repair, and measured drift repair in
//! explicit mode. Port of `frame_dedup/repair.py`.
//!
//! When a reference twin D is dropped, the decoder's default reference lists
//! change for the pictures that follow it until D would have left them: an
//! index that pointed at D now points at the kept twin K, and with more than
//! one active reference later indices can shift as well (e.g. [D, K] ->
//! [K, older]).
//!
//! Which pictures are affected is found from header-only ref lists of the
//! original and the deduplicated stream ([`crate::stream::new_lists`]). Only
//! those pictures, and B pictures whose co-located picture is one of them, are
//! parsed. For each, motion is derived twice from the same macroblock records:
//! in the original context and in the new one. Every 4x4 block whose reference
//! picture or motion vector differs is a direct reader of changed data; its
//! macroblock is replaced by an I_PCM copy of the original decoded pixels
//! ([`crate::patch`]).
//!
//! Work is split by IDR period and run in parallel. Unlike the Python, the
//! workers are threads of one process that share the memory-mapped streams and
//! the parsed pictures' slices; results are collected in task (period) order,
//! so the output does not depend on the worker count or on completion order.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use serde_json::{Value, json};

use crate::decode::{FrameIter, decode_pocs, frame_size, iter_frames};
use crate::explicit::Sigma;
use crate::h264hdr::{Picture, Ppss, SliceType, Sps, Spss};
use crate::mv::{Motion, derive_motion};
use crate::patch::{
    LruStore, MapStore, Nal, PARSED_CAP, ParsedPic, ParsedStore, PatchCache, PatchCtx, PatchResult, PatchStats, patch_pictures, pcm_mb,
    pcm_record, write_segment,
};
use crate::picture::{Parsed, decode_picture};
use crate::stream::{new_lists, period_ranges};
use crate::system::{FFMPEG_THREADS, Logger, REPAIR_PEAK, REPAIR_PEAK_EXPLICIT, peak_rss, pool_size};
use crate::verify::{mb_maxdiff, visible_maxdiff};

// Explicit mode (see explicit.rs) repairs by measurement: a dropped twin D
// and its substitute K are two encodes of the same frame, so reads of K
// instead of D shift predictions by encode noise. Macroblocks whose decoded
// result moves from the original by DRIFT_THRESHOLD or more (max over Y, U,
// V, visible area) become I_PCM copies of the original, earliest reference
// picture first, until nothing in the IDR period crosses the threshold, with
// at most DRIFT_MAX_FRACTION of any picture patched.
pub const DRIFT_THRESHOLD: u8 = 16;
pub const DRIFT_MAX_FRACTION: f64 = 0.02;
pub const DRIFT_MAX_ROUNDS: usize = 12;
/// After these rounds, each round patches every picture still over.
pub const DRIFT_SEQUENTIAL_ROUNDS: usize = 6;

/// Stack size of the worker threads (the CABAC codec and the motion
/// derivation keep large arrays on the stack).
const WORKER_STACK: usize = 32 << 20;

const NONE: usize = usize::MAX;

// ---------------------------------------------------------------------------
// candidates and tasks
// ---------------------------------------------------------------------------

/// Original decode indices of kept pictures whose reference context changes.
/// Returns (candidates, new lists by orig dec). With `sigma` (explicit mode)
/// the new lists are the original ones with every dropped reference replaced
/// by its substitute (a dropped reference without one becomes `usize::MAX`,
/// Python's `None`).
#[allow(clippy::type_complexity)]
pub fn find_candidates(
    spss: &Spss,
    pics: &[Picture],
    drop: &HashSet<usize>,
    sigma: Option<&Sigma>,
) -> Result<(Vec<usize>, BTreeMap<usize, [Vec<usize>; 2]>)> {
    let nl: BTreeMap<usize, [Vec<usize>; 2]> = if let Some(sigma) = sigma {
        pics.iter()
            .filter(|p| !drop.contains(&p.dec))
            .map(|p| {
                let l = [0, 1].map(|l| {
                    p.lists[l]
                        .iter()
                        .map(|x| match sigma.get(x) {
                            Some(Some(k)) => *k,
                            Some(None) => NONE,
                            None => *x,
                        })
                        .collect::<Vec<_>>()
                });
                (p.dec, l)
            })
            .collect()
    } else {
        new_lists(spss, pics, drop)?.into_iter().collect()
    };
    let mut cand: BTreeSet<usize> = nl.iter().filter(|(d, l)| **l != pics[**d].lists).map(|(d, _)| *d).collect();
    // Python iterates the dict in insertion (decode) order while adding to
    // the set it tests, so chains of co-located B pictures propagate
    for &d in nl.keys() {
        let p = &pics[d];
        if p.typ() == SliceType::B && !p.lists[1].is_empty() && cand.contains(&p.lists[1][0]) {
            cand.insert(d);
        }
    }
    Ok((cand.into_iter().collect(), nl))
}

/// One repair task: an IDR period of the new stream.
#[derive(Clone, Debug)]
pub struct Task {
    /// The IDR period of the new stream (Python: the task's `period`).
    pub period: usize,
    /// Exact mode: original decode indices of the candidate pictures of the period.
    pub cands: Vec<usize>,
    /// Exact mode: decode range of the period in the original stream.
    pub orig: Range<usize>,
    /// Decode range of the period in the new stream.
    pub new: Range<usize>,
    /// Explicit mode: the original IDR periods the new period covers.
    pub orig_periods: Vec<usize>,
}

/// One task per IDR period that holds candidates (exact mode).
pub fn build_tasks(pics: &[Picture], new_pics: &[Picture], cands: &[usize]) -> Result<Vec<Task>> {
    let mut by_period: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &d in cands {
        by_period.entry(pics[d].period).or_default().push(d);
    }
    let (oranges, nranges) = (period_ranges(pics), period_ranges(new_pics));
    let mut tasks = Vec::new();
    for (g, ds) in by_period {
        let new = nranges.get(g).cloned().ok_or_else(|| anyhow!("KeyError: {g}"))?;
        tasks.push(Task { period: g, cands: ds, orig: oranges[g].clone(), new, orig_periods: vec![g] });
    }
    Ok(tasks)
}

/// Explicit mode: one task per IDR period of the new stream that holds a
/// picture reading a substituted picture (anchor rebuilds merge periods, so a
/// task can span several original periods). Returns (tasks, candidate original
/// decode indices).
pub fn build_tasks_explicit(
    pics: &[Picture],
    new_pics: &[Picture],
    orig_of_new: &[usize],
    sigma: &Sigma,
) -> Result<(Vec<Task>, Vec<usize>)> {
    let subst: HashSet<usize> = sigma.iter().filter(|(_, k)| k.is_some()).map(|(d, _)| *d).collect();
    let nranges = period_ranges(new_pics);
    let mut tasks = Vec::new();
    let mut cands = Vec::new();
    for (g, nr) in nranges.iter().enumerate() {
        let readers: Vec<usize> = nr
            .clone()
            .filter(|&nd| pics[orig_of_new[nd]].lists.iter().any(|l| l.iter().any(|x| subst.contains(x))))
            .collect();
        if readers.is_empty() {
            continue;
        }
        cands.extend(readers.iter().map(|&nd| orig_of_new[nd]));
        let og: BTreeSet<usize> = nr.clone().map(|nd| pics[orig_of_new[nd]].period).collect();
        tasks.push(Task { period: g, cands: Vec::new(), orig: 0..0, new: nr.clone(), orig_periods: og.into_iter().collect() });
    }
    Ok((tasks, cands))
}

/// 4x4 blocks (gx, gy) whose reference picture or motion vector differs
/// between the original and the new reference context, in row-major order.
pub fn changed_blocks(m_orig: &Motion, m_new: &Motion, orig_of_new: &[usize], w: usize, h: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for gy in 0..h * 4 {
        for gx in 0..w * 4 {
            let i = m_orig.idx(gx, gy);
            for l in 0..2 {
                let ro = m_orig.refpic[l][i];
                let rn = m_new.refpic[l][i];
                let rn = if rn < 0 { -1 } else { orig_of_new[rn as usize] as i32 };
                if ro != rn || (ro >= 0 && m_orig.mv[l][i] != m_new.mv[l][i]) {
                    out.push((gx, gy));
                    break;
                }
            }
        }
    }
    out
}

/// Macroblocks a 4x4 block's luma and chroma prediction can read (6-tap /
/// bilinear margins, generously rounded): (y0, y1, x0, x1) in MBs, exclusive
/// ends.
pub fn mb_rect(gx: usize, gy: usize, mv: [i16; 2], w: usize, h: usize) -> (usize, usize, usize, usize) {
    let x = gx as i64 * 4 + (mv[0] as i64 >> 2);
    let y = gy as i64 * 4 + (mv[1] as i64 >> 2);
    // out-of-picture reads replicate the edge samples: clamp both ends
    let cx = |v: i64| v.clamp(0, w as i64 * 16 - 1) as usize;
    let cy = |v: i64| v.clamp(0, h as i64 * 16 - 1) as usize;
    let (x0, x1) = (cx(x - 4), cx(x + 3 + 5));
    let (y0, y1) = (cy(y - 4), cy(y + 3 + 5));
    (y0 / 16, y1 / 16 + 1, x0 / 16, x1 / 16 + 1)
}

/// Answers whether a changed block's prediction is provably unchanged, from
/// the original decoded pixels of the reference pictures.
pub struct PixelOracle<'a> {
    frames: HashMap<i32, Vec<u8>>,
    pics: &'a [Picture],
    w: usize,
    h: usize,
    eq: HashMap<(usize, usize), Vec<bool>>,
}

impl<'a> PixelOracle<'a> {
    /// `frames`: decoded original frames by POC; `pics`: the original pictures
    /// (Python's `poc_of`).
    pub fn new(frames: HashMap<i32, Vec<u8>>, pics: &'a [Picture], w: usize, h: usize) -> Self {
        PixelOracle { frames, pics, w, h, eq: HashMap::new() }
    }

    fn same(&mut self, a: usize, b: usize, rect: (usize, usize, usize, usize)) -> Result<bool> {
        if a == b {
            return Ok(true);
        }
        let key = (a.min(b), a.max(b));
        if !self.eq.contains_key(&key) {
            let fa = self.frames.get(&self.pics[a].poc).ok_or_else(|| anyhow!("KeyError: {}", self.pics[a].poc))?;
            let fb = self.frames.get(&self.pics[b].poc).ok_or_else(|| anyhow!("KeyError: {}", self.pics[b].poc))?;
            let (m, _) = mb_maxdiff(fa, fb, self.w, self.h);
            self.eq.insert(key, m.iter().map(|&v| v == 0).collect());
        }
        let (y0, y1, x0, x1) = rect;
        if y0 >= y1 || x0 >= x1 {
            bail!("AssertionError: {rect:?}");
        }
        let eq = &self.eq[&key];
        Ok((y0..y1).all(|y| (x0..x1).all(|x| eq[y * self.w + x])))
    }

    /// Whether the changed 4x4 block (gx, gy) predicts the same samples in the
    /// new context as in the original one.
    pub fn harmless(&mut self, gx: usize, gy: usize, mo: &Motion, mn: &Motion, orig_of_new: &[usize]) -> Result<bool> {
        let (w, h) = (self.w, self.h);
        let i = mo.idx(gx, gy);
        let mut used: Vec<(usize, [i16; 2])> = Vec::new();
        let mut changed = false;
        for l in 0..2 {
            let ro = mo.refpic[l][i];
            let rn = mn.refpic[l][i];
            let rn = if rn < 0 { -1 } else { orig_of_new[rn as usize] as i32 };
            if (ro < 0) != (rn < 0) {
                return Ok(false);
            }
            if ro < 0 {
                continue;
            }
            let mv = mo.mv[l][i];
            if mv != mn.mv[l][i] {
                return Ok(false);
            }
            if ro != rn {
                changed = true;
                if !self.same(ro as usize, rn as usize, mb_rect(gx, gy, mv, w, h))? {
                    return Ok(false);
                }
            }
            used.push((ro as usize, mv));
        }
        if changed && used.len() == 2 {
            // implicit bi-prediction weights follow the reference POCs, so the
            // result is only unchanged if both predictions are identical
            let ((r0, mv0), (r1, mv1)) = (used[0], used[1]);
            if mv0 != mv1 || !self.same(r0, r1, mb_rect(gx, gy, mv0, w, h))? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// worker pool
// ---------------------------------------------------------------------------

/// Run `work` on every task with `nproc` threads pulling from a shared queue.
/// `done(n, result)` is called on the calling thread as each task completes
/// (`n` counts completions from 0). Returns the results in task order. After
/// the first error no new task is started, and the error of the lowest task
/// index is returned.
pub(crate) fn run_pool<T: Sync, R: Send>(
    nproc: usize,
    tasks: &[T],
    work: &(dyn Fn(&T) -> Result<R> + Sync),
    mut done: impl FnMut(usize, &R),
) -> Result<Vec<R>> {
    let n = tasks.len();
    let next = AtomicUsize::new(0);
    let abort = AtomicBool::new(false);
    let (tx, rx) = mpsc::channel::<(usize, Result<R>)>();
    let mut slots: Vec<Option<R>> = (0..n).map(|_| None).collect();
    let mut first_err: Option<(usize, anyhow::Error)> = None;
    std::thread::scope(|s| -> Result<()> {
        for _ in 0..nproc.max(1) {
            let tx = tx.clone();
            let (next, abort) = (&next, &abort);
            std::thread::Builder::new().stack_size(WORKER_STACK).spawn_scoped(s, move || {
                loop {
                    if abort.load(Ordering::Relaxed) {
                        break;
                    }
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    let r = work(&tasks[i]);
                    if r.is_err() {
                        abort.store(true, Ordering::Relaxed);
                    }
                    if tx.send((i, r)).is_err() {
                        break;
                    }
                }
            })?;
        }
        drop(tx);
        let mut count = 0;
        while let Ok((i, r)) = rx.recv() {
            match r {
                Ok(v) => {
                    done(count, &v);
                    count += 1;
                    slots[i] = Some(v);
                }
                Err(e) => {
                    if first_err.as_ref().is_none_or(|(j, _)| i < *j) {
                        first_err = Some((i, e));
                    }
                }
            }
        }
        Ok(())
    })?;
    if let Some((_, e)) = first_err {
        return Err(e);
    }
    slots.into_iter().map(|s| s.ok_or_else(|| anyhow!("a worker thread stopped before finishing its task"))).collect()
}

/// A temporary file removed when dropped (Python's `finally: os.remove`).
pub(crate) struct TmpFile(pub PathBuf);

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub(crate) fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub(crate) fn map_file(path: &Path) -> Result<Mmap> {
    let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    Ok(unsafe { Mmap::map(&file)? })
}

// ---------------------------------------------------------------------------
// the tasks
// ---------------------------------------------------------------------------

struct Ctx<'a> {
    orig_path: &'a Path,
    orig_es: &'a [u8],
    new_es: &'a [u8],
    spss: &'a Spss,
    ppss: &'a Ppss,
    pics: &'a [Picture],
    new_pics: &'a [Picture],
    /// Original decode index of every picture of the new stream.
    orig_of_new: &'a [usize],
    /// New decode index of every original picture (`NONE` if dropped).
    new_of: Vec<usize>,
    sigma: Option<&'a Sigma>,
    work: &'a Path,
    orig_ranges: Vec<Range<usize>>,
}

impl Ctx<'_> {
    fn sps(&self) -> Result<&Sps> {
        self.spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))
    }
}

struct TaskResult {
    period: usize,
    repl: BTreeMap<usize, Nal>,
    stats: BTreeMap<usize, PatchStats>,
    sniff: Vec<Value>,
    rounds: Option<usize>,
    patches: BTreeMap<usize, usize>,
    seconds: f64,
}

/// A 128-bit hash of a frame (the seen-frame cache of the measured repair
/// only needs equality of identical decodes).
fn frame_key(data: &[u8]) -> (u64, u64) {
    const S0: u64 = 0x243F_6A88_85A3_08D3;
    const S1: u64 = 0x1319_8A2E_0370_7344;
    const S2: u64 = 0xA409_3822_299F_31D0;
    let (mut a, mut b) = (S0 ^ data.len() as u64, S1);
    let mut chunks = data.chunks_exact(16);
    for c in &mut chunks {
        let x = u64::from_le_bytes(c[..8].try_into().unwrap());
        let y = u64::from_le_bytes(c[8..].try_into().unwrap());
        let m = ((x ^ S2 ^ a) as u128).wrapping_mul((y ^ S0 ^ b) as u128);
        a = (m as u64) ^ b.rotate_left(17);
        b = ((m >> 64) as u64) ^ a.rotate_left(41);
    }
    let mut tail = [0u8; 16];
    let rest = chunks.remainder();
    tail[..rest.len()].copy_from_slice(rest);
    let x = u64::from_le_bytes(tail[..8].try_into().unwrap());
    let y = u64::from_le_bytes(tail[8..].try_into().unwrap());
    let m = ((x ^ S2 ^ a) as u128).wrapping_mul((y ^ S1 ^ b ^ rest.len() as u64) as u128);
    ((m as u64) ^ a, ((m >> 64) as u64) ^ b)
}

/// (original dec, frame) of the original pictures of `periods`, in (period,
/// POC) order, decoding one period at a time (Python `_orig_frames`).
struct OrigFrames<'a> {
    cx: &'a Ctx<'a>,
    periods: &'a [usize],
    next_period: usize,
    cur: Option<(FrameIter, Vec<usize>, usize)>,
    fs: usize,
}

impl OrigFrames<'_> {
    fn next(&mut self) -> Result<Option<(usize, Vec<u8>)>> {
        loop {
            if self.cur.is_none() {
                let Some(&g) = self.periods.get(self.next_period) else {
                    return Ok(None);
                };
                self.next_period += 1;
                let r = self.cx.orig_ranges[g].clone();
                let pics = self.cx.pics;
                let mut by_poc: Vec<usize> = r.clone().collect();
                by_poc.sort_by_key(|&d| pics[d].poc);
                let it = iter_frames(self.cx.orig_path, &[(pics[r.start].au_start, pics[r.end - 1].au_end)], self.fs, FFMPEG_THREADS)?;
                self.cur = Some((it, by_poc, 0));
            }
            let (it, by_poc, pos) = self.cur.as_mut().expect("just set");
            // zip(by_poc, frames): the picture list is consulted first, so the
            // decoder is not read (or checked) once it is exhausted
            if *pos >= by_poc.len() {
                self.cur = None;
                continue;
            }
            match it.next() {
                None => {
                    self.cur = None;
                    continue;
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(fr)) => {
                    let d = by_poc[*pos];
                    *pos += 1;
                    return Ok(Some((d, fr)));
                }
            }
        }
    }
}

/// The records of original picture `d` and its motion in the original
/// context (Python `run_task_explicit.orig_motion`).
fn orig_motion(cx: &Ctx, d: usize) -> Result<(Parsed, Motion)> {
    let sps = cx.sps()?;
    let po = &cx.pics[d];
    let dd = decode_picture(cx.orig_es, cx.spss, cx.ppss, po)?;
    let col = if po.typ() == SliceType::B {
        Some(orig_motion(cx, *po.lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?)?.1)
    } else {
        None
    };
    let m = derive_motion(po, &dd.mbs, sps.mb_w, sps.mb_h, cx.pics, col.as_ref()).with_context(|| format!("dec {d}"))?;
    Ok((dd, m))
}

/// Records of B picture `nd` of the new stream under its new header, with the
/// motion it had in the original (its co-located picture there was D).
fn keep_original_motion(cx: &Ctx, nd: usize) -> Result<ParsedPic> {
    let (dd, mo) = orig_motion(cx, cx.orig_of_new[nd])?;
    let ns = &cx.new_pics[nd].slices[0];
    let nal = &cx.new_es[ns.offset..ns.offset + ns.size];
    let mut head = crate::bits::unescape(&nal[1..]);
    head.truncate(ns.data_bit / 8);
    let parsed = Parsed { nal_header: nal[0], nal_len: nal.len(), sl: ns.clone(), head, mbs: dd.mbs, end_bit: dd.end_bit };
    Ok(ParsedPic { dd: parsed, motion: Arc::new(mo) })
}

/// Explicit mode: kept pictures keep their reference indices and mvs; only
/// reads of a dropped picture D now read its substitute K. B pictures whose
/// co-located picture was D keep their original motion: direct blocks whose
/// derived motion would change are made explicit. Then the period is decoded
/// and compared with the original, and drifting MBs are patched (see
/// [`DRIFT_THRESHOLD`]).
///
/// Memory stays bounded however many IDR periods anchor rebuilds merged: the
/// original is decoded again in lockstep every round, only the target frames
/// of patched pictures are kept, and parsed pictures live in an LRU.
fn run_task_explicit(cx: &Ctx, task: &Task) -> Result<TaskResult> {
    let t0 = Instant::now();
    let sigma = cx.sigma.ok_or_else(|| anyhow!("explicit repair needs sigma"))?;
    let sps = cx.sps()?;
    let (w, h) = (sps.mb_w, sps.mb_h);
    let fs = frame_size(sps);
    let (pics, new_pics, orig_of_new) = (cx.pics, cx.new_pics, cx.orig_of_new);
    let new_range = task.new.clone();

    let force: BTreeSet<usize> = new_range
        .clone()
        .filter(|&nd| {
            let o = &pics[orig_of_new[nd]];
            o.typ() == SliceType::B && o.lists[1].first().is_some_and(|c| sigma.get(c).is_some_and(|v| v.is_some()))
        })
        .collect();
    let mut store = LruStore::new(PARSED_CAP, force.iter().copied(), |nd| keep_original_motion(cx, nd));

    let mut order: Vec<usize> = new_range.clone().collect();
    order.sort_by_key(|&nd| new_pics[nd].poc); // stable, like sorted(key=poc)
    let budget = ((DRIFT_MAX_FRACTION * w as f64 * h as f64) as usize).max(1);
    let tmp = TmpFile(cx.work.join(format!("drift_{}.h264", task.period)));

    type Samples = Box<[u8; 384]>;
    let mut patches: BTreeMap<usize, Vec<(u16, u16)>> = BTreeMap::new();
    let mut target: HashMap<(usize, u16, u16), Samples> = HashMap::new();
    let mut cache = PatchCache::default();
    let mut seen: HashMap<usize, ((u64, u64), Vec<u8>)> = HashMap::new();
    let mut capped: BTreeSet<usize> = BTreeSet::new();
    let mut worst: HashMap<usize, u8> = HashMap::new();
    let mut rounds = 0usize;
    let mut viol: BTreeMap<usize, Vec<(u8, u16, u16)>>;
    let mut res: PatchResult;
    loop {
        let ctx = PatchCtx { es: cx.new_es, spss: cx.spss, ppss: cx.ppss, pics: new_pics, range: new_range.clone() };
        let targets = |dec: usize, x: u16, y: u16| -> Result<crate::records::Mb> {
            let s = target.get(&(dec, x, y)).ok_or_else(|| anyhow!("KeyError: no target for picture {dec} MB ({x}, {y})"))?;
            Ok(pcm_mb(x, y, s.clone()))
        };
        res = patch_pictures(&ctx, &patches, &targets, &mut store as &mut dyn ParsedStore, &force, Some(&mut cache), &mut |_| {})?;
        {
            let mut f = BufWriter::with_capacity(1 << 20, File::create(&tmp.0).with_context(|| format!("cannot create {}", tmp.0.display()))?);
            write_segment(cx.new_es, new_pics, new_range.start, new_range.end, &res.repl, &mut f)?;
            f.flush()?;
        }
        viol = BTreeMap::new();
        let mut keep: HashMap<usize, HashMap<(u16, u16), Samples>> = HashMap::new();
        let mut ogen = OrigFrames { cx, periods: &task.orig_periods, next_period: 0, cur: None, fs };
        let mut n_out = 0usize;
        {
            let size = std::fs::metadata(&tmp.0)?.len() as usize;
            let mut it = iter_frames(&tmp.0, &[(0, size)], fs, FFMPEG_THREADS)?;
            let mut fr = Vec::new();
            for &nd in &order {
                if !it.next_frame(&mut fr)? {
                    break;
                }
                n_out += 1;
                let want = orig_of_new[nd];
                let ofr = loop {
                    // the mapping is monotone: skip dropped originals
                    match ogen.next()? {
                        Some((d, f)) if d == want => break f,
                        Some(_) => {}
                        None => bail!("original picture {want} not found in decode order"),
                    }
                };
                // a frame decoded exactly as in the previous round needs no new diff
                let key = frame_key(&fr);
                if seen.get(&nd).map(|s| s.0) != Some(key) {
                    seen.insert(nd, (key, visible_maxdiff(&fr, &ofr, sps)));
                }
                let m = &seen[&nd].1;
                let done = patches.get(&nd).map(Vec::as_slice).unwrap_or(&[]);
                let mut bad: Vec<(u8, u16, u16)> = Vec::new();
                for (i, &v) in m.iter().enumerate() {
                    if v >= DRIFT_THRESHOLD {
                        let (x, y) = ((i % w) as u16, (i / w) as u16);
                        if done.binary_search(&(x, y)).is_err() {
                            bad.push((v, x, y));
                        }
                    }
                }
                worst.insert(nd, m.iter().copied().max().unwrap_or(0));
                if !bad.is_empty() {
                    bad.sort_by(|a, b| b.cmp(a)); // sorted(..., reverse=True)
                    // keep the original samples of the MBs that may be patched
                    // (not whole frames: a merged period can hold hundreds)
                    let room = budget as i64 - done.len() as i64;
                    let k = room.max(0) as usize;
                    let samples: HashMap<(u16, u16), Samples> = bad
                        .iter()
                        .take(k)
                        .map(|&(_, x, y)| ((x, y), pcm_record(&ofr, w, x as usize, y as usize).samples.expect("PCM record has samples")))
                        .collect();
                    keep.insert(nd, samples);
                    viol.insert(nd, bad);
                }
            }
            drop(it);
        }
        // let the original decoder finish cleanly
        while ogen.next()?.is_some() {}
        drop(ogen);
        if n_out != order.len() {
            bail!("period {}: decoded {n_out} of {} pictures", task.period, order.len());
        }
        rounds += 1;
        let todo: Vec<usize> = viol.keys().copied().filter(|nd| !capped.contains(nd)).collect();
        if todo.is_empty() || rounds > DRIFT_MAX_ROUNDS {
            break;
        }
        // early rounds patch in decode order up to the first reference
        // picture (what it fixes downstream is measured again next round);
        // later rounds patch everything still over the threshold
        for nd in todo {
            let have = patches.get(&nd).map_or(0, Vec::len);
            let room = budget as i64 - have as i64;
            let v = &viol[&nd];
            if v.len() as i64 > room {
                capped.insert(nd);
            }
            let mbs: Vec<(u16, u16)> = v.iter().take(room.max(0) as usize).map(|&(_, x, y)| (x, y)).collect();
            if !mbs.is_empty() {
                let mut all: BTreeSet<(u16, u16)> = patches.get(&nd).map(|p| p.iter().copied().collect()).unwrap_or_default();
                all.extend(mbs.iter().copied());
                patches.insert(nd, all.into_iter().collect());
                let kept = keep.get_mut(&nd).expect("kept samples for every violating picture");
                for &(x, y) in &mbs {
                    target.insert((nd, x, y), kept.remove(&(x, y)).expect("kept sample"));
                }
            }
            if new_pics[nd].ref_idc != 0 && rounds <= DRIFT_SEQUENTIAL_ROUNDS {
                break;
            }
        }
    }
    drop(tmp);
    let mut sniff = Vec::new();
    let mut all: BTreeSet<usize> = patches.keys().copied().collect();
    all.extend(viol.keys().copied());
    all.extend(force.iter().copied());
    for nd in all {
        let d = orig_of_new[nd];
        sniff.push(json!({"dec": d, "display": pics[d].disp, "type": pics[d].typ().as_str(),
                          "mbs": patches.get(&nd).map_or(0, Vec::len), "worst_after": worst.get(&nd).copied().unwrap_or(0),
                          "over_threshold_after": viol.get(&nd).map_or(0, Vec::len), "capped": capped.contains(&nd)}));
    }
    Ok(TaskResult {
        period: task.period,
        repl: res.repl,
        stats: res.stats,
        sniff,
        rounds: Some(rounds),
        patches: patches.iter().map(|(nd, v)| (*nd, v.len())).collect(),
        seconds: round1(t0.elapsed().as_secs_f64()),
    })
}

/// Exact repair of one IDR period: every macroblock holding a block whose
/// prediction is not provably unchanged becomes an I_PCM copy of the original
/// pixels.
fn run_task_exact(cx: &Ctx, task: &Task) -> Result<TaskResult> {
    let t0 = Instant::now();
    let sps = cx.sps()?;
    let (w, h) = (sps.mb_w, sps.mb_h);
    let (pics, new_pics, orig_of_new, new_of) = (cx.pics, cx.new_pics, cx.orig_of_new, &cx.new_of);
    let key_err = |k: usize| anyhow!("KeyError: {k}");

    let mut need: BTreeSet<usize> = task.cands.iter().copied().collect();
    for &d in &task.cands {
        if pics[d].typ() == SliceType::B {
            need.insert(*pics[d].lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?);
        }
    }
    let mut parsed = MapStore::default();
    let mut m_orig: HashMap<usize, Motion> = HashMap::new();
    let mut blocks: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    let cand_set: HashSet<usize> = task.cands.iter().copied().collect();
    for &d in &need {
        let po = &pics[d];
        let nd = new_of[d];
        if nd == NONE {
            return Err(key_err(d));
        }
        let pn = &new_pics[nd];
        if pn.poc != po.poc {
            bail!("AssertionError: POC of picture {d} changed");
        }
        let dd = decode_picture(cx.new_es, cx.spss, cx.ppss, pn)?;
        let t = po.typ();
        let (mo, mn);
        if t == SliceType::B {
            let col_o = *po.lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?;
            let col_n = *pn.lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?;
            if orig_of_new[col_n] != col_o {
                bail!("AssertionError: co-located picture changed");
            }
            let col_m = m_orig.get(&col_o).ok_or_else(|| key_err(col_o))?;
            mo = derive_motion(po, &dd.mbs, w, h, pics, Some(col_m)).with_context(|| format!("dec {d}"))?;
            let cn = new_of[col_o];
            if cn == NONE {
                return Err(key_err(col_o));
            }
            let col_n_m = parsed.0.get(&cn).ok_or_else(|| key_err(cn))?;
            mn = derive_motion(pn, &dd.mbs, w, h, new_pics, Some(&col_n_m.motion)).with_context(|| format!("dec {nd}"))?;
        } else {
            mo = derive_motion(po, &dd.mbs, w, h, pics, None).with_context(|| format!("dec {d}"))?;
            mn = derive_motion(pn, &dd.mbs, w, h, new_pics, None).with_context(|| format!("dec {nd}"))?;
        }
        if cand_set.contains(&d) && t != SliceType::I {
            blocks.insert(d, changed_blocks(&mo, &mn, orig_of_new, w, h));
        }
        m_orig.insert(d, mo);
        parsed.0.insert(nd, Arc::new(ParsedPic { dd, motion: Arc::new(mn) }));
    }
    let mut patches: BTreeMap<usize, Vec<(u16, u16)>> = BTreeMap::new();
    let mut sniff = Vec::new();
    let mut frames_for_targets: HashMap<i32, Vec<u8>> = HashMap::new();
    let todo: BTreeMap<usize, &Vec<(usize, usize)>> = blocks.iter().filter(|(_, b)| !b.is_empty()).map(|(d, b)| (*d, b)).collect();
    if !todo.is_empty() {
        // original pixels of the affected pictures and of every picture their
        // changed blocks read, old and new context
        let mut refs: BTreeSet<usize> = todo.keys().copied().collect();
        for (&d, bl) in &todo {
            let mo = &m_orig[&d];
            let mn = &parsed.0[&new_of[d]].motion;
            for &(gx, gy) in bl.iter() {
                let i = mo.idx(gx, gy);
                for l in 0..2 {
                    if mo.refpic[l][i] >= 0 {
                        refs.insert(mo.refpic[l][i] as usize);
                    }
                    if mn.refpic[l][i] >= 0 {
                        refs.insert(orig_of_new[mn.refpic[l][i] as usize]);
                    }
                }
            }
        }
        let first = task.orig.start;
        let last = *refs.iter().next_back().expect("non-empty");
        let want: BTreeSet<i32> = refs.iter().map(|&d| pics[d].poc).collect();
        let frames = decode_pocs(cx.orig_path, pics, first, last + 1, &want, frame_size(sps), FFMPEG_THREADS)?;
        let mut oracle = PixelOracle::new(frames, pics, w, h);
        for (&d, bl) in &todo {
            let (mo, mn) = (&m_orig[&d], &parsed.0[&new_of[d]].motion);
            let mut mbs: BTreeSet<(u16, u16)> = BTreeSet::new();
            for &(gx, gy) in bl.iter() {
                if !oracle.harmless(gx, gy, mo, mn, orig_of_new)? {
                    mbs.insert(((gx / 4) as u16, (gy / 4) as u16));
                }
            }
            sniff.push(json!({"dec": d, "display": pics[d].disp, "type": pics[d].typ().as_str(),
                              "changed_blocks": bl.len(), "mbs": mbs.len()}));
            if !mbs.is_empty() {
                patches.insert(new_of[d], mbs.into_iter().collect());
            }
        }
        frames_for_targets = oracle.frames;
    }
    let (mut repl, mut stats) = (BTreeMap::new(), BTreeMap::new());
    if !patches.is_empty() {
        let ctx = PatchCtx { es: cx.new_es, spss: cx.spss, ppss: cx.ppss, pics: new_pics, range: task.new.clone() };
        let frames = &frames_for_targets;
        let targets = |dec: usize, x: u16, y: u16| -> Result<crate::records::Mb> {
            let poc = new_pics[dec].poc;
            let fr = frames.get(&poc).ok_or_else(|| anyhow!("KeyError: {poc}"))?;
            Ok(pcm_record(fr, w, x as usize, y as usize))
        };
        let r = patch_pictures(&ctx, &patches, &targets, &mut parsed, &BTreeSet::new(), None, &mut |_| {})?;
        repl = r.repl;
        stats = r.stats;
    }
    Ok(TaskResult {
        period: task.period,
        repl,
        stats,
        sniff,
        rounds: None,
        patches: patches.iter().map(|(nd, v)| (*nd, v.len())).collect(),
        seconds: round1(t0.elapsed().as_secs_f64()),
    })
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

/// Returns ({new decode index: replacement NAL}, the report's `repair`
/// section with Python's keys). Explicit mode (see explicit.rs): `sigma` =
/// {dropped reference: substitute or None}, `orig_of_out` = original decode
/// index of every picture of the new stream.
pub fn repair(a: &RepairArgs, log: &Logger) -> Result<(HashMap<usize, Vec<u8>>, Value)> {
    let sps = a.spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let (pics, new_pics) = (a.pics, a.new_pics);
    let explicit = a.sigma.is_some();
    let (tasks, cands, orig_of_new, peak): (Vec<Task>, Vec<usize>, Vec<usize>, f64);
    if let Some(sigma) = a.sigma {
        let oo = a.orig_of_out.ok_or_else(|| anyhow!("explicit repair needs orig_of_out"))?;
        if oo.len() != new_pics.len() {
            bail!("orig_of_out has {} entries for {} pictures", oo.len(), new_pics.len());
        }
        let (t, c) = build_tasks_explicit(pics, new_pics, oo, sigma)?;
        log.log(&format!("repair: {} pictures read a substituted twin, in {} IDR periods", c.len(), t.len()));
        (tasks, cands, orig_of_new, peak) = (t, c, oo.to_vec(), REPAIR_PEAK_EXPLICIT);
    } else {
        let oon: Vec<usize> = pics.iter().map(|p| p.dec).filter(|d| !a.drop.contains(d)).collect();
        if oon.len() != new_pics.len() {
            bail!("{} kept pictures for {} pictures in the new stream", oon.len(), new_pics.len());
        }
        let (c, _) = find_candidates(a.spss, pics, a.drop, None)?;
        let t = build_tasks(pics, new_pics, &c)?;
        log.log(&format!("repair: {} pictures with a changed reference context in {} IDR periods", c.len(), t.len()));
        (tasks, cands, orig_of_new, peak) = (t, c, oon, REPAIR_PEAK);
    }
    let mut new_of = vec![NONE; pics.len()];
    for (nd, &d) in orig_of_new.iter().enumerate() {
        new_of[d] = nd;
    }
    let nproc = if tasks.is_empty() { 0 } else { pool_size(a.workers, peak, Some(tasks.len())) };
    let mut results: Vec<TaskResult> = Vec::new();
    if !tasks.is_empty() {
        log.log(&format!("repair: {nproc} workers"));
        let orig_map = map_file(a.orig_es)?;
        let new_map = map_file(a.new_es)?;
        let cx = Ctx {
            orig_path: a.orig_es,
            orig_es: &orig_map,
            new_es: &new_map,
            spss: a.spss,
            ppss: a.ppss,
            pics,
            new_pics,
            orig_of_new: &orig_of_new,
            new_of,
            sigma: a.sigma,
            work: a.work,
            orig_ranges: period_ranges(pics),
        };
        let n_tasks = tasks.len();
        let work = |t: &Task| if explicit { run_task_explicit(&cx, t) } else { run_task_exact(&cx, t) };
        results = run_pool(nproc, &tasks, &work, |i, r: &TaskResult| {
            let n_pcm: usize = r.patches.values().sum();
            let rounds = r.rounds.map(|n| format!(", {n} rounds")).unwrap_or_default();
            log.log(&format!(
                "  [{}/{}] IDR period {}: {} pictures rewritten, {n_pcm} PCM MBs ({:.1} s{rounds})",
                i + 1,
                n_tasks,
                r.period,
                r.repl.len(),
                r.seconds
            ));
        })?;
    }
    let mut repl: HashMap<usize, Vec<u8>> = HashMap::new();
    let (mut pictures_rewritten, mut pcm_mbs, mut bytes_added) = (0usize, 0usize, 0i64);
    let (mut skip_to_explicit, mut direct_to_explicit, mut qp_carriers) = (0usize, 0usize, 0usize);
    let mut sniff: Vec<Value> = Vec::new();
    let mut pictures: Vec<Value> = Vec::new();
    results.sort_by_key(|r| r.period);
    for r in &results {
        for (nd, nal) in &r.repl {
            repl.insert(*nd, nal.as_ref().clone());
        }
        sniff.extend(r.sniff.iter().cloned());
        for (&nd, s) in &r.stats {
            pictures_rewritten += 1;
            pcm_mbs += s.pcm;
            bytes_added += s.bytes_after as i64 - s.bytes_before as i64;
            skip_to_explicit += s.skip_explicit;
            direct_to_explicit += s.direct_explicit;
            qp_carriers += s.qp_carrier;
            let mut v = json!({"out_dec": nd, "out_display": new_pics[nd].disp, "in_display": pics[orig_of_new[nd]].disp});
            if let (Some(o), Some(sj)) = (v.as_object_mut(), s.to_json().as_object()) {
                for (k, x) in sj {
                    o.insert(k.clone(), x.clone());
                }
            }
            pictures.push(v);
        }
    }
    let peak_mb = (peak_rss() as f64 / (1u64 << 20) as f64).round() as u64;
    let mut summary = json!({
        "candidate_pictures": cands.len(), "periods": tasks.len(), "pictures_rewritten": pictures_rewritten,
        "pcm_mbs": pcm_mbs, "bytes_added": bytes_added, "skip_to_explicit": skip_to_explicit,
        "direct_to_explicit": direct_to_explicit, "qp_carriers": qp_carriers, "workers": nproc,
        "worker_peak_rss_mb": peak_mb,
    });
    let o = summary.as_object_mut().expect("object");
    if explicit {
        o.insert("drift_threshold".into(), json!(DRIFT_THRESHOLD));
        o.insert("drift_max_per_picture".into(), json!((DRIFT_MAX_FRACTION * sps.mb_w as f64 * sps.mb_h as f64) as usize));
        o.insert("seconds".into(), json!(round1(results.iter().map(|r| r.seconds).sum::<f64>())));
    }
    if explicit {
        let n_over = sniff.iter().filter(|x| x["over_threshold_after"].as_u64().unwrap_or(0) > 0).count();
        let n_capped = sniff.iter().filter(|x| x["capped"].as_bool().unwrap_or(false)).count();
        let worst = sniff.iter().map(|x| x["worst_after"].as_u64().unwrap_or(0)).max().unwrap_or(0);
        o.insert("sniff".into(), Value::Array(sniff));
        o.insert("pictures".into(), Value::Array(pictures));
        o.insert("pictures_over_threshold_after".into(), json!(n_over));
        o.insert("pictures_capped".into(), json!(n_capped));
        o.insert("worst_after".into(), json!(worst));
    } else {
        o.insert("sniff".into(), Value::Array(sniff));
        o.insert("pictures".into(), Value::Array(pictures));
    }
    Ok((repl, summary))
}

/// Inputs of [`repair`] (Python: `repair(orig_es, new_es, spss, ppss, pics,
/// new_pics, drop, workers, log, sigma, work, orig_of_out)`).
pub struct RepairArgs<'a> {
    /// The original Annex-B stream.
    pub orig_es: &'a Path,
    /// The deduplicated stream (stage 1) to repair.
    pub new_es: &'a Path,
    /// Parameter sets of the original stream.
    pub spss: &'a Spss,
    pub ppss: &'a Ppss,
    /// The original stream's pictures (after `stream::load`).
    pub pics: &'a [Picture],
    /// The deduplicated stream's pictures (after `stream::load`).
    pub new_pics: &'a [Picture],
    /// Original decode indices that were dropped.
    pub drop: &'a HashSet<usize>,
    pub workers: usize,
    /// Explicit mode: dropped reference picture -> its substitute (or None).
    pub sigma: Option<&'a Sigma>,
    /// Work directory (temporary segment files).
    pub work: &'a Path,
    /// Explicit mode: the original decode index of every new picture.
    pub orig_of_out: Option<&'a [usize]>,
}

// ---------------------------------------------------------------------------
// dev repair
// ---------------------------------------------------------------------------

/// `frame-dedup dev repair <scenario.json> [--out repaired.h264] [--report r.json] [--workers N] [--work DIR]`:
/// run the repair stage on the streams named in the scenario (written by
/// `python -m port.oracle repair-scenario`), write the repaired stream with
/// `write_stream` and the `repair` report section. Prints the timing.
pub fn dev_repair(args: &[String]) -> Result<()> {
    const USAGE: &str = "frame-dedup dev repair <scenario.json> [--out repaired.h264] [--report r.json] [--workers N] [--work DIR]";
    let (mut scen, mut out, mut report, mut workers, mut work) = (None, None, None, 4usize, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = |n: &str| it.next().cloned().ok_or_else(|| anyhow!("{n} needs a value"));
        match a.as_str() {
            "--out" => out = Some(val("--out")?),
            "--report" => report = Some(val("--report")?),
            "--workers" => workers = val("--workers")?.parse()?,
            "--work" => work = Some(val("--work")?),
            _ if scen.is_none() => scen = Some(a.clone()),
            _ => bail!("usage: {USAGE}"),
        }
    }
    let scen = scen.ok_or_else(|| anyhow!("usage: {USAGE}"))?;
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&scen)?)?;
    let s = |k: &str| j[k].as_str().map(PathBuf::from).ok_or_else(|| anyhow!("scenario has no {k}"));
    let (orig_es, new_es) = (s("orig_es")?, s("new_es")?);
    let work = match work {
        Some(w) => PathBuf::from(w),
        None => s("work")?,
    };
    let ids = |v: &Value| -> Vec<usize> { v.as_array().map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect()).unwrap_or_default() };
    let drop: HashSet<usize> = ids(&j["drop"]).into_iter().collect();
    let sigma: Option<Sigma> = j["sigma"].as_object().map(|m| m.iter().map(|(k, v)| (k.parse().unwrap(), v.as_u64().map(|x| x as usize))).collect());
    let orig_of_out: Option<Vec<usize>> = j["orig_of_out"].as_array().map(|_| ids(&j["orig_of_out"]));
    crate::system::lower_priority();
    let log = Logger::new(None)?;
    let (spss, ppss, pics) = crate::stream::load(&orig_es)?;
    let (_nspss, _nppss, npics) = crate::stream::load(&new_es)?;
    let t0 = Instant::now();
    let (repl, rep) = repair(
        &RepairArgs {
            orig_es: &orig_es,
            new_es: &new_es,
            spss: &spss,
            ppss: &ppss,
            pics: &pics,
            new_pics: &npics,
            drop: &drop,
            workers,
            sigma: sigma.as_ref(),
            work: &work,
            orig_of_out: orig_of_out.as_deref(),
        },
        &log,
    )?;
    let secs = t0.elapsed().as_secs_f64();
    if let Some(out) = &out {
        // Python: out_es = dedup.h264 itself when nothing was rewritten
        let opts = crate::stream::WriteOpts { repl: Some(&repl), ..Default::default() };
        crate::stream::write_stream(&new_es, &npics, &_nspss, Path::new(out), &opts)?;
    }
    if let Some(r) = &report {
        std::fs::write(r, crate::records::canonical_text(&json!({"repair": rep, "wall_seconds": secs})))?;
    }
    println!(
        "repair: {} pictures rewritten, {} PCM MBs, {:+} bytes; {:.1} s; process peak {} MB",
        rep["pictures_rewritten"],
        rep["pcm_mbs"],
        rep["bytes_added"].as_i64().unwrap_or(0),
        secs,
        peak_rss() >> 20
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mb_rect_clamps() {
        // block at the top-left corner reading up-left: clamped to MB 0
        assert_eq!(mb_rect(0, 0, [-40, -40], 4, 3), (0, 1, 0, 1));
        // in the middle: x = 8*4 + 2 = 34 -> 30..42 -> MBs 1..3; y the same
        assert_eq!(mb_rect(8, 8, [8, 8], 10, 10), (1, 3, 1, 3));
        // bottom-right: clamped at the last MB
        assert_eq!(mb_rect(15, 11, [100, 100], 4, 3), (2, 3, 3, 4));
    }

    #[test]
    fn frame_key_distinguishes() {
        let a = vec![7u8; 1000];
        let mut b = a.clone();
        b[999] = 8;
        let mut c = a.clone();
        c[0] = 8;
        assert_eq!(frame_key(&a), frame_key(&a.clone()));
        assert_ne!(frame_key(&a), frame_key(&b));
        assert_ne!(frame_key(&a), frame_key(&c));
        assert_ne!(frame_key(&b), frame_key(&c));
        assert_ne!(frame_key(&a[..999]), frame_key(&a));
    }

    #[test]
    fn pool_runs_all_tasks_in_order() {
        let tasks: Vec<usize> = (0..50).collect();
        let mut seen = 0;
        let r = run_pool(4, &tasks, &|t: &usize| Ok(*t * 2), |_, _| seen += 1).unwrap();
        assert_eq!(seen, 50);
        assert_eq!(r, (0..50).map(|x| x * 2).collect::<Vec<_>>());
        let e = run_pool(4, &tasks, &|t: &usize| if *t == 7 { Err(anyhow!("boom {t}")) } else { Ok(*t) }, |_, _| {}).unwrap_err();
        assert_eq!(e.to_string(), "boom 7");
    }
}

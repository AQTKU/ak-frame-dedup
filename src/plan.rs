//! Drop planning and A/V-sync corrections. Port of `frame_dedup/plan.py`
//! (plus the numerical part of `Run.plan`, as [`run_plan`]).
//!
//! Drop rules for a duplicate pair (n-1, n) in display order:
//!   * the later twin if it is a non-reference picture (bit-exact for the rest),
//!   * else the earlier twin if that is a non-reference picture,
//!   * else (both reference pictures) a twin whose removal changes no other
//!     picture's reference lists ("free": nothing reads it, e.g. a P picture
//!     right before an IDR), later twin preferred -- also bit-exact,
//!   * else a "substitute": a cheap (non-reference or free) picture within 3
//!     frames that is a visually lossless repeat of its predecessor (black or
//!     static), keeping the frame count without any repair,
//!   * else the later twin -- a "hard" drop that is repaired afterwards --
//!     unless it is an IDR: then the earlier twin K is dropped by an "anchor
//!     rebuild" (the IDR is demoted and takes K's place, see rebuild.py) when
//!     `rebuildable` allows it; otherwise the cheap picture within 3 frames
//!     with the least motion is dropped instead (a visible substitute: one
//!     real frame is lost, but A/V sync holds); only if there is none is the
//!     pair kept ("unsupported") and the count corrected at the next sync point.
//!     A hard drop that would leave a later picture's reference list shorter
//!     than its slice header expects (`hard_ok`) is not possible either: that
//!     pair is kept too.
//!
//! A/V sync: kept frame n becomes output frame k, shown at k / fps_out where
//! it was shown at n / fps_in. Its offset in output frames is k - n * r.
//! Inside a stretch of constant phase the offsets cycle around a constant
//! mean. After a phase change or a kept pair that mean moves; it is pulled
//! back within +-0.5 frame at the next point where one frame more or less
//! goes unseen: a black run (drop an extra black frame or keep a black
//! duplicate), a hard cut (trim a frame next to the cut, or keep a duplicate
//! just before it) or a static frame (drop a repeat, or keep one).

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::cadence::{CadenceReport, Chunk, Ratio};
use crate::evidence::np;
use crate::h264hdr::Picture;

/// Frames either side of a pair searched for a substitute drop.
pub const SUB_WINDOW: usize = 3;
/// Proxy mean / block-max difference below which a frame is a visually
/// lossless repeat of its predecessor.
pub const INVISIBLE_MAD: f64 = 0.25;
pub const INVISIBLE_BMAX: f64 = 1.0;

// --------------------------------------------------------------------------
// report entries (serialise to exactly the Python dicts)

/// A reference pair with no cheaper option: the later twin is dropped and
/// the damage repaired afterwards.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Hard {
    pub kept_display: usize,
    pub dropped_display: usize,
    pub kept_dec: usize,
    pub dropped_dec: usize,
    pub types: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Unsupported {
    pub display: usize,
    pub reason: String,
    pub types: String,
}

/// An unreferenced reference picture dropped bit-exactly. `dec` / `sigma`
/// exist only in explicit mode (`sigma`: the substitute twin, or null).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Freed {
    pub display: usize,
    pub twin: &'static str,
    pub types: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dec: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sigma: Option<Option<usize>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Substitute {
    pub pair: usize,
    pub display: usize,
    pub visible: bool,
    pub mad: f64,
    pub types: String,
}

/// An anchor rebuild: K (earlier twin) dropped, the IDR D takes its place.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Rebuild {
    pub pair: usize,
    pub dropped_display: usize,
    pub kept_display: usize,
    pub types: String,
    #[serde(rename = "K_dec")]
    pub k_dec: usize,
    #[serde(rename = "D_dec")]
    pub d_dec: usize,
    #[serde(rename = "B_dec")]
    pub b_dec: Vec<usize>,
}

/// A pair whose anchor rebuild was not possible (Python: a rebuilds entry
/// with a 'fallback' key).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RebuildFallback {
    pub pair: usize,
    pub types: String,
    pub fallback: Option<String>,
}

/// `rebuildable(K dec, D dec)` -> (the rebuild's leading pictures `B` (decode
/// ids) or None, the reason it is not possible).
pub type RebuildableFn<'a> = dyn Fn(usize, usize) -> (Option<Vec<usize>>, Option<String>) + 'a;
/// `free(dec)`: dropping a reference picture leaves all other reference lists unchanged.
/// Also the type of `hard_ok(dec)`: a hard drop of reference picture `dec`
/// leaves no reference list shorter than in the original.
pub type FreeFn<'a> = dyn Fn(usize) -> bool + 'a;

/// What `choose_drops` returns (Python: a 6-tuple, with the rebuild entries
/// that fell back split out into `fallbacks`).
#[derive(Clone, Debug, Default)]
pub struct Choice {
    /// Display indices to drop.
    pub drop: BTreeSet<usize>,
    pub hard: Vec<Hard>,
    pub unsupported: Vec<Unsupported>,
    pub freed: Vec<Freed>,
    pub subs: Vec<Substitute>,
    pub rebuilds: Vec<Rebuild>,
    pub fallbacks: Vec<RebuildFallback>,
}

fn types_of(a: &Picture, b: &Picture) -> String {
    let mut s = String::new();
    s += a.typ().as_str();
    if a.idr {
        s += "i";
    }
    s += b.typ().as_str();
    if b.idr {
        s += "i";
    }
    s
}

/// Decide which picture of every duplicate pair to drop.
///
/// `disp`: pictures in display order; `dup`: bool per display index;
/// `free(dec)`; `mad_bmax`: proxy differences of each frame to its
/// predecessor (for substitute drops); `rebuildable(K dec, D dec)` for pairs
/// whose later twin is an IDR.
///
/// `explicit`: the stream's headers are rewritten with explicit reference
/// control (explicit.py). A reference twin can then only be replaced by a
/// twin decoded before it, so the twin later in decode order is the one
/// dropped (its substitute is the other: 'kept_dec'); a free twin earlier in
/// decode order is dropped with no substitute ('sigma' None); and when the
/// twin later in decode order is an IDR, the pair goes to the anchor rebuild.
pub fn choose_drops(
    disp: &[&Picture],
    dup: &[bool],
    free: Option<&FreeFn>,
    mad_bmax: Option<(&[f64], &[f64])>,
    visible_subs: bool,
    rebuildable: Option<&RebuildableFn>,
    explicit: bool,
    hard_ok: Option<&FreeFn>,
) -> Choice {
    let n_pics = disp.len();
    let mut c = Choice::default();
    let dups: Vec<usize> = (0..dup.len()).filter(|&i| dup[i]).collect();
    let mut twins: BTreeSet<usize> = BTreeSet::new();
    for &n in &dups {
        twins.insert(n.wrapping_sub(1));
        twins.insert(n);
    }
    let free_dec = |d: usize| free.map_or(false, |f| f(d));

    let cheap = |i: usize| {
        let p = disp[i];
        !p.idr && (p.ref_idc == 0 || (free.is_some() && free_dec(p.dec)))
    };

    // The nearest cheap picture to pair `n` with the least motion.
    let substitute = |n: usize, invisible: bool, drop: &BTreeSet<usize>| -> Option<usize> {
        let (mad, bmax) = mad_bmax?;
        let mut best: Option<((f64, usize), usize)> = None;
        let lo = (n.saturating_sub(1 + SUB_WINDOW)).max(1);
        let hi = n_pics.min(n + SUB_WINDOW + 1);
        for i in lo..hi {
            if twins.contains(&i) || drop.contains(&i) || !cheap(i) {
                continue;
            }
            if invisible && !(mad[i] < INVISIBLE_MAD && bmax[i] < INVISIBLE_BMAX) {
                continue;
            }
            let key = (np::round(mad[i], 3), i.abs_diff(n));
            // Python: tuple comparison `key < best[0]`
            let better = match &best {
                None => true,
                Some((k, _)) => key.0 < k.0 || (key.0 == k.0 && key.1 < k.1),
            };
            if better {
                best = Some((key, i));
            }
        }
        best.map(|(_, i)| i)
    };

    for &n in &dups {
        if n == 0 {
            continue;
        }
        let (a, b) = (disp[n - 1], disp[n]);
        let types = types_of(a, b);
        if b.ref_idc == 0 {
            c.drop.insert(n);
        } else if a.ref_idc == 0 {
            c.drop.insert(n - 1);
        } else if explicit {
            // by decode order: L is decoded after E
            let ((l, li), (e, ei)) = if b.dec > a.dec { ((b, n), (a, n - 1)) } else { ((a, n - 1), (b, n)) };
            if !l.idr && free_dec(l.dec) {
                c.drop.insert(li);
                c.freed.push(Freed {
                    display: li,
                    twin: if li == n { "later" } else { "earlier" },
                    types,
                    dec: Some(l.dec),
                    sigma: Some(Some(e.dec)),
                });
                continue;
            }
            if !e.idr && free_dec(e.dec) {
                c.drop.insert(ei);
                c.freed.push(Freed {
                    display: ei,
                    twin: if ei == n { "later" } else { "earlier" },
                    types,
                    dec: Some(e.dec),
                    sigma: Some(None),
                });
                continue;
            }
            let s = substitute(n, true, &c.drop);
            if let Some(s) = s {
                c.drop.insert(s);
                c.subs.push(Substitute { pair: n, display: s, visible: false, mad: np::round(mad_bmax.unwrap().0[s], 3), types });
            } else if !l.idr {
                c.drop.insert(li);
                c.hard.push(Hard { kept_display: ei, dropped_display: li, kept_dec: e.dec, dropped_dec: l.dec, types });
            } else {
                let (rb, why) = match rebuildable {
                    Some(f) => f(e.dec, l.dec),
                    None => (None, None),
                };
                if let Some(bs) = rb {
                    c.drop.insert(ei);
                    c.rebuilds.push(Rebuild {
                        pair: n,
                        dropped_display: ei,
                        kept_display: li,
                        types,
                        k_dec: e.dec,
                        d_dec: l.dec,
                        b_dec: bs,
                    });
                    continue;
                }
                if rebuildable.is_some() {
                    c.fallbacks.push(RebuildFallback { pair: n, types: types.clone(), fallback: why });
                }
                let s = if visible_subs { substitute(n, false, &c.drop) } else { None };
                if let Some(s) = s {
                    c.drop.insert(s);
                    c.subs.push(Substitute { pair: n, display: s, visible: true, mad: np::round(mad_bmax.unwrap().0[s], 3), types });
                } else {
                    c.unsupported.push(Unsupported {
                        display: n,
                        reason: "later twin is an IDR and no substitute".to_string(),
                        types,
                    });
                }
            }
        } else if free.is_some() && free_dec(b.dec) {
            c.drop.insert(n);
            c.freed.push(Freed { display: n, twin: "later", types, dec: None, sigma: None });
        } else if free.is_some() && free_dec(a.dec) {
            c.drop.insert(n - 1);
            c.freed.push(Freed { display: n - 1, twin: "earlier", types, dec: None, sigma: None });
        } else {
            let s = substitute(n, true, &c.drop);
            if let Some(s) = s {
                c.drop.insert(s);
                c.subs.push(Substitute { pair: n, display: s, visible: false, mad: np::round(mad_bmax.unwrap().0[s], 3), types });
            } else if !b.idr && !hard_ok.map_or(true, |f| f(b.dec)) {
                c.unsupported.push(Unsupported {
                    display: n,
                    reason: "dropping the later twin would leave a reference list short".to_string(),
                    types,
                });
            } else if !b.idr {
                c.drop.insert(n);
                c.hard.push(Hard { kept_display: n - 1, dropped_display: n, kept_dec: a.dec, dropped_dec: b.dec, types });
            } else {
                let (rb, why) = match rebuildable {
                    Some(f) => f(a.dec, b.dec),
                    None => (None, None),
                };
                if let Some(bs) = rb {
                    c.drop.insert(n - 1);
                    c.rebuilds.push(Rebuild {
                        pair: n,
                        dropped_display: n - 1,
                        kept_display: n,
                        types,
                        k_dec: a.dec,
                        d_dec: b.dec,
                        b_dec: bs,
                    });
                    continue;
                }
                if rebuildable.is_some() {
                    c.fallbacks.push(RebuildFallback { pair: n, types: types.clone(), fallback: why });
                }
                let s = if visible_subs { substitute(n, false, &c.drop) } else { None };
                if let Some(s) = s {
                    c.drop.insert(s);
                    c.subs.push(Substitute { pair: n, display: s, visible: true, mad: np::round(mad_bmax.unwrap().0[s], 3), types });
                } else {
                    c.unsupported.push(Unsupported {
                        display: n,
                        reason: "later twin is an IDR and no substitute".to_string(),
                        types,
                    });
                }
            }
        }
    }
    c
}

/// Offset k - n*r (output frames) of every display index (NaN if dropped).
pub fn offsets(n: usize, drop: &BTreeSet<usize>, r: Ratio) -> Vec<f64> {
    let rf = r.to_f64();
    let mut off = vec![0.0; n];
    let mut k: i64 = -1;
    for i in 0..n {
        if drop.contains(&i) {
            off[i] = f64::NAN;
        } else {
            k += 1;
            off[i] = k as f64 - i as f64 * rf;
        }
    }
    off
}

/// Kept frames either side of a cut that a sync correction may trim, nearest
/// to the cut first: the last frame of the old shot and the first of the new
/// one are pure trims; the second ones skip a frame right next to the cut.
pub const CUT_TRIM: usize = 2;
/// Frames before a cut searched for a dropped duplicate to keep instead.
pub const CUT_KEEP_WINDOW: usize = 6;

/// One adjustment of the frame count for A/V sync.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Correction {
    /// "black" (a black run), "cut" (a hard cut) or "static" (a frame that
    /// repeats its predecessor).
    pub at: &'static str,
    /// The black run [start, end), the cut's first frame [c, c], or the
    /// static frame [i, i].
    pub frames: [usize; 2],
    pub action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<usize>,
    /// 'none possible' only (unrounded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_offset_before: Option<f64>,
}

/// Where the frame count can change invisibly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Point {
    /// A black run [s, e): drop an extra black frame or keep a black duplicate.
    Black(usize, usize),
    /// A hard cut at c (c starts the new shot): trim the old shot's last
    /// frame, or keep a duplicate just before the cut.
    Cut(usize),
    /// Frame i repeats its predecessor: drop it, or keep a dropped duplicate
    /// there when the next frame repeats too.
    Static(usize),
}

impl Point {
    /// First frame of the stretch whose mean offset the point corrects.
    fn from(self) -> usize {
        match self {
            Point::Black(_, e) => e,
            Point::Cut(c) | Point::Static(c) => c,
        }
    }
}

/// What [`sync_correct`] needs besides the drops.
pub struct SyncInputs<'a> {
    pub disp: &'a [&'a Picture],
    pub r: Ratio,
    pub black: &'a [(usize, usize)],
    /// First frames of new shots (`cadence::hard_cuts`).
    pub cuts: &'a [usize],
    /// Proxy differences of each frame to its predecessor.
    pub mad: &'a [f64],
    pub bmax: &'a [f64],
    /// Display indices where the offset can jump outside a correction point
    /// (phase changes at hard cuts, kept pairs).
    pub events: &'a [usize],
    /// Both twins of every duplicate pair: never dropped at a static point,
    /// since the other twin may already be gone.
    pub twins: &'a BTreeSet<usize>,
    /// Pictures other drops depend on (substitute twins, rebuild pictures).
    pub protect: &'a BTreeSet<usize>,
    /// A picture that can be dropped without changing any other picture
    /// (non-reference, or a reference nothing reads), not an IDR.
    pub cheap: &'a dyn Fn(usize) -> bool,
    /// Largest acceptable |mean offset| of a stretch, in output frames.
    pub max_off: f64,
}

/// Prefix sums of the kept frames' offsets, for O(1) stretch means.
struct OffsetSums {
    sum: Vec<f64>,
    cnt: Vec<usize>,
}

impl OffsetSums {
    fn new(n: usize, drop: &BTreeSet<usize>, r: Ratio) -> Self {
        let off = offsets(n, drop, r);
        let mut sum = vec![0.0; n + 1];
        let mut cnt = vec![0; n + 1];
        for (i, v) in off.iter().enumerate() {
            let kept = v.is_finite();
            sum[i + 1] = sum[i] + if kept { *v } else { 0.0 };
            cnt[i + 1] = cnt[i] + usize::from(kept);
        }
        OffsetSums { sum, cnt }
    }

    /// Mean offset of the kept frames in [a, b), NaN if there are none.
    fn mean(&self, a: usize, b: usize) -> f64 {
        let c = self.cnt[b] - self.cnt[a];
        if c == 0 { f64::NAN } else { (self.sum[b] - self.sum[a]) / c as f64 }
    }
}

/// Frame i is a visually lossless repeat of its predecessor.
fn repeats(mad: &[f64], bmax: &[f64], i: usize) -> bool {
    i < mad.len() && mad[i] < INVISIBLE_MAD && bmax[i] < INVISIBLE_BMAX
}

/// Adjust the frame count at black runs, hard cuts and static frames so that
/// the mean offset of every stretch stays within +-`max_off`. Points are
/// taken in display order, so a drift is corrected at the first point after
/// it; each corrects the stretch from itself to the next black run or event.
/// Returns (drop, corrections).
pub fn sync_correct(drop: &BTreeSet<usize>, s: &SyncInputs) -> (BTreeSet<usize>, Vec<Correction>) {
    let disp = s.disp;
    let n = disp.len();
    let mut drop = drop.clone();
    let mut corrections = Vec::new();
    let mut stops: BTreeSet<usize> = s.black.iter().map(|&(a, _)| a).collect();
    stops.extend(s.events.iter().copied());
    stops.insert(n);
    let in_black = |i: usize| s.black.iter().any(|&(a, e)| a <= i && i <= e);

    let mut points: Vec<Point> = s.black.iter().map(|&(a, e)| Point::Black(a, e)).collect();
    points.extend(s.cuts.iter().filter(|&&c| c > 0 && c < n && !in_black(c)).map(|&c| Point::Cut(c)));
    points.extend((1..n).filter(|&i| repeats(s.mad, s.bmax, i) && !in_black(i)).map(Point::Static));
    points.sort_by_key(|p| (p.from(), *p));

    let mut sums = OffsetSums::new(n, &drop, s.r);
    for p in points {
        let from = p.from();
        let Some(&nxt) = stops.range(from + 1..).next() else { continue };
        let tries = if matches!(p, Point::Black(..)) { 4 } else { 1 };
        for _ in 0..tries {
            let m = sums.mean(from, nxt);
            if m.is_nan() || m.abs() <= s.max_off {
                break;
            }
            let too_many = m > s.max_off;
            let pick: Option<(usize, &'static str)> = match p {
                Point::Black(a, e) => {
                    if too_many {
                        (a..e)
                            .rev()
                            .find(|&i| !drop.contains(&i) && disp[i].ref_idc == 0 && !disp[i].idr && !s.protect.contains(&i))
                            .map(|i| (i, "drop extra black frame"))
                    } else {
                        (a..e).rev().find(|i| drop.contains(i)).map(|i| (i, "keep black duplicate"))
                    }
                }
                Point::Cut(c) => {
                    if too_many {
                        const ACTIONS: [[&str; 2]; 2] = [
                            ["drop last frame of shot", "drop first frame of shot"],
                            ["drop second-last frame of shot", "drop second frame of shot"],
                        ];
                        let before: Vec<usize> = (0..c).rev().filter(|i| !drop.contains(i)).take(CUT_TRIM).collect();
                        let after: Vec<usize> = (c..nxt).filter(|i| !drop.contains(i)).take(CUT_TRIM).collect();
                        (0..CUT_TRIM)
                            .flat_map(|k| [(k, 0, before.get(k)), (k, 1, after.get(k))])
                            .find_map(|(k, side, i)| {
                                let i = *i?;
                                ((s.cheap)(i) && !s.protect.contains(&i)).then_some((i, ACTIONS[k][side]))
                            })
                    } else {
                        (c.saturating_sub(CUT_KEEP_WINDOW)..c)
                            .rev()
                            .find(|i| drop.contains(i))
                            .map(|i| (i, "keep duplicate before cut"))
                    }
                }
                Point::Static(i) => {
                    if too_many {
                        (!drop.contains(&i) && !s.twins.contains(&i) && (s.cheap)(i) && !s.protect.contains(&i))
                            .then_some((i, "drop static frame"))
                    } else {
                        (drop.contains(&i) && repeats(s.mad, s.bmax, i + 1)).then_some((i, "keep static duplicate"))
                    }
                }
            };
            let Some((i, action)) = pick else {
                if let Point::Black(a, e) = p {
                    corrections.push(Correction {
                        at: "black",
                        frames: [a, e],
                        action: "none possible",
                        display: None,
                        mean_offset: Some(m),
                        mean_offset_before: None,
                    });
                }
                break;
            };
            if too_many {
                drop.insert(i);
            } else {
                drop.remove(&i);
            }
            let (at, frames) = match p {
                Point::Black(a, e) => ("black", [a, e]),
                Point::Cut(c) => ("cut", [c, c]),
                Point::Static(i) => ("static", [i, i]),
            };
            corrections.push(Correction {
                at,
                frames,
                action,
                display: Some(i),
                mean_offset: None,
                mean_offset_before: Some(np::round(m, 3)),
            });
            sums = OffsetSums::new(n, &drop, s.r);
        }
    }
    (drop, corrections)
}

/// Per-chunk offset statistics (`sync.chunks`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChunkOffset {
    pub start: usize,
    pub mean_offset_frames: f64,
    pub max_abs_offset_frames: f64,
}

/// The report's `sync` statistics (without `corrections`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SyncStats {
    pub kept_frames: usize,
    pub ideal_frames: f64,
    pub final_offset_frames: f64,
    pub max_abs_offset_frames: f64,
    pub mean_offset_frames: f64,
    pub max_abs_offset_ms: f64,
    /// Filled by [`chunk_offsets`] (Run.plan adds it after `sync_stats`).
    pub chunks: Vec<ChunkOffset>,
}

pub fn sync_stats(n: usize, drop: &BTreeSet<usize>, r: Ratio, fps_out: f64) -> SyncStats {
    let off = offsets(n, drop, r);
    let kept = n - drop.len();
    let abs: Vec<f64> = off.iter().map(|v| v.abs()).collect();
    let max_abs = np::nanmax_f64(&abs);
    let last = off.iter().rev().find(|v| !v.is_nan()).copied().unwrap_or(f64::NAN);
    SyncStats {
        kept_frames: kept,
        ideal_frames: np::round(n as f64 * r.to_f64(), 3),
        final_offset_frames: np::round(last, 3),
        max_abs_offset_frames: np::round(max_abs, 3),
        mean_offset_frames: np::round(np::nanmean_f64(&off), 3),
        max_abs_offset_ms: np::round(max_abs * 1000.0 / fps_out, 1),
        chunks: Vec::new(),
    }
}

/// Mean and max |offset| over each chunk (chunks without a kept frame are skipped).
pub fn chunk_offsets(off: &[f64], chunks: &[Chunk]) -> Vec<ChunkOffset> {
    let mut out = Vec::new();
    for ch in chunks {
        let seg = &off[ch.start..ch.end];
        if seg.iter().any(|v| v.is_finite()) {
            let abs: Vec<f64> = seg.iter().map(|v| v.abs()).collect();
            out.push(ChunkOffset {
                start: ch.start,
                mean_offset_frames: np::round(np::nanmean_f64(seg), 3),
                max_abs_offset_frames: np::round(np::nanmax_f64(&abs), 3),
            });
        }
    }
    out
}

// --------------------------------------------------------------------------
// Run.plan, numerical part

/// The report's `drops` section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DropsReport {
    pub total: usize,
    pub duplicates: usize,
    pub non_reference: i64,
    pub unreferenced_reference: usize,
    pub substitute_invisible: usize,
    pub substitute_visible: usize,
    pub hard: usize,
    pub anchor_rebuild: usize,
    pub anchor_rebuild_fallbacks: usize,
    pub sync_net_change: i64,
}

/// The report's `sync` section.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SyncReport {
    pub corrections: Vec<Correction>,
    #[serde(flatten)]
    pub stats: SyncStats,
}

/// The plan-related report sections, serialising to the Python report's keys.
#[derive(Clone, Debug, Serialize)]
pub struct PlanReport {
    pub drops: DropsReport,
    pub unreferenced_reference_drops: Vec<Freed>,
    pub substitute_drops: Vec<Substitute>,
    pub hard_pairs: Vec<Hard>,
    pub anchor_rebuild_fallbacks: Vec<RebuildFallback>,
    pub unsupported_pairs: Vec<Unsupported>,
    pub sync: SyncReport,
}

/// Inputs of `Run.plan` that are not pictures.
pub struct PlanInputs<'a> {
    pub disp: &'a [&'a Picture],
    pub dup: &'a [bool],
    pub free: Option<&'a FreeFn<'a>>,
    pub mad: &'a [f64],
    pub bmax: &'a [f64],
    /// `not --keep-idr-pairs`
    pub visible_subs: bool,
    pub rebuildable: Option<&'a RebuildableFn<'a>>,
    pub explicit: bool,
    /// Old path: `hard_ok(dec)` (see [`FreeFn`]); a pair that fails it is kept.
    pub hard_ok: Option<&'a FreeFn<'a>>,
    pub black: &'a [(usize, usize)],
    /// First frames of new shots.
    pub cuts: &'a [usize],
    /// Cuts where the cadence's duplicate was left out
    /// ([`crate::cadence::cut_repeats`]): the frame count steps there.
    pub cut_repeats: &'a [usize],
    pub r: Ratio,
    pub fps_out: Ratio,
    pub cadence: &'a CadenceReport,
}

/// Result of [`run_plan`]: the final drop set and the filtered entry lists
/// that `Run.plan` stores (`self.drop`, `self.hard`, `self.rebuilds`,
/// `self.freed`), plus the report sections.
#[derive(Clone, Debug)]
pub struct PlanOutput {
    pub drop: BTreeSet<usize>,
    /// Hard pairs whose dropped picture is still dropped after the sync correction.
    pub hard: Vec<Hard>,
    pub rebuilds: Vec<Rebuild>,
    /// Freed entries still dropped after the sync correction.
    pub freed: Vec<Freed>,
    pub report: PlanReport,
}

/// `Run.plan` without the header-dependent callbacks and logging: choose the
/// drops, correct the A/V sync, and build the report sections.
pub fn run_plan(inp: &PlanInputs) -> PlanOutput {
    let n = inp.disp.len();
    let ch = choose_drops(inp.disp, inp.dup, inp.free, Some((inp.mad, inp.bmax)), inp.visible_subs, inp.rebuildable, inp.explicit, inp.hard_ok);
    let n_planned = ch.drop.len();
    let n_nonref = n_planned as i64 - ch.hard.len() as i64 - ch.freed.len() as i64 - ch.subs.len() as i64 - ch.rebuilds.len() as i64;
    let mut events: Vec<usize> = inp.cadence.phase_changes.iter().filter(|c| c.at == "cut").map(|c| c.display).collect();
    events.extend(ch.unsupported.iter().map(|u| u.display));
    events.extend(inp.cut_repeats.iter().copied());
    let mut twins = BTreeSet::new();
    for n in (1..inp.dup.len()).filter(|&i| inp.dup[i]) {
        twins.insert(n - 1);
        twins.insert(n);
    }
    // the kept twin of a hard or freed pair may be the dropped one's substitute
    let mut protect: BTreeSet<usize> = BTreeSet::new();
    for h in &ch.hard {
        protect.extend([h.kept_display, h.dropped_display]);
    }
    for f in &ch.freed {
        protect.extend([f.display, if f.twin == "later" { f.display - 1 } else { f.display + 1 }]);
    }
    let disp_of: BTreeMap<usize, usize> = inp.disp.iter().enumerate().map(|(i, p)| (p.dec, i)).collect();
    for rb in &ch.rebuilds {
        protect.insert(rb.kept_display);
        protect.insert(rb.dropped_display);
        protect.extend(rb.b_dec.iter().filter_map(|b| disp_of.get(b).copied()));
    }
    let free = inp.free;
    let cheap = |i: usize| {
        let p = inp.disp[i];
        !p.idr && (p.ref_idc == 0 || free.is_some_and(|f| f(p.dec)))
    };
    let sync = SyncInputs {
        disp: inp.disp,
        r: inp.r,
        black: inp.black,
        cuts: inp.cuts,
        mad: inp.mad,
        bmax: inp.bmax,
        events: &events,
        twins: &twins,
        protect: &protect,
        cheap: &cheap,
        max_off: 0.5,
    };
    let (drop, corrections) = sync_correct(&ch.drop, &sync);
    let hard: Vec<Hard> = ch.hard.iter().filter(|h| drop.contains(&h.dropped_display)).cloned().collect();
    let rebuilds: Vec<Rebuild> = ch
        .rebuilds
        .iter()
        .filter(|r| drop.contains(&r.dropped_display) && !drop.contains(&r.kept_display))
        .cloned()
        .collect();
    let freed: Vec<Freed> = ch.freed.iter().filter(|f| drop.contains(&f.display)).cloned().collect();
    let mut st = sync_stats(n, &drop, inp.r, inp.fps_out.to_f64());
    st.chunks = chunk_offsets(&offsets(n, &drop, inp.r), &inp.cadence.chunks);
    let drops = DropsReport {
        total: drop.len(),
        duplicates: inp.dup.iter().filter(|&&d| d).count(),
        non_reference: n_nonref,
        unreferenced_reference: ch.freed.len(),
        substitute_invisible: ch.subs.iter().filter(|x| !x.visible).count(),
        substitute_visible: ch.subs.iter().filter(|x| x.visible).count(),
        hard: hard.len(),
        anchor_rebuild: rebuilds.len(),
        anchor_rebuild_fallbacks: ch.fallbacks.len(),
        sync_net_change: drop.len() as i64 - n_planned as i64,
    };
    let report = PlanReport {
        drops,
        unreferenced_reference_drops: ch.freed.clone(),
        substitute_drops: ch.subs.clone(),
        hard_pairs: hard.clone(),
        anchor_rebuild_fallbacks: ch.fallbacks.clone(),
        unsupported_pairs: ch.unsupported.clone(),
        sync: SyncReport { corrections, stats: st },
    };
    PlanOutput { drop, hard, rebuilds, freed, report }
}

// --------------------------------------------------------------------------
// dev command

#[derive(serde::Deserialize)]
struct PicIn {
    dec: usize,
    idr: bool,
    ref_idc: u8,
    #[serde(rename = "type")]
    typ: String,
}

#[derive(serde::Deserialize)]
struct PlanFile {
    disp: Vec<PicIn>,
    /// display indices of the predicted duplicates
    dup: Vec<usize>,
    /// recorded answers of free(dec), by decode id
    free: BTreeMap<String, bool>,
    /// whether `free` was passed at all
    #[serde(default = "yes")]
    has_free: bool,
    /// recorded answers of rebuildable(K, D), keyed "K,D": [B list or null, reason or null]
    rebuildable: Option<BTreeMap<String, (Option<Vec<usize>>, Option<String>)>>,
    mad: Vec<f64>,
    bmax: Vec<f64>,
    visible_subs: bool,
    explicit: bool,
    black: Vec<(usize, usize)>,
    #[serde(default)]
    cuts: Vec<usize>,
    r: (i64, i64),
    fps_out: (i64, i64),
    cadence: CadenceIn,
}

fn yes() -> bool {
    true
}

#[derive(serde::Deserialize)]
struct CadenceIn {
    chunks: Vec<(usize, usize)>,
    /// (display, at) of every phase change
    phase_changes: Vec<(usize, String)>,
}

fn dummy_picture(p: &PicIn) -> Picture {
    use crate::h264hdr::{SliceHeader, SliceType};
    let slice_type = match p.typ.as_str() {
        "P" => SliceType::P,
        "B" => SliceType::B,
        "I" => SliceType::I,
        "SP" => SliceType::SP,
        _ => SliceType::SI,
    };
    let sl = SliceHeader {
        first_mb: 0,
        slice_type,
        pps_id: 0,
        frame_num_bit: 0,
        frame_num: 0,
        field_pic: false,
        bottom_field: None,
        idr_pic_id_bit: None,
        idr_pic_id: None,
        poc_lsb_bit: 0,
        poc_lsb: 0,
        delta_poc_bottom: None,
        direct_spatial: None,
        num_ref_l0: 0,
        num_ref_l1: 0,
        rplm: [Vec::new(), Vec::new()],
        rplm_bit: 0,
        rplm_end_bit: 0,
        weighted: false,
        pwt: None,
        mmco: Vec::new(),
        adaptive: false,
        no_output_prior: None,
        long_term_ref: None,
        drpm_bit: 0,
        drpm_end_bit: 0,
        cabac_init_idc: 0,
        qp: 0,
        deblock_idc: 0,
        alpha_off: 0,
        beta_off: 0,
        hdr_bits: 0,
        data_bit: 0,
        nal_type: 0,
        ref_idc: p.ref_idc,
        offset: 0,
        size: 0,
    };
    Picture { slices: vec![sl], idr: p.idr, ref_idc: p.ref_idc, dec: p.dec, ..Default::default() }
}

/// `dev plan <inputs.json>`: `choose_drops` + `sync_correct` + `sync_stats` +
/// `offsets` on recorded inputs (see `python -m port.oracle plan-inputs`),
/// printing the drop set and the report's plan sections.
pub fn dev_plan(args: &[String]) -> Result<()> {
    let Some(path) = args.first() else { bail!("usage: dev plan <inputs.json>") };
    let inp: PlanFile = serde_json::from_str(&std::fs::read_to_string(path)?).context("reading plan inputs")?;
    let pics: Vec<Picture> = inp.disp.iter().map(dummy_picture).collect();
    let disp: Vec<&Picture> = pics.iter().collect();
    let n = disp.len();
    let mut dup = vec![false; n];
    for &i in &inp.dup {
        dup[i] = true;
    }
    let free_tab: BTreeMap<usize, bool> = inp.free.iter().map(|(k, v)| (k.parse().unwrap(), *v)).collect();
    let free = |d: usize| -> bool {
        *free_tab.get(&d).unwrap_or_else(|| panic!("free({d}) was not recorded by the oracle"))
    };
    let rb_tab = inp.rebuildable.clone();
    let rebuildable = |k: usize, d: usize| -> (Option<Vec<usize>>, Option<String>) {
        let t = rb_tab.as_ref().unwrap();
        t.get(&format!("{k},{d}")).unwrap_or_else(|| panic!("rebuildable({k}, {d}) was not recorded")).clone()
    };
    let r = Ratio::new(inp.r.0, inp.r.1);
    // chunk starts / ends only matter to the plan; phase changes give the events
    let cadence = CadenceReport {
        source_rate: String::new(),
        output_fps: String::new(),
        ratio: String::new(),
        phase_bins: 0,
        chunks: inp
            .cadence
            .chunks
            .iter()
            .map(|&(s, e)| Chunk { start: s, end: e, start_s: 0.0, phases: Vec::new(), frames: e - s, phi: Vec::new() })
            .collect(),
        phase_changes: inp
            .cadence
            .phase_changes
            .iter()
            .map(|(d, at)| crate::cadence::PhaseChange {
                display: *d,
                time_s: 0.0,
                at: if at == "cut" { "cut" } else { "black" },
                from_phase: 0,
                to_phase: 0,
                q: 0,
            })
            .collect(),
    };
    let plan = PlanInputs {
        disp: &disp,
        dup: &dup,
        free: if inp.has_free { Some(&free) } else { None },
        mad: &inp.mad,
        bmax: &inp.bmax,
        visible_subs: inp.visible_subs,
        rebuildable: if rb_tab.is_some() { Some(&rebuildable) } else { None },
        explicit: inp.explicit,
        hard_ok: None,
        black: &inp.black,
        cuts: &inp.cuts,
        cut_repeats: &[],
        r,
        fps_out: Ratio::new(inp.fps_out.0, inp.fps_out.1),
        cadence: &cadence,
    };
    let out = run_plan(&plan);
    let v = serde_json::json!({
        "drop": out.drop,
        "report": out.report,
        "hard": out.hard,
        "rebuilds": out.rebuilds,
        "freed": out.freed,
    });
    println!("{}", serde_json::to_string(&v)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_and_round() {
        let drop: BTreeSet<usize> = [2usize].into_iter().collect();
        let off = offsets(5, &drop, Ratio::new(4, 5));
        assert_eq!(off[0], 0.0);
        assert!(off[2].is_nan());
        assert_eq!(off[3], 2.0 - 3.0 * 0.8);
        let st = sync_stats(5, &drop, Ratio::new(4, 5), 24.0);
        assert_eq!(st.kept_frames, 4);
        assert_eq!(st.ideal_frames, 4.0);
    }

    /// 200 non-reference frames with a 4/5 cadence (duplicates at i % 5 == 4,
    /// all dropped), no motion anywhere except where `still` says otherwise.
    struct Scene {
        pics: Vec<Picture>,
        drop: BTreeSet<usize>,
        twins: BTreeSet<usize>,
        mad: Vec<f64>,
        cuts: Vec<usize>,
        events: Vec<usize>,
        protect: BTreeSet<usize>,
    }

    impl Scene {
        fn new() -> Self {
            let n = 200;
            let pics = (0..n).map(|i| Picture { dec: i, ..Default::default() }).collect();
            let drop: BTreeSet<usize> = (0..n).filter(|i| i % 5 == 4).collect();
            let twins = drop.iter().flat_map(|&i| [i - 1, i]).collect();
            Scene { pics, drop, twins, mad: vec![5.0; n], cuts: vec![], events: vec![], protect: BTreeSet::new() }
        }

        fn run(&self) -> (BTreeSet<usize>, Vec<Correction>) {
            let disp: Vec<&Picture> = self.pics.iter().collect();
            let cheap = |i: usize| self.pics[i].ref_idc == 0;
            let s = SyncInputs {
                disp: &disp,
                r: Ratio::new(4, 5),
                black: &[],
                cuts: &self.cuts,
                mad: &self.mad,
                bmax: &self.mad,
                events: &self.events,
                twins: &self.twins,
                protect: &self.protect,
                cheap: &cheap,
                max_off: 0.5,
            };
            sync_correct(&self.drop, &s)
        }
    }

    #[test]
    fn kept_pair_is_trimmed_at_the_next_cut() {
        let mut sc = Scene::new();
        sc.drop.remove(&49);
        sc.events.push(49);
        sc.cuts = vec![100, 150];
        let (drop, corr) = sc.run();
        assert_eq!(corr.len(), 1);
        assert_eq!((corr[0].at, corr[0].action, corr[0].display), ("cut", "drop last frame of shot", Some(98)));
        assert!(drop.contains(&98));
        // without the kept pair nothing changes
        let (drop, corr) = Scene { cuts: vec![100], ..Scene::new() }.run();
        assert!(corr.is_empty());
        assert_eq!(drop, Scene::new().drop);
    }

    #[test]
    fn cut_trim_order() {
        let mut sc = Scene::new();
        sc.drop.remove(&49);
        sc.events.push(49);
        sc.cuts = vec![100];
        sc.pics[98].ref_idc = 1; // last kept before the cut (99 is a dropped duplicate)
        let (_, corr) = sc.run();
        assert_eq!((corr[0].action, corr[0].display), ("drop first frame of shot", Some(100)));
        sc.protect.insert(100);
        let (_, corr) = sc.run();
        assert_eq!((corr[0].action, corr[0].display), ("drop second-last frame of shot", Some(97)));
        sc.pics[97].ref_idc = 1;
        let (_, corr) = sc.run();
        assert_eq!((corr[0].action, corr[0].display), ("drop second frame of shot", Some(101)));
    }

    #[test]
    fn too_few_frames_keeps_a_duplicate_before_the_cut() {
        let mut sc = Scene::new();
        sc.drop.insert(30);
        sc.events.push(30);
        sc.cuts = vec![101];
        let (drop, corr) = sc.run();
        assert_eq!((corr[0].action, corr[0].display), ("keep duplicate before cut", Some(99)));
        assert!(!drop.contains(&99) && drop.contains(&30));
    }

    #[test]
    fn static_frames_but_never_a_twin() {
        let mut sc = Scene::new();
        sc.drop.remove(&49);
        sc.events.push(49);
        sc.mad[118] = 0.0; // twin of the duplicate at 119
        sc.mad[121] = 0.1;
        let (drop, corr) = sc.run();
        assert_eq!(corr.len(), 1);
        assert_eq!((corr[0].at, corr[0].action, corr[0].display), ("static", "drop static frame", Some(121)));
        assert!(drop.contains(&121) && !drop.contains(&118));
    }
}

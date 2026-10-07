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
//!     pair kept ("unsupported") and the count corrected in the next black run.
//!
//! A/V sync: kept frame n becomes output frame k, shown at k / fps_out where
//! it was shown at n / fps_in. Its offset in output frames is k - n * r.
//! Inside a stretch of constant phase the offsets cycle around a constant
//! mean. After a phase change or a kept pair that mean moves; at each black
//! run the number of black frames kept is adjusted (one extra dropped, or one
//! duplicate kept) so that the mean offset of the stretch that follows lies
//! within +-0.5 frame.

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

/// One adjustment of the drops at a black run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Correction {
    pub black_run: [usize; 2],
    pub action: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<usize>,
    /// 'none possible' only (unrounded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_offset: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_offset_before: Option<f64>,
}

/// Adjust drops inside black runs so that the mean offset of the stretch
/// after each run is within +-`max_off` (Python default 0.5). `events`:
/// display indices where the offset can jump outside a black run (phase
/// changes at hard cuts, kept pairs). Returns (drop, corrections).
pub fn sync_correct(
    disp: &[&Picture],
    drop: &BTreeSet<usize>,
    r: Ratio,
    black: &[(usize, usize)],
    events: &[usize],
    max_off: f64,
) -> (BTreeSet<usize>, Vec<Correction>) {
    let n = disp.len();
    let mut drop = drop.clone();
    let mut corrections = Vec::new();
    let mut stops: BTreeSet<usize> = black.iter().map(|&(s, _)| s).collect();
    stops.extend(events.iter().copied());
    stops.insert(n);
    for &(s, e) in black {
        let mut nxt = stops.range(e..).next().copied().unwrap_or(n);
        if nxt <= e {
            nxt = stops.range(e + 1..).next().copied().unwrap_or(n).min(n);
        }
        if nxt <= e {
            continue;
        }
        for _ in 0..4 {
            let off = offsets(n, &drop, r);
            let m = np::nanmean_f64(&off[e..nxt]);
            if m.abs() <= max_off {
                break;
            }
            if m > max_off {
                // drop one more black frame: a non-reference picture in the run
                let cand = (s..e).rev().find(|i| !drop.contains(i) && disp[*i].ref_idc == 0 && !disp[*i].idr);
                let Some(c) = cand else {
                    corrections.push(Correction { black_run: [s, e], action: "none possible", display: None, mean_offset: Some(m), mean_offset_before: None });
                    break;
                };
                drop.insert(c);
                corrections.push(Correction {
                    black_run: [s, e],
                    action: "drop extra black frame",
                    display: Some(c),
                    mean_offset: None,
                    mean_offset_before: Some(np::round(m, 3)),
                });
            } else {
                let cand = (s..e).rev().find(|i| drop.contains(i));
                let Some(c) = cand else {
                    corrections.push(Correction { black_run: [s, e], action: "none possible", display: None, mean_offset: Some(m), mean_offset_before: None });
                    break;
                };
                drop.remove(&c);
                corrections.push(Correction {
                    black_run: [s, e],
                    action: "keep black duplicate",
                    display: Some(c),
                    mean_offset: None,
                    mean_offset_before: Some(np::round(m, 3)),
                });
            }
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
    pub black: &'a [(usize, usize)],
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
/// drops, correct the A/V sync at black runs, and build the report sections.
pub fn run_plan(inp: &PlanInputs) -> PlanOutput {
    let n = inp.disp.len();
    let ch = choose_drops(inp.disp, inp.dup, inp.free, Some((inp.mad, inp.bmax)), inp.visible_subs, inp.rebuildable, inp.explicit);
    let n_planned = ch.drop.len();
    let n_nonref = n_planned as i64 - ch.hard.len() as i64 - ch.freed.len() as i64 - ch.subs.len() as i64 - ch.rebuilds.len() as i64;
    let mut events: Vec<usize> = inp.cadence.phase_changes.iter().filter(|c| c.at == "cut").map(|c| c.display).collect();
    events.extend(ch.unsupported.iter().map(|u| u.display));
    let (drop, corrections) = sync_correct(inp.disp, &ch.drop, inp.r, inp.black, &events, 0.5);
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
        black: &inp.black,
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
}

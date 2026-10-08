//! I_PCM macroblock patcher that keeps the rest of a picture exact.
//! Port of `frame_dedup/patch.py`.
//!
//! Replaces macroblocks of a stream with I_PCM copies of target pixels and
//! holds everything else in the patched pictures fixed:
//!   * P_Skip MBs whose inferred motion changes because a neighbour became
//!     intra are rewritten as explicit P_L0_16x16 with the original motion,
//!   * explicit partitions get their mvd re-solved against the new neighbours,
//!   * the QP chain is preserved (PCM passes QP_pred through, so a following
//!     MB that does not code mb_qp_delta is given an empty coded chroma DC so
//!     it can carry one),
//!   * B pictures whose co-located picture got PCM MBs have the affected
//!     direct blocks rewritten as explicit bi-predicted blocks with the
//!     original direct motion.
//!
//! # `patch_pictures` API
//!
//! Python's `patch_pictures(mm, spss, ppss, pics, patches, yuv_of, log,
//! parsed, stats, force, cache)` becomes [`patch_pictures`] with
//!   * [`PatchCtx`]: the stream bytes, parameter sets, **all** pictures of
//!     the stream, and the decode-index `range` the pictures to consider
//!     live in (Python passes a dict holding one IDR period; the scan for B
//!     pictures whose co-located picture changed stays inside `range`),
//!   * `patches`: `{decode idx: [(mbx, mby)]}` of macroblocks to replace,
//!   * [`Targets`]: where the I_PCM records come from (Python's `yuv_of`;
//!     either a full frame via [`pcm_record`] or per-MB records),
//!   * [`ParsedStore`]: the parsed pictures (records plus the motion to
//!     keep), with [`MapStore`] (a plain map, pre-filled by the exact repair)
//!     and [`LruStore`] (the bounded `_ParsedLRU` of the measured repair,
//!     with "reload" keys recomputed by a callback),
//!   * `force`, and the cross-call [`PatchCache`] (Python's `cache` dict);
//!   * the result [`PatchResult`] holds the replacement NALs and the
//!     statistics (Python's `stats` dict, one fresh map per call), and
//!     `log` receives the log lines.
//!
//! Records of parsed pictures are never modified: the patcher works on
//! copies (`encode_picture` mutates the records it encodes).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::h264hdr::{Picture, Ppss, SliceType, Spss};
use crate::mv::{Motion, derive_motion, derive_motion_target};
use crate::picture::{Parsed, decode_picture, try_encode_picture};
use crate::records::{Coefs, Kind, Mb, Mvds, Part};
use crate::reexpress::{clear_residual, has_direct, reexpress, reexpress_compact};

/// A replacement slice NAL (header byte included, no start code), shared
/// between the cache and the results.
pub type Nal = Arc<Vec<u8>>;

/// An I_PCM record for macroblock (mbx, mby) from an uncropped (coded-size)
/// yuv420p frame; `mb_w` is the width in macroblocks.
pub fn pcm_record(yuv: &[u8], mb_w: usize, mbx: usize, mby: usize) -> Mb {
    let w = mb_w * 16;
    let h = yuv.len() * 2 / 3 / w;
    let (y_plane, rest) = yuv.split_at(w * h);
    let (u_plane, v_plane) = rest.split_at(w * h / 4);
    let mut s = Box::new([0u8; 384]);
    let mut k = 0;
    for y in 0..16 {
        let o = (mby * 16 + y) * w + mbx * 16;
        s[k..k + 16].copy_from_slice(&y_plane[o..o + 16]);
        k += 16;
    }
    for plane in [u_plane, v_plane] {
        for y in 0..8 {
            let o = (mby * 8 + y) * (w / 2) + mbx * 8;
            s[k..k + 8].copy_from_slice(&plane[o..o + 8]);
            k += 8;
        }
    }
    pcm_mb(mbx as u16, mby as u16, s)
}

/// An I_PCM record with the given samples (256 Y, 64 U, 64 V).
pub fn pcm_mb(x: u16, y: u16, samples: Box<[u8; 384]>) -> Mb {
    let mut mb = Mb::new(x, y);
    mb.kind = Kind::Pcm;
    mb.samples = Some(samples);
    mb.intra = true;
    mb.pcm = true;
    mb.cbp = 0x2f;
    mb
}

/// A copy of an MB record that is safe to modify (Python `copy_mb`, which
/// shares coefficients and samples; Rust owns them).
pub fn copy_mb(mb: &Mb) -> Mb {
    mb.clone()
}

/// Whether the motion of macroblock (mbx, mby) is the same in `a` and `b`:
/// equal reference indices, and equal mvs where a list is used.
pub fn same_motion(a: &Motion, b: &Motion, mbx: usize, mby: usize) -> bool {
    for y in mby * 4..mby * 4 + 4 {
        for x in mbx * 4..mbx * 4 + 4 {
            let i = a.idx(x, y);
            for l in 0..2 {
                if a.refidx[l][i] != b.refidx[l][i] {
                    return false;
                }
                if a.refidx[l][i] >= 0 && a.mv[l][i] != b.mv[l][i] {
                    return false;
                }
            }
        }
    }
    true
}

/// The QP of every macroblock under the QP chain of the records.
pub fn qp_chain(mbs: &[Mb], slice_qp: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(mbs.len());
    let mut pred = slice_qp;
    for mb in mbs {
        let q = if mb.skip || mb.kind == Kind::Pcm || (mb.cbp == 0 && mb.kind != Kind::I16) {
            pred
        } else {
            (pred + mb.qpd + 52).rem_euclid(52)
        };
        out.push(q);
        pred = q;
    }
    out
}

/// Re-solve mb_qp_delta so every non-PCM MB keeps its original QP. Returns
/// the number of MBs made to carry a delta (an empty coded chroma DC).
pub fn fix_qp(new: &mut [Mb], target: &[i32], slice_qp: i32) -> Result<usize> {
    let (mut forced, mut pred) = (0, slice_qp);
    for (mb, &tq) in new.iter_mut().zip(target) {
        if mb.kind == Kind::Pcm {
            continue;
        }
        let mut codes = !mb.skip && (mb.cbp != 0 || mb.kind == Kind::I16);
        if !codes && tq != pred {
            // make it carry a qp delta: empty coded chroma DC (no coefficients)
            if mb.skip {
                bail!("skip MB needing QP change must be converted first");
            }
            mb.cbp |= 0x10;
            if mb.coef.is_none() {
                mb.coef = Some(Box::<Coefs>::default());
            }
            codes = true;
            forced += 1;
        }
        if codes {
            mb.qpd = (tq - pred + 26).rem_euclid(52) - 26;
            pred = tq;
        }
    }
    Ok(forced)
}

/// Rewrite a skip / direct MB with explicit partitions carrying `motion`
/// (B: explicit 8x8 blocks, or with `compact` the coarsest partitioning that
/// fits; P: an explicit 16x16 with the mv left for target mode, `motion`
/// unused).
pub fn to_explicit(mb: &Mb, t: SliceType, motion: Option<&Motion>, compact: bool) -> Mb {
    if t == SliceType::P {
        let mut n = mb.clone();
        n.skip = false;
        n.kind = Kind::P;
        n.part = Some(Part::P16x16);
        n.refs = Some([vec![Some(0)], vec![None]]);
        n.mvds = Some(Mvds::Parts([vec![Some([0, 0])], vec![None]]));
        n.direct16 = false;
        if mb.skip {
            clear_residual(&mut n);
        }
        return n;
    }
    let motion = motion.expect("to_explicit for a B macroblock needs its motion");
    if compact { reexpress_compact(mb, motion) } else { reexpress(mb, SliceType::B, motion) }
}

// ---------------------------------------------------------------------------
// the API of patch_pictures
// ---------------------------------------------------------------------------

/// The stream and the part of it `patch_pictures` works on.
pub struct PatchCtx<'a> {
    pub es: &'a [u8],
    pub spss: &'a Spss,
    pub ppss: &'a Ppss,
    /// All pictures of the stream, indexed by decode index (after `dpb_walk`).
    pub pics: &'a [Picture],
    /// Decode indices of the pictures that take part (one IDR period for the
    /// repair): the patched pictures, their references, and the B pictures
    /// that may follow them. Only these are scanned for B pictures whose
    /// co-located picture changed.
    pub range: Range<usize>,
}

/// Source of the replacement records (Python's `yuv_of`): I_PCM records, or
/// inter records with a corrected residual (same motion, same QP).
pub trait Targets {
    /// The replacement record of macroblock (mbx, mby) of picture `dec`.
    /// `x` and `y` are set by the caller.
    fn pcm(&self, dec: usize, mbx: u16, mby: u16) -> Result<Mb>;
    /// Whether the replacement is intra (I_PCM). An inter replacement keeps
    /// its motion, so direct blocks that use it as co-located MB stay valid.
    fn intra(&self, _dec: usize, _mbx: u16, _mby: u16) -> bool {
        true
    }
    /// A version of the replacement: a cached picture is re-encoded when the
    /// version of one of its replacements changes.
    fn stamp(&self, _dec: usize, _mbx: u16, _mby: u16) -> u64 {
        0
    }
    /// The luma QP of an inter replacement, when it is not the original MB's.
    fn qp(&self, _dec: usize, _mbx: u16, _mby: u16) -> Option<i32> {
        None
    }
}

impl<F: Fn(usize, u16, u16) -> Result<Mb>> Targets for F {
    fn pcm(&self, dec: usize, mbx: u16, mby: u16) -> Result<Mb> {
        self(dec, mbx, mby)
    }
}

/// A parsed picture: its records (never modified), and the motion field to
/// keep for it (Python's `(decoded picture, motion)` pair of the `parsed`
/// dict).
#[derive(Debug)]
pub struct ParsedPic {
    pub dd: Parsed,
    pub motion: Arc<Motion>,
}

pub type PicRef = Arc<ParsedPic>;

/// Parsed pictures by decode index (Python's `parsed` dict or `_ParsedLRU`).
pub trait ParsedStore {
    /// The picture if it is stored or can be recomputed by the store itself
    /// (a "reload" key); `Ok(None)` if the caller has to parse it.
    fn get(&mut self, dec: usize) -> Result<Option<PicRef>>;
    fn put(&mut self, dec: usize, v: PicRef);
}

/// An unbounded map (the exact repair pre-fills it with every picture it needs).
#[derive(Default)]
pub struct MapStore(pub HashMap<usize, PicRef>);

impl ParsedStore for MapStore {
    fn get(&mut self, dec: usize) -> Result<Option<PicRef>> {
        Ok(self.0.get(&dec).cloned())
    }
    fn put(&mut self, dec: usize, v: PicRef) {
        self.0.insert(dec, v);
    }
}

/// Parsed pictures, at most `cap` of them in memory (each holds ~25-45 MB of
/// records and motion). Pictures in `reload_keys` (B pictures that keep
/// their original motion) are recomputed by `reload` when evicted; anything
/// else evicted is simply parsed again by `patch_pictures`. Python
/// `repair._ParsedLRU`.
///
/// With [`LruStore::pinned`], the first pictures stored are never evicted:
/// a measured repair walks its whole period every round, and plain LRU over
/// a cycle longer than `cap` evicts every picture before it is needed again.
pub struct LruStore<'a> {
    cap: usize,
    pin_cap: usize,
    clock: u64,
    /// (last use, pinned, picture)
    map: HashMap<usize, (u64, bool, PicRef)>,
    reload_keys: HashSet<usize>,
    reload: Box<dyn FnMut(usize) -> Result<ParsedPic> + 'a>,
}

impl<'a> LruStore<'a> {
    pub fn new(cap: usize, reload_keys: impl IntoIterator<Item = usize>, reload: impl FnMut(usize) -> Result<ParsedPic> + 'a) -> Self {
        LruStore { cap, pin_cap: 0, clock: 0, map: HashMap::new(), reload_keys: reload_keys.into_iter().collect(), reload: Box::new(reload) }
    }

    /// Never evict the first `n` pictures stored (`n` < `cap`; the rest of
    /// the capacity stays LRU).
    pub fn pinned(mut self, n: usize) -> Self {
        self.pin_cap = n.min(self.cap.saturating_sub(1));
        self
    }

    /// Number of stored pictures.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

impl ParsedStore for LruStore<'_> {
    fn get(&mut self, dec: usize) -> Result<Option<PicRef>> {
        if let Some((stamp, _, v)) = self.map.get_mut(&dec) {
            self.clock += 1;
            *stamp = self.clock;
            return Ok(Some(v.clone()));
        }
        if self.reload_keys.contains(&dec) {
            let v = Arc::new((self.reload)(dec)?);
            self.put(dec, v.clone());
            return Ok(Some(v));
        }
        Ok(None)
    }

    fn put(&mut self, dec: usize, v: PicRef) {
        self.clock += 1;
        let pin = match self.map.get(&dec) {
            Some(e) => e.1,
            None => self.map.values().filter(|e| e.1).count() < self.pin_cap,
        };
        self.map.insert(dec, (self.clock, pin, v));
        while self.map.len() > self.cap {
            let oldest = *self.map.iter().filter(|(_, e)| !e.1).min_by_key(|(_, e)| e.0).map(|(k, _)| k).expect("an unpinned entry");
            self.map.remove(&oldest);
        }
    }
}

/// What a picture's result depends on: its own patch set and the state of
/// its co-located picture (Python: `(sorted targets, sigs.get(col))`).
#[derive(Debug, PartialEq, Eq)]
pub struct Sig {
    targets: Vec<(u16, u16, u64)>,
    col: Option<Arc<Sig>>,
}

struct CacheEntry {
    sig: Arc<Sig>,
    motion: Option<Arc<Motion>>,
    nal: Option<Nal>,
    stats: Option<PatchStats>,
}

/// Results kept by the caller across calls with growing `patches`: a picture
/// whose own patch set and co-located picture's state are unchanged is not
/// re-encoded (Python's `cache` dict).
#[derive(Default)]
pub struct PatchCache(HashMap<usize, CacheEntry>);

/// Per-picture statistics (Python's `stats[d]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchStats {
    pub typ: SliceType,
    pub pcm: usize,
    /// Inter MBs with a corrected residual.
    pub corrected: usize,
    pub direct_explicit: usize,
    pub skip_explicit: usize,
    pub qp_carrier: usize,
    pub bytes_before: usize,
    pub bytes_after: usize,
}

impl PatchStats {
    /// The Python dict: `type, pcm, direct_explicit, skip_explicit, qp_carrier, bytes_before, bytes_after`.
    pub fn to_json(&self) -> Value {
        let mut v = json!({"type": self.typ.as_str(), "pcm": self.pcm, "direct_explicit": self.direct_explicit,
               "skip_explicit": self.skip_explicit, "qp_carrier": self.qp_carrier,
               "bytes_before": self.bytes_before, "bytes_after": self.bytes_after});
        if self.corrected > 0 {
            v["corrected"] = json!(self.corrected);
        }
        v
    }
}

#[derive(Default)]
pub struct PatchResult {
    /// Replacement NALs by decode index.
    pub repl: BTreeMap<usize, Nal>,
    /// Statistics of the pictures in `repl` (including those taken from the cache).
    pub stats: BTreeMap<usize, PatchStats>,
}

/// Parse picture `d` (if the store does not have it) with the motion of its
/// original context.
fn load(ctx: &PatchCtx, store: &mut dyn ParsedStore, d: usize) -> Result<PicRef> {
    if let Some(v) = store.get(d)? {
        return Ok(v);
    }
    let p = &ctx.pics[d];
    let sps = ctx.spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let dd = decode_picture(ctx.es, ctx.spss, ctx.ppss, p)?;
    let cm = if p.typ() == SliceType::B {
        let col = *p.lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?;
        Some(load(ctx, store, col)?)
    } else {
        None
    };
    let motion = derive_motion(p, &dd.mbs, sps.mb_w, sps.mb_h, ctx.pics, cm.as_ref().map(|c| &*c.motion))
        .with_context(|| format!("dec {d}"))?;
    let v = Arc::new(ParsedPic { dd, motion: Arc::new(motion) });
    store.put(d, v.clone());
    Ok(v)
}

/// Compute replacement slice NALs for the patched pictures.
///
/// * `patches`: `{decode idx: [(mbx, mby), ...]}`, the macroblocks to replace
///   with I_PCM records from `targets`.
/// * Pictures rewritten: the patched ones, those in `force` (e.g. B pictures
///   whose co-located picture changed), and the B pictures in `ctx.range`
///   that use a patched picture as co-located picture. A picture whose
///   records need no change is not rewritten.
/// * `parsed`: parsed pictures (see [`ParsedStore`]).
/// * `cache`: see [`PatchCache`].
/// * `log` receives one line per newly encoded picture.
pub fn patch_pictures(
    ctx: &PatchCtx,
    patches: &BTreeMap<usize, Vec<(u16, u16)>>,
    targets_src: &dyn Targets,
    parsed: &mut dyn ParsedStore,
    force: &BTreeSet<usize>,
    mut cache: Option<&mut PatchCache>,
    log: &mut dyn FnMut(&str),
) -> Result<PatchResult> {
    let pics = ctx.pics;
    let sps = ctx.spss.get(&0).ok_or_else(|| anyhow!("KeyError: 0 (no SPS with id 0)"))?;
    let (mb_w, mb_h) = (sps.mb_w, sps.mb_h);

    // pictures to rewrite: patched ones, plus B pictures that use a patched
    // reference picture as co-located picture (they follow it in decode order)
    let mut todo: BTreeSet<usize> = patches.keys().chain(force.iter()).copied().collect();
    for k in ctx.range.clone() {
        let p = &pics[k];
        if p.typ() == SliceType::B {
            if let Some(&c) = p.lists[1].first() {
                if k > c && patches.contains_key(&c) {
                    todo.insert(k);
                }
            }
        }
    }

    let mut new_motion: HashMap<usize, Arc<Motion>> = HashMap::new();
    let mut res = PatchResult::default();
    let mut sigs: HashMap<usize, Arc<Sig>> = HashMap::new();

    for &d in &todo {
        let p = &pics[d];
        let t = p.typ();
        let col = if t == SliceType::B { Some(*p.lists[1].first().ok_or_else(|| anyhow!("list index out of range"))?) } else { None };
        let targets: BTreeSet<(u16, u16)> = patches.get(&d).map(|v| v.iter().copied().collect()).unwrap_or_default();
        let col_pcm: BTreeSet<(u16, u16)> = col
            .and_then(|c| patches.get(&c).map(|v| v.iter().copied().filter(|&(x, y)| targets_src.intra(c, x, y)).collect()))
            .unwrap_or_default();
        // nothing this picture depends on changed since the cached result:
        // reuse it without parsing anything
        let sig = Arc::new(Sig {
            targets: targets.iter().map(|&(x, y)| (x, y, targets_src.stamp(d, x, y))).collect(),
            col: col.and_then(|c| sigs.get(&c).cloned()),
        });
        sigs.insert(d, sig.clone());
        if let Some(c) = cache.as_deref().and_then(|c| c.0.get(&d)) {
            if *c.sig == *sig {
                if let Some(m) = &c.motion {
                    new_motion.insert(d, m.clone());
                }
                if let Some(nal) = &c.nal {
                    res.repl.insert(d, nal.clone());
                    res.stats.insert(d, c.stats.clone().expect("a cached NAL has statistics"));
                }
                continue;
            }
        }
        let col_changed = !col_pcm.is_empty() && col.is_some_and(|c| res.repl.contains_key(&c));
        if targets.is_empty() && !col_changed && !force.contains(&d) {
            if let Some(c) = cache.as_deref_mut() {
                c.0.insert(d, CacheEntry { sig, motion: None, nal: None, stats: None });
            }
            continue;
        }
        let pd = load(ctx, parsed, d)?;
        let dd = &pd.dd;
        let m0: &Motion = &pd.motion;
        if let Some(c) = col {
            let cp = load(ctx, parsed, c)?;
            new_motion.entry(c).or_insert_with(|| cp.motion.clone());
        }
        let mut new: Vec<Mb> = dd.mbs.iter().map(copy_mb).collect();
        let mut n_corr = 0usize;
        if !targets.is_empty() {
            for mb in new.iter_mut() {
                if targets.contains(&(mb.x, mb.y)) {
                    let mut r = targets_src.pcm(d, mb.x, mb.y)?;
                    r.x = mb.x;
                    r.y = mb.y;
                    if r.kind != Kind::Pcm {
                        n_corr += 1;
                    }
                    *mb = r;
                }
            }
        }
        let mut n_direct = 0usize;
        let compact = p.slices[0].direct_spatial == Some(true);
        if t == SliceType::B {
            // direct blocks whose co-located MB became intra (PCM)
            for mb in new.iter_mut() {
                if mb.kind == Kind::Pcm || mb.intra {
                    continue;
                }
                if !col_pcm.contains(&(mb.x, mb.y)) {
                    continue;
                }
                if has_direct(mb) {
                    *mb = to_explicit(mb, SliceType::B, Some(m0), compact);
                    n_direct += 1;
                }
            }
        }
        // skip / direct MBs whose inferred motion moved (a neighbour became
        // intra, or the co-located motion changed): make explicit, repeat
        // until stable
        let col_motion: Option<Arc<Motion>> = col.and_then(|c| new_motion.get(&c).cloned());
        let mut n_skip = 0usize;
        let mut m1;
        loop {
            // target mode: MBs already made explicit carry the motion to keep
            m1 = derive_motion_target(p, &mut new, mb_w, mb_h, pics, col_motion.as_deref(), m0)
                .with_context(|| format!("dec {d} {}", t.as_str()))?;
            let skips: Vec<usize> = new
                .iter()
                .enumerate()
                .filter(|(_, mb)| {
                    mb.kind != Kind::Pcm
                        && !mb.intra
                        && !same_motion(m0, &m1, mb.x as usize, mb.y as usize)
                        && (mb.skip || (t == SliceType::B && has_direct(mb)))
                })
                .map(|(i, _)| i)
                .collect();
            if skips.is_empty() {
                break;
            }
            for i in skips {
                if new[i].skip {
                    n_skip += 1;
                } else {
                    n_direct += 1;
                }
                new[i] = to_explicit(&new[i], t, Some(m0), compact);
            }
        }
        // a skip MB inherits QP_pred; if a PCM in front of it changed QP_pred,
        // it has to become a coded MB so it can carry mb_qp_delta
        let slice_qp = p.slices[0].qp;
        let mut target_qp = qp_chain(&dd.mbs, slice_qp);
        for (i, mb) in new.iter().enumerate() {
            if targets.contains(&(mb.x, mb.y)) {
                if let Some(q) = targets_src.qp(d, mb.x, mb.y) {
                    target_qp[i] = q;
                }
            }
        }
        let mut pred = slice_qp;
        let mut n_qp = 0usize;
        for i in 0..new.len() {
            if new[i].kind == Kind::Pcm {
                continue;
            }
            if new[i].skip && target_qp[i] != pred {
                new[i] = to_explicit(&new[i], t, Some(m0), compact);
                n_qp += 1;
            }
            pred = target_qp[i];
        }
        // solve mvds so explicit partitions carry the original motion
        let m2 = if n_qp > 0 {
            derive_motion_target(p, &mut new, mb_w, mb_h, pics, col_motion.as_deref(), m0)
                .with_context(|| format!("dec {d} {}", t.as_str()))?
        } else {
            m1
        };
        let bad: Vec<(u16, u16)> = new
            .iter()
            .filter(|mb| mb.kind != Kind::Pcm && !mb.intra && !same_motion(m0, &m2, mb.x as usize, mb.y as usize))
            .map(|mb| (mb.x, mb.y))
            .collect();
        if let Some(&(bx, by)) = bad.first() {
            let mut dump = String::new();
            if let Some(mb) = new.iter().find(|mb| (mb.x, mb.y) == (bx, by)) {
                let mut c = mb.canonical();
                if let Some(o) = c.as_object_mut() {
                    o.remove("coef");
                    o.remove("samples");
                }
                dump = c.to_string();
            }
            bail!("dec {d} {}: motion could not be preserved at {:?}: [{dump}]", t.as_str(), &bad[..bad.len().min(5)]);
        }
        let forced = fix_qp(&mut new, &target_qp, slice_qp)?;
        // the records now carry exactly this motion; only reference pictures
        // can be co-located pictures
        let cm2 = if p.ref_idc != 0 { Some(Arc::new(m2)) } else { None };
        if let Some(m) = &cm2 {
            new_motion.insert(d, m.clone());
        }
        if targets.is_empty() && n_direct == 0 && n_skip == 0 && forced == 0 && new == dd.mbs {
            if let Some(c) = cache.as_deref_mut() {
                c.0.insert(d, CacheEntry { sig, motion: cm2, nal: None, stats: None });
            }
            continue;
        }
        let nal: Nal = Arc::new(try_encode_picture(dd, ctx.spss, ctx.ppss, &mut new).with_context(|| format!("dec {d} {}", t.as_str()))?);
        log(&format!(
            "dec {d} {}: PCM {} MB, {}direct->explicit {n_direct}, skip->explicit {n_skip}, qp-carrier {forced}; {}B -> {}B",
            t.as_str(),
            targets.len() - n_corr,
            if n_corr > 0 { format!("corrected {n_corr} MB, ") } else { String::new() },
            dd.nal_len,
            nal.len()
        ));
        let st = PatchStats {
            typ: t,
            pcm: targets.len() - n_corr,
            corrected: n_corr,
            direct_explicit: n_direct,
            skip_explicit: n_skip,
            qp_carrier: forced,
            bytes_before: dd.nal_len,
            bytes_after: nal.len(),
        };
        res.repl.insert(d, nal.clone());
        res.stats.insert(d, st.clone());
        if let Some(c) = cache.as_deref_mut() {
            c.0.insert(d, CacheEntry { sig, motion: cm2, nal: Some(nal), stats: Some(st) });
        }
    }
    Ok(res)
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// Replacement NALs by decode index.
pub trait NalMap {
    fn nal(&self, dec: usize) -> Option<&[u8]>;
}

impl NalMap for BTreeMap<usize, Nal> {
    fn nal(&self, dec: usize) -> Option<&[u8]> {
        self.get(&dec).map(|v| v.as_slice())
    }
}

impl NalMap for BTreeMap<usize, Vec<u8>> {
    fn nal(&self, dec: usize) -> Option<&[u8]> {
        self.get(&dec).map(|v| v.as_slice())
    }
}

impl NalMap for HashMap<usize, Vec<u8>> {
    fn nal(&self, dec: usize) -> Option<&[u8]> {
        self.get(&dec).map(|v| v.as_slice())
    }
}

/// Start of the first slice NAL's start code (the 4-byte form if the byte before the 3-byte one is zero).
fn slice_start(es: &[u8], p: &Picture) -> usize {
    let sl = &p.slices[0];
    let s0 = sl.offset - 3;
    if es[s0 - 1] == 0 { s0 - 1 } else { s0 }
}

/// Write pictures `start..end` of the stream (from the access unit of
/// `start`, i.e. including the parameter sets in front of an IDR) with the
/// NALs in `repl` substituted, into `out`. Python `verify_motion.segment`.
pub fn write_segment<R: NalMap + ?Sized>(es: &[u8], pics: &[Picture], start: usize, end: usize, repl: &R, out: &mut impl Write) -> Result<()> {
    let mut pos: Option<usize> = None;
    for p in &pics[start..end] {
        let sl = &p.slices[0];
        let s0 = slice_start(es, p);
        let pos0 = *pos.get_or_insert(p.au_start);
        out.write_all(&es[pos0..s0])?;
        match repl.nal(p.dec) {
            Some(nal) => {
                out.write_all(&[0, 0, 0, 1])?;
                out.write_all(nal)?;
            }
            None => out.write_all(&es[s0..sl.offset + sl.size])?,
        }
        pos = Some(sl.offset + sl.size);
    }
    Ok(())
}

/// [`write_segment`] into memory.
pub fn segment_bytes<R: NalMap + ?Sized>(es: &[u8], pics: &[Picture], start: usize, end: usize, repl: &R) -> Vec<u8> {
    let mut v = Vec::new();
    write_segment(es, pics, start, end, repl, &mut v).expect("writing to memory cannot fail");
    v
}

/// Write the whole stream with the NALs in `repl` substituted (Python
/// `write_patched`).
pub fn write_patched<R: NalMap + ?Sized>(es: &[u8], pics: &[Picture], repl: &R, out_path: &Path) -> Result<()> {
    let f = std::fs::File::create(out_path).with_context(|| format!("cannot create {}", out_path.display()))?;
    let mut o = std::io::BufWriter::with_capacity(1 << 20, f);
    let mut pos = 0usize;
    for p in pics {
        let sl = &p.slices[0];
        let s0 = slice_start(es, p);
        o.write_all(&es[pos..s0])?;
        match repl.nal(p.dec) {
            Some(nal) => {
                o.write_all(&[0, 0, 0, 1])?;
                o.write_all(nal)?;
            }
            None => o.write_all(&es[s0..sl.offset + sl.size])?,
        }
        pos = sl.offset + sl.size;
    }
    o.write_all(&es[pos..])?;
    o.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// dev patch-replay
// ---------------------------------------------------------------------------

/// Parsed pictures kept per measured-repair worker (Python `repair.PARSED_CAP`).
pub const PARSED_CAP: usize = 12;

struct MappedStream {
    map: Option<memmap2::Mmap>,
    spss: Spss,
    ppss: Ppss,
    pics: Vec<Picture>,
}

impl MappedStream {
    fn open(path: &str) -> Result<MappedStream> {
        let (spss, ppss, pics) = crate::stream::load(Path::new(path))?;
        let file = std::fs::File::open(path).with_context(|| format!("cannot open {path}"))?;
        let map = Some(unsafe { memmap2::Mmap::map(&file)? });
        Ok(MappedStream { map, spss, ppss, pics })
    }

    fn es(&self) -> &[u8] {
        self.map.as_deref().unwrap_or(&[])
    }
}

fn json_usize_map(v: &Value) -> Vec<(usize, &Value)> {
    v.as_object().map(|o| o.iter().map(|(k, v)| (k.parse().expect("numeric key"), v)).collect()).unwrap_or_default()
}

/// The records of picture `d` of the original stream with its motion in the
/// original context (Python `run_task_explicit.orig_motion`).
fn orig_motion(orig: &MappedStream, d: usize) -> Result<(Parsed, Motion)> {
    let sps = orig.spss.get(&0).ok_or_else(|| anyhow!("no SPS 0"))?;
    let po = &orig.pics[d];
    let dd = decode_picture(orig.es(), &orig.spss, &orig.ppss, po)?;
    let col = if po.typ() == SliceType::B { Some(orig_motion(orig, *po.lists[1].first().ok_or_else(|| anyhow!("no list 1"))?)?.1) } else { None };
    let m = derive_motion(po, &dd.mbs, sps.mb_w, sps.mb_h, &orig.pics, col.as_ref())?;
    Ok((dd, m))
}

/// Records of the B picture `nd` of the new stream under its new header, with
/// the motion it had in the original (its co-located picture there was D).
/// Python `run_task_explicit.keep_original_motion`.
fn keep_original_motion(orig: &MappedStream, new: &MappedStream, orig_of_new: &HashMap<usize, usize>, nd: usize) -> Result<ParsedPic> {
    let od = *orig_of_new.get(&nd).ok_or_else(|| anyhow!("KeyError: {nd}"))?;
    let (dd, mo) = orig_motion(orig, od)?;
    let ns = &new.pics[nd].slices[0];
    let nal = &new.es()[ns.offset..ns.offset + ns.size];
    let mut head = crate::bits::unescape(&nal[1..]);
    head.truncate(ns.data_bit / 8);
    let parsed = Parsed { nal_header: nal[0], nal_len: nal.len(), sl: ns.clone(), head, mbs: dd.mbs, end_bit: dd.end_bit };
    Ok(ParsedPic { dd: parsed, motion: Arc::new(mo) })
}

/// Result of [`replay_corpus`].
#[derive(Debug, Default)]
pub struct ReplaySummary {
    pub calls: usize,
    pub nals: usize,
    pub problems: usize,
    pub python_seconds: f64,
    pub rust_seconds: f64,
}

/// `frame-dedup dev patch-replay <corpus.json> [--task PERIOD]...`: replay the
/// `patch_pictures` calls recorded from Python's repair tasks (port/oracle.py
/// `patch-record`) and compare the NALs (length and SHA-256), the stats and the
/// log lines of every call. Exits with an error if anything differs.
pub fn dev_patch_replay(args: &[String]) -> Result<()> {
    const USAGE: &str = "frame-dedup dev patch-replay <corpus.json> [--task PERIOD]...";
    let mut path = None;
    let mut only: Vec<i64> = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--task" => only.push(it.next().ok_or_else(|| anyhow!("--task needs a value"))?.parse()?),
            _ if path.is_none() => path = Some(a.clone()),
            _ => bail!("usage: {USAGE}"),
        }
    }
    let path = path.ok_or_else(|| anyhow!("usage: {USAGE}"))?;
    let sum = replay_corpus(&path, &only, &mut |l| println!("{l}"))?;
    println!(
        "{} calls, {} NALs compared, {} problems; python {:.1}s, rust {:.1}s",
        sum.calls, sum.nals, sum.problems, sum.python_seconds, sum.rust_seconds
    );
    if sum.problems > 0 {
        bail!("{} differences from the Python recording", sum.problems);
    }
    Ok(())
}

/// Replay a recorded corpus (see [`dev_patch_replay`]); `out` gets one line per call.
pub fn replay_corpus(path: &str, only: &[i64], out: &mut dyn FnMut(String)) -> Result<ReplaySummary> {
    use crate::picture::{hex, sha256};
    let corpus: Value = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?)).context("bad corpus")?;
    let explicit = corpus["explicit"].as_bool().unwrap_or(false);
    let t_load = std::time::Instant::now();
    let new = MappedStream::open(corpus["new_es"].as_str().ok_or_else(|| anyhow!("no new_es"))?)?;
    let orig = if explicit { Some(MappedStream::open(corpus["orig_es"].as_str().ok_or_else(|| anyhow!("no orig_es"))?)?) } else { None };
    eprintln!("streams loaded in {:.1}s", t_load.elapsed().as_secs_f64());
    let mut sum = ReplaySummary::default();
    for task in corpus["tasks"].as_array().ok_or_else(|| anyhow!("no tasks"))? {
        let period = task["period"].as_i64().unwrap_or(-1);
        if !only.is_empty() && !only.contains(&period) {
            continue;
        }
        let lo = task["range"][0].as_u64().unwrap() as usize;
        let hi = task["range"][1].as_u64().unwrap() as usize;
        let force: BTreeSet<usize> = task["force"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect()).unwrap_or_default();
        let orig_of_new: HashMap<usize, usize> =
            json_usize_map(&task["orig_of_new"]).into_iter().map(|(k, v)| (k, v.as_u64().unwrap() as usize)).collect();
        let ctx = PatchCtx { es: new.es(), spss: &new.spss, ppss: &new.ppss, pics: &new.pics, range: lo..hi + 1 };
        let mut cache = PatchCache::default();
        let mut store: Box<dyn ParsedStore + '_> = match &orig {
            Some(orig) => {
                let (orig, new, map) = (orig, &new, &orig_of_new);
                Box::new(LruStore::new(PARSED_CAP, force.iter().copied(), move |nd| keep_original_motion(orig, new, map, nd)))
            }
            None => Box::new(MapStore::default()),
        };
        let mut task_bad = 0usize;
        let calls = task["calls"].as_array().ok_or_else(|| anyhow!("no calls"))?;
        for (ci, call) in calls.iter().enumerate() {
            let mut patches: BTreeMap<usize, Vec<(u16, u16)>> = BTreeMap::new();
            for (d, v) in json_usize_map(&call["patches"]) {
                patches.insert(d, v.as_array().unwrap().iter().map(|p| (p[0].as_u64().unwrap() as u16, p[1].as_u64().unwrap() as u16)).collect());
            }
            let call_force: BTreeSet<usize> = call["force"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap() as usize).collect()).unwrap_or_default();
            let mut samples: HashMap<(usize, u16, u16), Box<[u8; 384]>> = HashMap::new();
            for (d, mbs) in json_usize_map(&call["targets"]) {
                for (key, v) in mbs.as_object().unwrap() {
                    let (x, y) = key.split_once(',').unwrap();
                    let mut s = Box::new([0u8; 384]);
                    for (i, b) in v.as_array().unwrap().iter().enumerate() {
                        s[i] = b.as_u64().unwrap() as u8;
                    }
                    samples.insert((d, x.parse()?, y.parse()?), s);
                }
            }
            let targets = |d: usize, x: u16, y: u16| -> Result<Mb> {
                let s = samples.get(&(d, x, y)).ok_or_else(|| anyhow!("KeyError: no target for picture {d} MB ({x}, {y})"))?;
                Ok(pcm_mb(x, y, s.clone()))
            };
            let mut log_lines: Vec<String> = Vec::new();
            let t0 = std::time::Instant::now();
            let res = patch_pictures(&ctx, &patches, &targets, &mut *store, &call_force, Some(&mut cache), &mut |s| log_lines.push(s.to_string()))
                .with_context(|| format!("task {period} call {ci}"))?;
            let secs = t0.elapsed().as_secs_f64();
            sum.rust_seconds += secs;
            sum.python_seconds += call["seconds"].as_f64().unwrap_or(0.0);
            sum.calls += 1;
            // compare
            let mut problems: Vec<String> = Vec::new();
            let want = json_usize_map(&call["repl"]);
            if want.len() != res.repl.len() || want.iter().any(|(d, _)| !res.repl.contains_key(d)) {
                problems.push(format!(
                    "repl keys differ: python {:?}, rust {:?}",
                    want.iter().map(|(d, _)| *d).collect::<Vec<_>>(),
                    res.repl.keys().collect::<Vec<_>>()
                ));
            }
            for (d, w) in &want {
                if let Some(nal) = res.repl.get(d) {
                    sum.nals += 1;
                    let (wl, wh) = (w["len"].as_u64().unwrap() as usize, w["sha256"].as_str().unwrap());
                    if nal.len() != wl || hex(&sha256(nal)) != wh {
                        problems.push(format!("picture {d}: NAL differs (python {wl} bytes, rust {} bytes)", nal.len()));
                    }
                }
            }
            let stats_py = json_usize_map(&call["stats"]);
            if stats_py.len() != res.stats.len() {
                problems.push(format!("stats keys differ: python {}, rust {}", stats_py.len(), res.stats.len()));
            }
            for (d, w) in &stats_py {
                match res.stats.get(d) {
                    Some(st) if st.to_json() == **w => {}
                    other => problems.push(format!("picture {d}: stats differ: python {w}, rust {:?}", other.map(|s| s.to_json()))),
                }
            }
            let want_log: Vec<&str> = call["log"].as_array().map(|a| a.iter().filter_map(|v| v.as_str()).collect()).unwrap_or_default();
            if want_log != log_lines.iter().map(String::as_str).collect::<Vec<_>>() {
                problems.push(format!("log lines differ: python {want_log:?}, rust {log_lines:?}"));
            }
            out(format!(
                "task {period} call {ci}: {} patched pictures, {} rewritten, python {:.2}s, rust {:.2}s: {}",
                patches.len(),
                res.repl.len(),
                call["seconds"].as_f64().unwrap_or(0.0),
                secs,
                if problems.is_empty() { "identical".to_string() } else { format!("{} PROBLEMS", problems.len()) }
            ));
            for pr in problems.iter().take(5) {
                out(format!("    {pr}"));
            }
            task_bad += problems.len();
        }
        sum.problems += task_bad;
    }
    Ok(sum)
}

/// A slice header for unit tests (a P slice with one active reference).
#[cfg(test)]
pub(crate) fn test_slice_header() -> crate::h264hdr::SliceHeader {
    use crate::h264hdr::SliceHeader;
    SliceHeader {
        first_mb: 0,
        slice_type: SliceType::P,
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
        num_ref_l0: 1,
        num_ref_l1: 0,
        rplm: [Vec::new(), Vec::new()],
        rplm_bit: 0,
        rplm_end_bit: 0,
        weighted: false,
        pwt: None,
        adaptive: false,
        mmco: Vec::new(),
        no_output_prior: None,
        long_term_ref: None,
        drpm_bit: 0,
        drpm_end_bit: 0,
        cabac_init_idc: 0,
        qp: 26,
        deblock_idc: 0,
        alpha_off: 0,
        beta_off: 0,
        hdr_bits: 0,
        data_bit: 0,
        nal_type: 1,
        ref_idc: 0,
        offset: 0,
        size: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mb(skip: bool, kind: Kind, cbp: u8, qpd: i32) -> Mb {
        let mut m = Mb::new(0, 0);
        m.skip = skip;
        m.kind = kind;
        m.cbp = cbp;
        m.qpd = qpd;
        m
    }

    #[test]
    fn qp_chain_and_fix_qp() {
        let mbs = vec![mb(false, Kind::P, 0x10, 3), mb(true, Kind::Skip, 0, 0), mb(false, Kind::I16, 0, -2), mb(false, Kind::P, 0, 5)];
        assert_eq!(qp_chain(&mbs, 26), vec![29, 29, 27, 27]);
        // a PCM in front keeps QP_pred; the next coded MB must restore its original QP
        let mut new = vec![mb(false, Kind::Pcm, 0x2f, 0), mb(false, Kind::P, 0x10, 3), mb(true, Kind::Skip, 0, 0), mb(false, Kind::I16, 0, -2)];
        let target = vec![26, 29, 29, 27];
        assert_eq!(fix_qp(&mut new, &target, 26).unwrap(), 0);
        assert_eq!(new[1].qpd, 3);
        assert_eq!(new[3].qpd, -2);
        // a non-coding MB that has to change QP gets an empty chroma DC and a delta
        let mut new = vec![mb(false, Kind::P, 0, 0)];
        assert_eq!(fix_qp(&mut new, &[30], 26).unwrap(), 1);
        assert_eq!((new[0].cbp, new[0].qpd), (0x10, 4));
        assert!(new[0].coef.is_some());
        // wrap-around of the delta
        let mut new = vec![mb(false, Kind::I16, 0, 0)];
        fix_qp(&mut new, &[2], 50).unwrap();
        assert_eq!(new[0].qpd, 4);
        let mut sk = vec![mb(true, Kind::Skip, 0, 0)];
        assert!(fix_qp(&mut sk, &[30], 26).is_err());
    }

    /// Expected digests from frame_dedup.patch.pcm_record on the same frame.
    #[test]
    fn pcm_record_matches_python() {
        let (w, h) = (5usize, 3usize);
        let yuv: Vec<u8> = (0..w * 16 * h * 16 * 3 / 2).map(|i| ((i * 7 + i / 13 + i / 997) & 255) as u8).collect();
        let want = [
            ((0, 0), "85de09a4e6295d14463903ccb781ddc534709a51486faddccd3544cf309cf119"),
            ((2, 1), "ee18b8ece4bd5d4bf387a3e067395eb20d93a40a3a9b8b881838008ec54687eb"),
            ((4, 2), "3247e534780a1390c97ba1248c7ca9ba2255e5db998c09114ee628c3faad27fd"),
        ];
        for ((x, y), digest) in want {
            let r = pcm_record(&yuv, w, x, y);
            assert_eq!(crate::picture::hex(&crate::picture::sha256(&r.samples.unwrap()[..])), digest);
        }
    }

    #[test]
    fn pcm_record_layout() {
        // a 2x1 MB frame (32x16): Y = x + 100*y, U = 50 + x, V = 200 - x
        let (w, h) = (32usize, 16usize);
        let mut yuv = Vec::new();
        for y in 0..h {
            for x in 0..w {
                yuv.push((x + y * 7) as u8);
            }
        }
        for x in 0..(w / 2) * (h / 2) {
            yuv.push(50 + (x % 16) as u8);
        }
        for x in 0..(w / 2) * (h / 2) {
            yuv.push(200 - (x % 16) as u8);
        }
        let r = pcm_record(&yuv, 2, 1, 0);
        let s = r.samples.unwrap();
        assert_eq!((r.x, r.y, r.kind, r.cbp, r.pcm, r.intra), (1, 0, Kind::Pcm, 0x2f, true, true));
        assert_eq!(s[0], 16);
        assert_eq!(s[15], 31);
        assert_eq!(s[16], 16 + 7);
        assert_eq!(s[256], 50 + 8);
        assert_eq!(s[256 + 64], 200 - 8);
    }

    #[test]
    fn lru_store_evicts_oldest_and_reloads() {
        let make = |d: usize| ParsedPic {
            dd: Parsed { nal_header: 1, nal_len: d, sl: super::test_slice_header(), head: vec![], mbs: vec![], end_bit: 0 },
            motion: Arc::new(Motion::new(1, 1)),
        };
        let mut reloads = Vec::new();
        {
            let mut s = LruStore::new(2, [7], |d| {
                reloads.push(d);
                Ok(make(d))
            });
            s.put(1, Arc::new(make(1)));
            s.put(2, Arc::new(make(2)));
            assert!(s.get(1).unwrap().is_some()); // touch 1: 2 is now the oldest
            s.put(3, Arc::new(make(3)));
            assert!(s.get(2).unwrap().is_none());
            assert!(s.get(1).unwrap().is_some());
            assert_eq!(s.get(7).unwrap().unwrap().dd.nal_len, 7); // reload key: recomputed
            assert_eq!(s.len(), 2);
            assert!(s.get(3).unwrap().is_none()); // evicted by the reload's insertion
        }
        assert_eq!(reloads, vec![7]);
    }
}

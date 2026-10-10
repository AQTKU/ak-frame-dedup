//! Duplicate-cadence model and detector. Port of `frame_dedup/cadence.py`.
//!
//! Model: input frame n shows source frame floor(n*r + phi), r = S/T (source
//! rate over input rate). Frame n repeats frame n-1 iff
//! frac((n-1)*r + phi) < 1 - r.
//!
//! With r = p/q in lowest terms every arc endpoint is a multiple of 1/q, so
//! the score of a phase is piecewise constant on [m/q, (m+1)/q) and phi = m/q
//! for m in 0..q-1 covers every distinct duplicate pattern. Frame n is a
//! duplicate under phase m iff ((n-1)*p + m) mod q < q - p.
//!
//! For a segment [a, b) of frames with evidence e, bin each frame's evidence
//! by c = ((n-1)*p) mod q; the score of phase m is the sum of the bins c with
//! (c + m) mod q < q - p, i.e. a circular window sum: the exact sweep over the
//! circle in O(L + q).
//!
//! Per candidate ratio the per-segment phases are chosen by dynamic
//! programming: continuing the previous segment's phase is free, changing it
//! costs `PEN_BLACK` at a black-run boundary and `PEN_CUT` at a hard cut
//! inside a chunk. The ratio with the best total wins.
//!
//! Also here: `analyse`, the numerical part of `Run.analyse` (evidence,
//! black runs, cuts, segments, candidates) with the report's `detection` and
//! `cadence` sections as serialisable structs.

use std::collections::BTreeMap;
use std::fmt;

use anyhow::{Context, Result, bail};
use serde::Serialize;

use crate::evidence::{self, Features, Model, np};

// --------------------------------------------------------------------------
// exact rationals

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// A rational number in lowest terms with a positive denominator (Python's `Fraction`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ratio {
    pub num: i64,
    pub den: i64,
}

impl Ratio {
    pub fn new(num: i64, den: i64) -> Ratio {
        assert!(den != 0, "zero denominator");
        let g = gcd(num, den).max(1);
        let s = if den < 0 { -1 } else { 1 };
        Ratio { num: s * num / g, den: s * den / g }
    }

    pub fn int(n: i64) -> Ratio {
        Ratio { num: n, den: 1 }
    }

    /// Python `float(Fraction)`: numerator / denominator, correctly rounded
    /// (both are exactly representable here).
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn div(self, o: Ratio) -> Ratio {
        Ratio::new((self.num as i128 * o.den as i128) as i64, (self.den as i128 * o.num as i128) as i64)
    }

    pub fn mul(self, o: Ratio) -> Ratio {
        Ratio::new(self.num * o.num, self.den * o.den)
    }
}

impl PartialOrd for Ratio {
    fn partial_cmp(&self, o: &Ratio) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

impl Ord for Ratio {
    fn cmp(&self, o: &Ratio) -> std::cmp::Ordering {
        (self.num as i128 * o.den as i128).cmp(&(o.num as i128 * self.den as i128))
    }
}

/// `str(Fraction)`: "4/5", or "24" for an integer.
impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 { write!(f, "{}", self.num) } else { write!(f, "{}/{}", self.num, self.den) }
    }
}

/// Candidate source rates, in the Python dict's order.
pub fn source_rates() -> [(&'static str, Ratio); 3] {
    [("23.976", Ratio::new(24000, 1001)), ("24", Ratio::int(24)), ("25", Ratio::int(25))]
}

/// Accepted input rates.
pub fn input_rates() -> [(&'static str, Ratio); 2] {
    [("29.97", Ratio::new(30000, 1001)), ("30", Ratio::int(30))]
}

/// Nats to change phase at a black run of >= 0.5 s.
pub const PEN_BLACK: f64 = 25.0;
/// Nats to change phase at a hard cut inside a chunk.
pub const PEN_CUT: f64 = 150.0;
/// Minimum (mean evidence of predicted dups) - (mean evidence).
pub const FAIL_SCORE: f64 = 2.0;
/// Predicted duplicates with usable (non-static) evidence.
pub const MIN_EVIDENCE_DUPS: usize = 200;

pub fn rate_name(fr: Ratio, names: &[(&'static str, Ratio)]) -> Option<&'static str> {
    names.iter().find(|(_, v)| *v == fr).map(|(k, _)| *k)
}

// --------------------------------------------------------------------------

/// Exact score of every phase m (0..q-1) over frames [a, b).
pub fn phase_scores(e: &[f64], a: usize, b: usize, p: i64, q: i64) -> Vec<f64> {
    let qu = q as usize;
    let mut h = vec![0.0f64; qu];
    for n in a..b {
        let c = ((n as i64 - 1) * p).rem_euclid(q) as usize;
        h[c] += e[n];
    }
    // C = concat([0], cumsum(concat([h, h])))
    let mut cs = Vec::with_capacity(2 * qu + 1);
    cs.push(0.0);
    let mut acc = 0.0;
    for k in 0..2 * qu {
        acc += h[k % qu];
        cs.push(acc);
    }
    let w = (q - p) as usize;
    (0..qu)
        .map(|m| {
            let s = (qu - m) % qu;
            cs[s + w] - cs[s]
        })
        .collect()
}

pub fn dup_mask(a: usize, b: usize, p: i64, q: i64, m: usize) -> Vec<bool> {
    (a..b).map(|n| ((n as i64 - 1) * p + m as i64).rem_euclid(q) < q - p).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegKind {
    Start,
    Black,
    Cut,
}

impl SegKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SegKind::Start => "start",
            SegKind::Black => "black",
            SegKind::Cut => "cut",
        }
    }
}

/// A stretch of frames [start, end) over which the cadence phase is constant
/// unless the dynamic programme changes it at `kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start: usize,
    pub end: usize,
    pub kind: SegKind,
}

/// Split [0, N) into segments. A black run starts a new segment (the run
/// belongs to the segment after it); a hard cut starts a new segment too.
pub fn segments_from(n: usize, black: &[(usize, usize)], cuts: &[usize]) -> Vec<Segment> {
    let mut bounds: BTreeMap<usize, SegKind> = BTreeMap::new();
    bounds.insert(0, SegKind::Start);
    for &(s, _) in black {
        if s > 0 {
            bounds.insert(s, SegKind::Black);
        }
    }
    for &c in cuts {
        if !bounds.contains_key(&c) && 0 < c && c < n {
            // a cut inside (or right after) a black run is not a separate boundary
            if black.iter().any(|&(s, e)| s <= c && c <= e) {
                continue;
            }
            bounds.insert(c, SegKind::Cut);
        }
    }
    let keys: Vec<usize> = bounds.keys().copied().collect();
    keys.iter()
        .enumerate()
        .map(|(i, &k)| Segment { start: k, end: keys.get(i + 1).copied().unwrap_or(n), kind: bounds[&k] })
        .collect()
}

/// Frames that start a new shot: a large difference both absolutely and
/// relative to the neighbourhood. (Python also takes `e_static`, unused.)
pub fn hard_cuts(feat: &Features) -> Vec<usize> {
    hard_cuts_with(feat, 4.0, 12.0)
}

pub fn hard_cuts_with(feat: &Features, rel_thresh: f64, abs_thresh: f64) -> Vec<usize> {
    let r = evidence::rel_diff(&feat.mad);
    (1..feat.mad.len()).filter(|&i| r[i] > rel_thresh && feat.mad[i] > abs_thresh).collect()
}

fn argmax(v: &[f64]) -> usize {
    // np.argmax: first maximum
    let mut b = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b
}

/// scores: per-segment phase-score vectors (length q each). Returns
/// (total, phases per segment).
pub fn dp_phases(scores: &[Vec<f64>], kinds: &[SegKind], pen_black: f64, pen_cut: f64) -> (f64, Vec<usize>) {
    let mut v = scores[0].clone();
    let mut back: Vec<Vec<usize>> = Vec::new();
    for (s, &kind) in scores[1..].iter().zip(&kinds[1..]) {
        let pen = if kind == SegKind::Black { pen_black } else { pen_cut };
        let best = argmax(&v);
        let switch = v[best] - pen;
        let mut bk = Vec::with_capacity(v.len());
        for i in 0..v.len() {
            let stay = v[i] >= switch;
            bk.push(if stay { i } else { best });
            v[i] = (if stay { v[i] } else { switch }) + s[i];
        }
        back.push(bk);
    }
    let mut m = argmax(&v);
    let total = v[m];
    let mut phases = vec![m];
    for b in back.iter().rev() {
        m = b[m];
        phases.push(m);
    }
    phases.reverse();
    (total, phases)
}

/// One candidate source rate with its best per-segment phases.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub source: &'static str,
    pub rate: Ratio,
    pub r: Ratio,
    pub p: i64,
    pub q: i64,
    pub total: f64,
    pub phases: Vec<usize>,
    pub segment_scores: Vec<f64>,
}

/// Score every candidate source rate. Best first (stable sort on -total).
pub fn detect(e: &[f64], segments: &[Segment], in_rate: Ratio) -> Vec<Candidate> {
    detect_with(e, segments, in_rate, PEN_BLACK, PEN_CUT)
}

pub fn detect_with(e: &[f64], segments: &[Segment], in_rate: Ratio, pen_black: f64, pen_cut: f64) -> Vec<Candidate> {
    let mut out = Vec::new();
    let kinds: Vec<SegKind> = segments.iter().map(|s| s.kind).collect();
    for (name, src) in source_rates() {
        let r = src.div(in_rate);
        if !(Ratio::new(1, 2) < r && r < Ratio::int(1)) {
            continue;
        }
        let (p, q) = (r.num, r.den);
        let sc: Vec<Vec<f64>> = segments.iter().map(|s| phase_scores(e, s.start, s.end, p, q)).collect();
        let (total, phases) = dp_phases(&sc, &kinds, pen_black, pen_cut);
        let segment_scores = sc.iter().zip(&phases).map(|(s, &m)| s[m]).collect();
        out.push(Candidate { source: name, rate: src, r, p, q, total, phases, segment_scores });
    }
    // Python: out.sort(key=lambda c: -c['total']) (stable)
    out.sort_by(|a, b| (-a.total).partial_cmp(&(-b.total)).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// The duplicates of a candidate: each segment's phase mask, except the first
/// frame after a hard cut ([`cut_repeats`]), which cannot repeat the
/// previous shot.
pub fn predicted_dups(n: usize, segments: &[Segment], cand: &Candidate) -> Vec<bool> {
    let mut dup = vec![false; n];
    for (s, &m) in segments.iter().zip(&cand.phases) {
        dup[s.start..s.end].copy_from_slice(&dup_mask(s.start, s.end, cand.p, cand.q, m));
        if s.kind == SegKind::Cut {
            dup[s.start] = false;
        }
    }
    if n > 0 {
        dup[0] = false;
    }
    dup
}

/// First frames after a hard cut that a candidate's phase mask marks as
/// duplicates; [`predicted_dups`] leaves them out. In native 29.97 shots the
/// phase is arbitrary and often lands there, and with encoders that start
/// each shot with an I picture such a pair is two reference pictures that no
/// drop handles cheaply. The frame count changes at each of them, so the A/V
/// sync correction treats them as events.
pub fn cut_repeats(segments: &[Segment], cand: &Candidate) -> Vec<usize> {
    segments
        .iter()
        .zip(&cand.phases)
        .filter(|(s, m)| s.kind == SegKind::Cut && s.start > 0 && dup_mask(s.start, s.start + 1, cand.p, cand.q, **m)[0])
        .map(|(s, _)| s.start)
        .collect()
}

/// Mean evidence of the predicted duplicates minus what a random phase would
/// score (the mean evidence over all frames), both over frames whose
/// evidence is not masked as static. Returns (score, number of such duplicates).
pub fn cadence_score(e: &[f64], static_: &[bool], dup: &[bool]) -> (f64, usize) {
    let mut use_: Vec<bool> = static_.iter().map(|&s| !s).collect();
    if !use_.is_empty() {
        use_[0] = false;
    }
    let sel_d: Vec<f64> = (0..e.len()).filter(|&i| dup[i] && use_[i]).map(|i| e[i]).collect();
    let nd = sel_d.len();
    let sel_u: Vec<f64> = (0..e.len()).filter(|&i| use_[i]).map(|i| e[i]).collect();
    if nd == 0 || sel_u.is_empty() {
        return (0.0, nd);
    }
    (np::mean_f64(&sel_d) - np::mean_f64(&sel_u), nd)
}

// --------------------------------------------------------------------------
// Run.analyse, numerical part

#[derive(Clone, Debug, Serialize)]
pub struct BlackRunReport {
    pub start: usize,
    pub end: usize,
    pub start_s: f64,
    pub frames: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct CandidateReport {
    pub source: String,
    pub ratio: String,
    pub total_llr: f64,
    pub cadence_score: f64,
    pub phase_changes: usize,
}

/// The report's `detection` section.
#[derive(Clone, Debug, Serialize)]
pub struct DetectionReport {
    pub model: Option<String>,
    pub black_runs: Vec<BlackRunReport>,
    pub hard_cuts: usize,
    pub segments: usize,
    pub static_frames: usize,
    pub candidates: Vec<CandidateReport>,
    pub cadence_score: f64,
    pub fail_threshold: f64,
    pub duplicates_with_evidence: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
    pub start_s: f64,
    pub phases: Vec<usize>,
    pub frames: usize,
    pub phi: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PhaseChange {
    pub display: usize,
    pub time_s: f64,
    pub at: &'static str,
    pub from_phase: usize,
    pub to_phase: usize,
    pub q: i64,
}

/// The report's `cadence` section.
#[derive(Clone, Debug, Serialize)]
pub struct CadenceReport {
    pub source_rate: String,
    pub output_fps: String,
    pub ratio: String,
    pub phase_bins: i64,
    pub chunks: Vec<Chunk>,
    pub phase_changes: Vec<PhaseChange>,
}

/// Everything `Run.analyse` computes before its pass / fail check.
#[derive(Clone, Debug)]
pub struct Analysis {
    pub evidence: Vec<f64>,
    pub static_mask: Vec<bool>,
    pub black: Vec<(usize, usize)>,
    pub cuts: Vec<usize>,
    pub segments: Vec<Segment>,
    /// Best first.
    pub candidates: Vec<Candidate>,
    /// Predicted duplicates of the best candidate.
    pub dup: Vec<bool>,
    /// The best candidate's [`cut_repeats`].
    pub cut_repeats: Vec<usize>,
    pub score: f64,
    pub n_evidence: usize,
    pub detection: DetectionReport,
}

impl Analysis {
    pub fn best(&self) -> &Candidate {
        &self.candidates[0]
    }
}

/// Evidence, black runs, cuts, segments, candidates and the `detection`
/// report section for features at the input rate `fps_in`.
pub fn analyse(feat: &Features, fps_in: Ratio, model: &Model, trained_on: Option<&str>) -> Result<Analysis> {
    let n = feat.mad.len();
    let e = evidence::evidence(feat, model);
    let static_ = evidence::static_mask(&feat.bmax);
    let fps = fps_in.to_f64();
    let black = evidence::black_runs(feat, fps);
    let cuts = hard_cuts(feat);
    let segs = segments_from(n, &black, &cuts);
    let cands = detect(&e, &segs, fps_in);
    if cands.is_empty() {
        bail!("no candidate source rate fits an input rate of {fps_in}");
    }
    let dup = predicted_dups(n, &segs, &cands[0]);
    let cut_repeats = cut_repeats(&segs, &cands[0]);
    let (score, n_ev) = cadence_score(&e, &static_, &dup);
    let mut det = DetectionReport {
        model: trained_on.map(str::to_string),
        black_runs: black
            .iter()
            .map(|&(s, x)| BlackRunReport { start: s, end: x, start_s: np::round(s as f64 / fps, 3), frames: x - s })
            .collect(),
        hard_cuts: cuts.len(),
        segments: segs.len(),
        static_frames: static_.iter().filter(|&&s| s).count(),
        candidates: Vec::new(),
        cadence_score: np::round(score, 3),
        fail_threshold: FAIL_SCORE,
        duplicates_with_evidence: n_ev,
    };
    for c in &cands {
        let cd = predicted_dups(n, &segs, c);
        det.candidates.push(CandidateReport {
            source: c.source.to_string(),
            ratio: c.r.to_string(),
            total_llr: np::round(c.total, 1),
            cadence_score: np::round(cadence_score(&e, &static_, &cd).0, 3),
            phase_changes: (1..segs.len()).filter(|&i| c.phases[i] != c.phases[i - 1]).count(),
        });
    }
    Ok(Analysis {
        evidence: e,
        static_mask: static_,
        black,
        cuts,
        segments: segs,
        candidates: cands,
        dup,
        cut_repeats,
        score,
        n_evidence: n_ev,
        detection: det,
    })
}

/// Chunks (split at black runs) and phase changes of the best candidate: the
/// report's `cadence` section.
pub fn cadence_report(a: &Analysis, fps_in: Ratio) -> CadenceReport {
    let best = a.best();
    let fps = fps_in.to_f64();
    let mut chunks: Vec<Chunk> = Vec::new();
    let mut changes = Vec::new();
    for (i, (seg, &m)) in a.segments.iter().zip(&best.phases).enumerate() {
        if matches!(seg.kind, SegKind::Start | SegKind::Black) || chunks.is_empty() {
            chunks.push(Chunk {
                start: seg.start,
                end: seg.end,
                start_s: np::round(seg.start as f64 / fps, 3),
                phases: vec![m],
                frames: 0,
                phi: Vec::new(),
            });
        } else {
            let c = chunks.last_mut().unwrap();
            c.end = seg.end;
            if m != *c.phases.last().unwrap() {
                c.phases.push(m);
            }
        }
        if i > 0 && m != best.phases[i - 1] {
            changes.push(PhaseChange {
                display: seg.start,
                time_s: np::round(seg.start as f64 / fps, 3),
                at: seg.kind.as_str(),
                from_phase: best.phases[i - 1],
                to_phase: m,
                q: best.q,
            });
        }
    }
    for c in &mut chunks {
        c.frames = c.end - c.start;
        c.phi = c.phases.iter().map(|&m| np::round(m as f64 / best.q as f64, 6)).collect();
    }
    CadenceReport {
        source_rate: best.source.to_string(),
        output_fps: best.rate.to_string(),
        ratio: best.r.to_string(),
        phase_bins: best.q,
        chunks,
        phase_changes: changes,
    }
}

// --------------------------------------------------------------------------
// dev command

fn parse_ratio(s: &str) -> Result<Ratio> {
    let (n, d) = s.split_once('/').unwrap_or((s, "1"));
    Ok(Ratio::new(n.parse()?, d.parse()?))
}

/// `dev detect <features.json> <fps num/den>`: evidence, static mask, black
/// runs, cuts, segments, all candidates, best phases, predicted duplicates,
/// cadence score, and the report's `detection` / `cadence` sections.
pub fn dev_detect(args: &[String]) -> Result<()> {
    if args.len() < 2 {
        bail!("usage: dev detect <features.json> <fps num/den>");
    }
    let feat: Features = serde_json::from_str(&std::fs::read_to_string(&args[0])?).context("reading features")?;
    let fps_in = parse_ratio(&args[1])?;
    let (model, trained_on) = evidence::load_model()?;
    let a = analyse(&feat, fps_in, &model, trained_on.as_deref())?;
    let cad = cadence_report(&a, fps_in);
    let v = serde_json::json!({
        "evidence": a.evidence,
        "static": a.static_mask.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i).collect::<Vec<_>>(),
        "black": a.black,
        "cuts": a.cuts,
        "segments": a.segments.iter().map(|s| (s.start, s.end, s.kind.as_str())).collect::<Vec<_>>(),
        "candidates": a.candidates.iter().map(|c| serde_json::json!({
            "source": c.source, "rate": c.rate.to_string(), "r": c.r.to_string(), "p": c.p, "q": c.q,
            "total": c.total, "phases": c.phases, "segment_scores": c.segment_scores})).collect::<Vec<_>>(),
        "predicted_dups": a.dup.iter().enumerate().filter(|(_, s)| **s).map(|(i, _)| i).collect::<Vec<_>>(),
        "cadence_score": [a.score, a.n_evidence],
        "detection": a.detection,
        "cadence": cad,
    });
    println!("{}", serde_json::to_string(&v)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios() {
        let r = Ratio::new(24000, 1001).div(Ratio::new(30000, 1001));
        assert_eq!((r.num, r.den), (4, 5));
        assert_eq!(r.to_string(), "4/5");
        assert_eq!(Ratio::int(24).to_string(), "24");
        assert!(Ratio::new(1, 2) < r && r < Ratio::int(1));
        assert_eq!(Ratio::new(25, 1).div(Ratio::new(30000, 1001)), Ratio::new(1001, 1200));
        assert_eq!(rate_name(Ratio::new(60, 2), &input_rates()), Some("30"));
    }

    /// A phase whose duplicate falls on the first frame after a hard cut: that
    /// frame is not a duplicate (it cannot repeat the previous shot) and is
    /// reported in `cut_repeats`; the rest of the pattern stays.
    #[test]
    fn no_duplicate_on_a_cut() {
        // r = 4/5, phase 4: frame n is a duplicate iff ((n - 1) * 4 + 4) % 5 == 0
        let segs = [
            Segment { start: 0, end: 10, kind: SegKind::Start },
            Segment { start: 10, end: 20, kind: SegKind::Cut },
        ];
        let cand = Candidate {
            source: "24",
            rate: Ratio::int(24),
            r: Ratio::new(4, 5),
            p: 4,
            q: 5,
            total: 0.0,
            phases: vec![4, 4],
            segment_scores: vec![0.0, 0.0],
        };
        let dup = predicted_dups(20, &segs, &cand);
        let at: Vec<usize> = (0..20).filter(|&i| dup[i]).collect();
        assert_eq!(at, vec![5, 15]);
        assert_eq!(cut_repeats(&segs, &cand), vec![10]);
    }

    #[test]
    fn dup_pattern_4_5() {
        // 24 -> 30 fps (r = 4/5): one duplicate in five, at phase m
        let m = dup_mask(0, 10, 4, 5, 0);
        assert_eq!(m.iter().filter(|&&d| d).count(), 2);
        let e: Vec<f64> = m.iter().map(|&d| if d { 5.0 } else { -1.0 }).collect();
        let sc = phase_scores(&e, 0, 10, 4, 5);
        assert_eq!(argmax(&sc), 0);
    }
}

//! End-to-end pipeline: MKV in, constant-frame-rate deduplicated MKV out.
//! Port of `frame_dedup/pipeline.py`.
//!
//! Differences from the Python, all in the report or in log text only:
//! `tool` and `version` name the Rust crate, `resources.main_peak_rss_mb` and
//! `verify.worker_peak_rss_mb` are the process peak (workers are threads), the
//! proxy-feature cache is `features.json` (with the stream size) in the work
//! directory instead of `features.npz`, and the `repair` / `anchor_rebuilds`
//! sections come from the Rust drivers.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use serde::ser::{Serialize, SerializeMap, Serializer};
use serde::Deserialize;
use serde_json::Value;

use crate::cadence::{self, CadenceReport, DetectionReport, FAIL_SCORE, MIN_EVIDENCE_DUPS, Ratio};
use crate::evidence::{self, Features};
use crate::explicit::{self, Sigma};
use crate::h264hdr::{Picture, Ppss, Spss};
use crate::mux::{self, DurationCheck, py_round};
use crate::plan::{self, Freed, Hard, PlanInputs, PlanReport};
use crate::rebuild;
use crate::repair::{self, RepairArgs};
use crate::stream::{self, Unsupported, WriteOpts};
use crate::system::{self, Logger};
use crate::verify::{self, MbFrame, MbFrameInfo, MbOptions, MbReportMeta, MbReportWriter, VerifyResult};

/// Command-line arguments; the same options as `python -m frame_dedup`.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "frame-dedup",
    version = crate::VERSION,
    about = "Remove inserted duplicate frames (23.976/24/25 fps content delivered at 29.97/30 fps) \
             from an H.264 MKV without re-encoding. Writes a constant-frame-rate MKV at the source rate."
)]
pub struct Args {
    /// input .mkv
    pub input: PathBuf,
    /// output .mkv (required unless --dry-run)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// write a JSON report here
    #[arg(long)]
    pub report: Option<PathBuf>,
    /// decode output and original and compare every kept frame by POC
    #[arg(long)]
    pub verify: bool,
    /// with --verify: write every differing frame with its macroblocks at or
    /// above --mb-threshold (location, severity, input and output frame
    /// numbers and times) as JSON here
    #[arg(long)]
    pub mb_report: Option<PathBuf>,
    /// with --mb-report: list the macroblocks whose max difference over Y/U/V
    /// is at least this (default 8)
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u8).range(1..))]
    pub mb_threshold: u8,
    /// work directory for intermediates (default: work/<input name>)
    #[arg(long)]
    pub work: Option<PathBuf>,
    /// keep the work directory
    #[arg(long)]
    pub keep_work: bool,
    /// parallel workers (default: half the logical CPUs); memory-heavy stages
    /// use fewer if RAM is short
    #[arg(long)]
    pub workers: Option<usize>,
    /// never demote an IDR to drop the earlier twin of a pair whose later
    /// twin is an IDR (anchor rebuild); use the fallbacks instead
    #[arg(long)]
    pub no_rebuild: bool,
    /// when an IDR-later pair cannot be rebuilt, keep both twins (A/V
    /// corrected only at black runs) instead of dropping the nearest
    /// low-motion non-reference frame
    #[arg(long)]
    pub keep_idr_pairs: bool,
    /// detect and plan only; write no video
    #[arg(long)]
    pub dry_run: bool,
    /// repair drift with I_PCM only, as version 2.0 did (no residual
    /// corrections)
    #[arg(long)]
    pub no_corrections: bool,
    /// measured repair: correct macroblocks off by this much or more
    /// (default 6; 16 with --no-corrections)
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..))]
    pub drift_threshold: Option<u8>,
    /// measured repair: I_PCM for macroblocks off by this much or more that
    /// no correction fixes (default 8); below it they are left as they are
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..))]
    pub pcm_threshold: Option<u8>,
    /// measured repair: an IDR period that has added more than this times the
    /// bytes dropped from it only repairs from --pcm-threshold on (default 1;
    /// 0: no budget)
    #[arg(long, default_value_t = crate::repair::DRIFT_BUDGET)]
    pub drift_budget: f64,
    /// measured repair: patch at most this fraction of a picture (default 1
    /// with corrections, 0.02 without)
    #[arg(long, hide = true)]
    pub drift_cap: Option<f64>,

    /// use explicit reference control (and the measured repair) even when
    /// the stream does not need it
    #[arg(long, hide = true)]
    pub force_explicit: bool,
}

/// No duplicate-frame cadence was found (exit code 2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoCadence(pub String);

impl std::fmt::Display for NoCadence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NoCadence {}

// ---------------------------------------------------------------- formatting

/// `f'{float(fr):.3f}'.rstrip('0').rstrip('.')`
fn fps_str(fr: Ratio) -> String {
    let s = format!("{:.3}", fr.to_f64());
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Python's `str(float)`.
fn pyf(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    format!("{x:?}")
}

/// Python's `str(None | float)`.
fn pyopt(x: Option<f64>) -> String {
    x.map(pyf).unwrap_or_else(|| "None".into())
}

// ---------------------------------------------------------------- the report

/// A JSON object with keys in the given order (Python's dict insertion order).
struct Pairs(Vec<(String, Value)>);

impl Serialize for Pairs {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

/// A `serde_json::Value` object whose listed keys come first, in that order
/// (serde_json's maps are sorted; the Python dicts are in insertion order).
struct Ordered {
    v: Value,
    order: &'static [&'static str],
}

impl Serialize for Ordered {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let Value::Object(m) = &self.v else { return self.v.serialize(s) };
        let mut out = s.serialize_map(Some(m.len()))?;
        for k in self.order {
            if let Some(v) = m.get(*k) {
                out.serialize_entry(k, v)?;
            }
        }
        for (k, v) in m {
            if !self.order.contains(&k.as_str()) {
                out.serialize_entry(k, v)?;
            }
        }
        out.end()
    }
}

/// Key order of `repair.repair`'s summary.
const REPAIR_ORDER: &[&str] = &[
    "candidate_pictures", "periods", "pictures_rewritten", "pcm_mbs", "bytes_added", "skip_to_explicit",
    "direct_to_explicit", "qp_carriers", "workers", "worker_peak_rss_mb", "drift_threshold", "drift_max_per_picture",
    "seconds", "sniff", "pictures", "pictures_over_threshold_after", "pictures_capped", "worst_after",
];

/// Key order of `rebuild.rebuild`'s summary.
const REBUILD_ORDER: &[&str] = &[
    "pairs", "leading_b_pictures", "pcm_threshold", "pcm_max_per_picture", "pictures_pcm_capped", "pcm_mbs",
    "direct_explicit", "bytes_added", "worst_in_leading_b", "workers", "worker_peak_rss_mb", "seconds", "pairs_detail",
];

#[derive(serde::Serialize)]
struct Resources {
    workers: usize,
    cpu_count: usize,
    physical_ram_gb: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    main_peak_rss_mb: Option<u64>,
}

#[derive(serde::Serialize)]
struct InputRate {
    fps: String,
    exact: String,
    source: &'static str,
    num_units_in_tick: u32,
    time_scale: u32,
}

#[derive(serde::Serialize)]
struct SpsTiming {
    num_units_in_tick: u32,
    time_scale: u32,
    sps_patched: usize,
}

#[derive(serde::Serialize)]
struct OutputStream {
    fps: String,
    exact: String,
    frames: usize,
    mkvmerge_messages: Vec<String>,
}

/// Explicit mode's `anchor_rebuilds` section.
#[derive(serde::Serialize)]
struct ExplicitRebuilds {
    pairs: usize,
    leading_pictures: usize,
    pairs_detail: Vec<plan::Rebuild>,
}

#[derive(serde::Serialize)]
#[serde(untagged)]
enum AnchorRebuilds {
    Old(Ordered),
    Explicit(ExplicitRebuilds),
}

#[derive(serde::Serialize)]
struct LeadingB {
    pictures: usize,
    differing: usize,
    worst: u32,
    mbs_ge3: usize,
    worst_luma: u32,
    mbs_ge3_luma: usize,
}

#[derive(serde::Serialize)]
struct VerifySection {
    #[serde(flatten)]
    base: VerifyResult,
    rebuild_leading_b: LeadingB,
    differing_outside_hard_neighbourhoods: usize,
    worst_outside: u32,
    ok: bool,
}

/// The JSON report; fields are in the Python dict's insertion order.
#[derive(serde::Serialize)]
struct Report {
    tool: &'static str,
    version: &'static str,
    input: String,
    output: Option<String>,
    resources: Resources,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_rate: Option<InputRate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frames_in: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    explicit_reference_control: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    detection: Option<DetectionReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cadence: Option<CadenceReport>,
    /// drops, unreferenced_reference_drops, substitute_drops, hard_pairs,
    /// anchor_rebuild_fallbacks, unsupported_pairs, sync
    #[serde(flatten)]
    plan: Option<PlanReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sps_timing: Option<SpsTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repair: Option<Ordered>,
    #[serde(skip_serializing_if = "Option::is_none")]
    anchor_rebuilds: Option<AnchorRebuilds>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_stream: Option<OutputStream>,
    #[serde(skip_serializing_if = "Option::is_none")]
    durations: Option<Pairs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verify: Option<VerifySection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_s: Option<Pairs>,
}

/// `json.dump(report, f, indent=1)`: one-space indent, no trailing newline.
fn write_report(path: &Path, rep: &Report) -> Result<()> {
    let mut buf = Vec::new();
    {
        let fmt = serde_json::ser::PrettyFormatter::with_indent(b" ");
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, fmt);
        rep.serialize(&mut ser)?;
    }
    std::fs::write(path, buf).with_context(|| format!("cannot write {}", path.display()))
}

/// The proxy features with the size of the stream they were decoded from.
#[derive(serde::Serialize)]
struct FeatCacheOut<'a> {
    es_size: u64,
    mad: &'a [f64],
    bmax: &'a [f64],
    mean: &'a [f64],
    max: &'a [f64],
}

#[derive(Deserialize)]
struct FeatCacheIn {
    es_size: u64,
    #[serde(flatten)]
    feat: Features,
}

// ---------------------------------------------------------------- the run

/// Everything `analyse` finds out and the later stages need.
struct Loaded {
    es: PathBuf,
    spss: Spss,
    ppss: Ppss,
    pics: Vec<Picture>,
    explicit: bool,
    fps_in: Ratio,
    feat: Features,
    fps_out: Ratio,
    r: Ratio,
    black: Vec<(usize, usize)>,
    cuts: Vec<usize>,
    dup: Vec<bool>,
    cadence: CadenceReport,
}

/// The plan: `drop` holds display indices.
struct Planned {
    drop: BTreeSet<usize>,
    hard: Vec<Hard>,
    rebuilds: Vec<plan::Rebuild>,
    freed: Vec<Freed>,
}

/// The stream the video is muxed from, and the original decode index of each
/// of its pictures.
struct Written {
    out_es: PathBuf,
    orig_of_out: Vec<usize>,
}

struct Run {
    a: Args,
    work: PathBuf,
    workers: usize,
    log: Logger,
    t0: Instant,
    times: Vec<(String, Value)>,
    rep: Report,
}

fn abspath(p: &Path) -> String {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()).display().to_string()
}

impl Run {
    fn new(args: Args) -> Result<Run> {
        let work = args.work.clone().ok_or_else(|| anyhow!("no work directory"))?;
        std::fs::create_dir_all(&work)?;
        let log = Logger::new(Some(&work.join("run.log")))?;
        let workers = args.workers.unwrap_or_else(system::default_workers).max(1);
        let (total, _) = system::physical_ram();
        let rep = Report {
            tool: "frame-dedup",
            version: crate::VERSION,
            input: abspath(&args.input),
            output: args.output.as_deref().map(abspath),
            resources: Resources {
                workers,
                cpu_count: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
                physical_ram_gb: py_round(total as f64 / (1u64 << 30) as f64, 1),
                main_peak_rss_mb: None,
            },
            input_rate: None,
            frames_in: None,
            explicit_reference_control: None,
            detection: None,
            cadence: None,
            plan: None,
            sps_timing: None,
            repair: None,
            anchor_rebuilds: None,
            output_stream: None,
            durations: None,
            verify: None,
            status: None,
            message: None,
            runtime_s: None,
        };
        Ok(Run { a: args, work, workers, log, t0: Instant::now(), times: Vec::new(), rep })
    }

    fn log(&self, msg: &str) {
        self.log.log(msg);
    }

    fn stage(&mut self, name: &str, t: Instant) {
        self.times.push((name.to_string(), Value::from(py_round(t.elapsed().as_secs_f64(), 1))));
    }

    // ------------------------------------------------------------------
    fn analyse(&mut self) -> Result<Loaded> {
        let workers = self.workers;
        let es = self.work.join("in.h264");
        let t = Instant::now();
        self.log(&format!("extracting video stream from {}", self.a.input.display()));
        stream::extract_annexb(&self.a.input, &es)?;
        let (spss, ppss, pics) = stream::load(&es)?;
        let n = pics.len();
        let (num, den) = stream::input_rate(&spss)?;
        let fps_in = Ratio::new(num as i64, den as i64);
        let in_name = cadence::rate_name(fps_in, &cadence::input_rates());
        let timing = spss[&0].vui.as_ref().and_then(|v| v.timing.as_ref()).ok_or_else(|| anyhow!("KeyError: 'timing'"))?;
        self.rep.input_rate = Some(InputRate {
            fps: fps_str(fps_in),
            exact: fps_in.to_string(),
            source: "SPS VUI timing_info",
            num_units_in_tick: timing.num_units_in_tick,
            time_scale: timing.time_scale,
        });
        self.rep.frames_in = Some(n);
        let explicit = explicit::stream_is_explicit(&pics) || self.a.force_explicit;
        self.rep.explicit_reference_control = Some(explicit);
        self.log(&format!(
            "{n} pictures, input rate {} fps ({fps_in}) from the SPS VUI, {} IDR periods",
            fps_str(fps_in),
            pics.iter().filter(|p| p.idr).count()
        ));
        if explicit {
            self.log(
                "the stream uses MMCO / reference list modification: headers will be rewritten with explicit reference control",
            );
        }
        let Some(in_name) = in_name else {
            return Err(stream::unsupported(format!(
                "input rate {} fps is not 29.97 or 30; nothing to do",
                fps_str(fps_in)
            )));
        };
        self.stage("extract_parse", t);

        let t = Instant::now();
        let feat_path = self.work.join("features.json");
        let es_size = std::fs::metadata(&es)?.len();
        let mut feat: Option<Features> = None;
        if let Ok(text) = std::fs::read_to_string(&feat_path) {
            if let Ok(c) = serde_json::from_str::<FeatCacheIn>(&text) {
                if c.feat.mad.len() == n && c.es_size == es_size {
                    feat = Some(c.feat);
                }
            }
        }
        let feat = match feat {
            Some(f) => f,
            None => {
                self.log("decoding the low-res luma proxy");
                let f = evidence::proxy_features(es.to_str().ok_or_else(|| anyhow!("non-UTF-8 path"))?, workers.min(8))?;
                if f.mad.len() != n {
                    bail!("proxy decode produced {} frames for {n} pictures", f.mad.len());
                }
                let c = FeatCacheOut { es_size, mad: &f.mad, bmax: &f.bmax, mean: &f.mean, max: &f.max };
                std::fs::write(&feat_path, serde_json::to_vec(&c)?)?;
                f
            }
        };
        self.stage("proxy_decode", t);

        let t = Instant::now();
        let (model, trained_on) = evidence::load_model()?;
        let an = cadence::analyse(&feat, fps_in, &model, trained_on.as_deref())?;
        for c in &an.detection.candidates {
            self.log(&format!(
                "  candidate {:>6} fps (r={}): total LLR {:10.1}, score {:.2}, {} phase changes",
                c.source, c.ratio, c.total_llr, c.cadence_score, c.phase_changes
            ));
        }
        self.rep.detection = Some(an.detection.clone());
        self.stage("detect", t);
        let best = an.best();
        if an.score < FAIL_SCORE || an.n_evidence < MIN_EVIDENCE_DUPS || best.total <= 0.0 {
            return Err(anyhow::Error::new(NoCadence(format!(
                "no duplicate-frame cadence found: the best model ({} fps in {in_name} fps) scores {:.2} (mean evidence of \
                 its predicted duplicates above a random phase; needs >= {}, real cadences score about 10). This looks like \
                 native {in_name} fps content; no output written.",
                best.source,
                an.score,
                pyf(FAIL_SCORE)
            ))));
        }
        let fps_out = best.rate;
        let r = best.r;
        self.log(&format!("detected source rate {} fps (r = {r}), cadence score {:.2}", best.source, an.score));

        let cad = cadence::cadence_report(&an, fps_in);
        for c in &cad.phase_changes {
            self.log(&format!(
                "  phase change at frame {} ({} s, {}): {}/{} -> {}/{}",
                c.display,
                pyf(c.time_s),
                c.at,
                c.from_phase,
                c.q,
                c.to_phase,
                c.q
            ));
        }
        self.log(&format!(
            "{} chunks, {} phase changes, {} duplicates",
            cad.chunks.len(),
            cad.phase_changes.len(),
            an.dup.iter().filter(|&&d| d).count()
        ));
        self.rep.cadence = Some(cad.clone());
        Ok(Loaded { es, spss, ppss, pics, explicit, fps_in, feat, fps_out, r, black: an.black.clone(), cuts: an.cuts.clone(), dup: an.dup.clone(), cadence: cad })
    }

    // ------------------------------------------------------------------
    fn plan(&mut self, ld: &Loaded) -> Result<Planned> {
        let t = Instant::now();
        let pics = &ld.pics;
        let n = pics.len();
        let mut slots: Vec<Option<&Picture>> = vec![None; n];
        for p in pics {
            slots[p.disp] = Some(p);
        }
        let disp: Vec<&Picture> = slots.into_iter().map(|p| p.expect("every display index is filled")).collect();
        // an error inside a callback is kept and raised after the planning
        let err: RefCell<Option<anyhow::Error>> = RefCell::new(None);
        let periods = stream::period_ranges(pics);
        let rebuildable = |k: usize, d: usize| -> (Option<Vec<usize>>, Option<String>) {
            let r = if ld.explicit {
                explicit::candidate(&ld.spss, pics, &periods, k, d)
            } else {
                rebuild::candidate(&ld.spss, pics, &periods, k, d)
            };
            match r {
                Ok(Ok(rb)) => (Some(rb.b), None),
                Ok(Err(why)) => (None, Some(why)),
                Err(e) => {
                    err.borrow_mut().get_or_insert(e);
                    (None, Some("error".into()))
                }
            }
        };
        let free_old = stream::free_drop_checker(&ld.spss, pics);
        let free_explicit = explicit::free_checker(pics);
        let free = |d: usize| -> bool {
            if ld.explicit {
                free_explicit(d)
            } else {
                match free_old(d) {
                    Ok(b) => b,
                    Err(e) => {
                        err.borrow_mut().get_or_insert(e);
                        false
                    }
                }
            }
        };
        let inp = PlanInputs {
            disp: &disp,
            dup: &ld.dup,
            free: Some(&free),
            mad: &ld.feat.mad,
            bmax: &ld.feat.bmax,
            visible_subs: !self.a.keep_idr_pairs,
            rebuildable: if self.a.no_rebuild { None } else { Some(&rebuildable) },
            explicit: ld.explicit,
            black: &ld.black,
            cuts: &ld.cuts,
            r: ld.r,
            fps_out: ld.fps_out,
            cadence: &ld.cadence,
        };
        let out = plan::run_plan(&inp);
        if let Some(e) = err.into_inner() {
            return Err(e);
        }
        let rep = out.report;
        for f in &rep.anchor_rebuild_fallbacks {
            self.log(&format!(
                "  anchor rebuild not possible at display {} ({}): {}",
                f.pair,
                f.types,
                f.fallback.as_deref().unwrap_or("None")
            ));
        }
        for u in &rep.unsupported_pairs {
            self.log(&format!("  unsupported pair at display {} ({}): {}; both kept", u.display, u.types, u.reason));
        }
        for c in &rep.sync.corrections {
            let at = match c.at {
                "black" => format!("black run [{}, {}]", c.frames[0], c.frames[1]),
                "cut" => format!("cut at {}", c.frames[0]),
                _ => format!("static frame {}", c.frames[0]),
            };
            let what = c.display.map_or(String::new(), |d| format!(" (display {d})"));
            self.log(&format!("  sync correction at {at}: {}{what}", c.action));
        }
        let st = &rep.sync.stats;
        self.log(&format!(
            "dropping {} pictures ({} unreferenced reference twins, {} anchor rebuilds ({} fell back), {} substitutes \
             ({} visible), {} hard pairs, {} unsupported pairs, {} sync corrections); output {} frames, final A/V offset \
             {} frames, max {} ms",
            out.drop.len(),
            rep.unreferenced_reference_drops.len(),
            out.rebuilds.len(),
            rep.anchor_rebuild_fallbacks.len(),
            rep.substitute_drops.len(),
            rep.substitute_drops.iter().filter(|x| x.visible).count(),
            out.hard.len(),
            rep.unsupported_pairs.len(),
            rep.sync.corrections.len(),
            n - out.drop.len(),
            pyf(st.final_offset_frames),
            pyf(st.max_abs_offset_ms)
        ));
        self.rep.plan = Some(rep);
        self.stage("plan", t);
        Ok(Planned { drop: out.drop, hard: out.hard, rebuilds: out.rebuilds, freed: out.freed })
    }

    // ------------------------------------------------------------------
    /// (num_units_in_tick, time_scale) of the output rate.
    fn timing(ld: &Loaded) -> (u32, u32) {
        (ld.fps_out.den as u32, (2 * ld.fps_out.num) as u32)
    }

    fn default_repair() -> Value {
        serde_json::json!({"pictures_rewritten": 0, "pcm_mbs": 0, "bytes_added": 0})
    }

    fn set_repair(&mut self, v: Value) {
        self.rep.repair = Some(Ordered { v, order: REPAIR_ORDER });
    }

    fn write(&mut self, ld: &Loaded, pl: &Planned) -> Result<Written> {
        if ld.explicit {
            return self.write_explicit(ld, pl);
        }
        let timing = Self::timing(ld);
        let t = Instant::now();
        let stage1 = self.work.join("dedup.h264");
        // rebuilt pairs drop K in a third stage, after the hard-drop repair
        let rb_k: HashSet<usize> = pl.rebuilds.iter().map(|r| r.k_dec).collect();
        let disp_dec = display_to_dec(&ld.pics);
        let drop_dec: HashSet<usize> = pl.drop.iter().map(|&i| disp_dec[i]).filter(|d| !rb_k.contains(d)).collect();
        let (n_renum, n_sps) = stream::write_stream(
            &ld.es,
            &ld.pics,
            &ld.spss,
            &stage1,
            &WriteOpts { drop: drop_dec.clone(), timing: Some(timing), ..Default::default() },
        )?;
        self.rep.sps_timing = Some(SpsTiming { num_units_in_tick: timing.0, time_scale: timing.1, sps_patched: n_sps });
        self.log(&format!(
            "wrote {}: {n_renum} slices renumbered, {n_sps} SPS timing patched to {}/{}",
            stage1.display(),
            timing.0,
            timing.1
        ));
        self.stage("write_dedup", t);
        let mut out_es = stage1.clone();
        let mut rep = Self::default_repair();
        if !pl.hard.is_empty() {
            let t = Instant::now();
            let (nspss, _nppss, npics) = stream::load(&stage1)?;
            let (repl, r) = repair::repair(
                &RepairArgs {
                    orig_es: &ld.es,
                    new_es: &stage1,
                    spss: &ld.spss,
                    ppss: &ld.ppss,
                    pics: &ld.pics,
                    new_pics: &npics,
                    drop: &drop_dec,
                    workers: self.workers,
                    sigma: None,
                    work: &self.work,
                    orig_of_out: None,
                    drift: Default::default(),
                },
                &self.log,
            )?;
            rep = r;
            if !repl.is_empty() {
                out_es = self.work.join("out.h264");
                stream::write_stream(&stage1, &npics, &nspss, &out_es, &WriteOpts { repl: Some(&repl), ..Default::default() })?;
            }
            self.log(&format!(
                "repair: {} pictures rewritten, {} PCM MBs, {:+} bytes",
                rep["pictures_rewritten"], rep["pcm_mbs"], rep["bytes_added"].as_i64().unwrap_or(0)
            ));
            self.stage("repair", t);
        }
        self.set_repair(rep);
        // stage-1/2 decode order
        let src_orig: Vec<usize> = ld.pics.iter().map(|p| p.dec).filter(|d| !drop_dec.contains(d)).collect();
        let mut orig_of_out = src_orig.clone();
        if !pl.rebuilds.is_empty() {
            let t = Instant::now();
            let (nspss, nppss, src_pics) = stream::load(&out_es)?;
            let src_of: HashMap<usize, usize> = src_orig.iter().enumerate().map(|(i, &d)| (d, i)).collect();
            let get = |d: usize| src_of.get(&d).copied().ok_or_else(|| anyhow!("KeyError: {d}"));
            let mut rbs = Vec::new();
            for r in &pl.rebuilds {
                rbs.push(rebuild::Rebuild {
                    k: get(r.k_dec)?,
                    d: get(r.d_dec)?,
                    b: r.b_dec.iter().filter_map(|b| src_of.get(b).copied()).collect(),
                });
            }
            let rb_es = self.work.join("rebuilt.h264");
            let (o, rb) = rebuild::rebuild(
                &ld.es, &out_es, &rb_es, &nspss, &nppss, &ld.pics, &src_pics, &src_orig, &rbs, self.workers, &self.work, &self.log,
                if self.a.no_corrections { None } else { Some(rebuild::CORR_THRESHOLD) },
            )?;
            orig_of_out = o;
            self.log(&format!(
                "anchor rebuild: {} pairs, {} leading B pictures, {} PCM MBs, {} direct MBs made explicit, {:+} bytes, \
                 worst difference in the leading B pictures {}",
                rb["pairs"], rb["leading_b_pictures"], rb["pcm_mbs"], rb["direct_explicit"],
                rb["bytes_added"].as_i64().unwrap_or(0), rb["worst_in_leading_b"]
            ));
            self.rep.anchor_rebuilds = Some(AnchorRebuilds::Old(Ordered { v: rb, order: REBUILD_ORDER }));
            out_es = rb_es;
            self.stage("rebuild", t);
        }
        Ok(Written { out_es, orig_of_out })
    }

    /// The measured repair's settings from the command line.
    fn drift(&self) -> crate::repair::Drift {
        use crate::repair::*;
        let corrections = !self.a.no_corrections;
        let threshold = self.a.drift_threshold.unwrap_or(if corrections { DRIFT_THRESHOLD_CORR } else { DRIFT_THRESHOLD });
        crate::repair::Drift {
            threshold,
            max_fraction: self.a.drift_cap.unwrap_or(if corrections { 1.0 } else { DRIFT_MAX_FRACTION }),
            corrections,
            pcm_threshold: self.a.pcm_threshold.unwrap_or(if corrections { PCM_THRESHOLD_CORR } else { threshold }),
            budget: if corrections { self.a.drift_budget } else { 0.0 },
        }
    }

    /// Streams with MMCO / list modification: one header plan for every drop
    /// and anchor rebuild (explicit.py), then the measured repair.
    fn write_explicit(&mut self, ld: &Loaded, pl: &Planned) -> Result<Written> {
        let timing = Self::timing(ld);
        let t = Instant::now();
        let pics = &ld.pics;
        let disp_dec = display_to_dec(pics);
        let drop_dec: HashSet<usize> = pl.drop.iter().map(|&i| disp_dec[i]).collect();
        let rbs: Vec<(usize, usize)> = pl.rebuilds.iter().map(|r| (r.k_dec, r.d_dec)).collect();
        let rb_k: HashSet<usize> = rbs.iter().map(|&(k, _)| k).collect();
        let mut sigma: Sigma = HashMap::new();
        for f in &pl.freed {
            sigma.insert(f.dec.ok_or_else(|| anyhow!("KeyError: 'dec'"))?, f.sigma.ok_or_else(|| anyhow!("KeyError: 'sigma'"))?);
        }
        for h in &pl.hard {
            sigma.insert(h.dropped_dec, Some(h.kept_dec));
        }
        let plain: HashSet<usize> = drop_dec.difference(&rb_k).copied().collect();
        for &d in &plain {
            if pics[d].ref_idc != 0 && !sigma.contains_key(&d) {
                sigma.insert(d, None); // an unreferenced reference picture (e.g. a substitute)
            }
        }
        let (entries, hdrs) = explicit::plan_stream(&ld.spss, pics, &plain, &sigma, &rbs)?;
        let stage1 = self.work.join("dedup.h264");
        let (n_rw, n_sps) =
            stream::write_explicit(&ld.es, pics, &ld.spss, &ld.ppss, &stage1, &entries, &hdrs, Some(timing))?;
        let orig_of_out: Vec<usize> = entries.iter().map(|e| e.pic).collect();
        self.rep.sps_timing = Some(SpsTiming { num_units_in_tick: timing.0, time_scale: timing.1, sps_patched: n_sps });
        self.log(&format!(
            "wrote {}: {n_rw} slice headers rewritten (explicit reference control), {} IDRs demoted for anchor rebuilds, \
             {n_sps} SPS timing patched to {}/{}",
            stage1.display(),
            rbs.len(),
            timing.0,
            timing.1
        ));
        self.stage("write_dedup", t);
        let mut out_es = stage1.clone();
        let mut rep = Self::default_repair();
        let (nspss, _nppss, npics) = stream::load(&stage1)?;
        // the written stream must parse to the planned lists and POCs
        if npics.len() != entries.len() {
            bail!("AssertionError: the written stream has {} pictures, planned {}", npics.len(), entries.len());
        }
        for (q, (e, h)) in npics.iter().zip(entries.iter().zip(&hdrs)) {
            let lists: [Vec<usize>; 2] = [0, 1].map(|i| q.lists[i].iter().map(|&x| orig_of_out[x]).collect());
            if q.poc != e.poc || q.idr != e.idr || lists != h.lists {
                bail!("AssertionError: picture {} of the written stream differs from the plan", q.dec);
            }
        }
        if sigma.values().any(|v| v.is_some()) || !rbs.is_empty() {
            let t = Instant::now();
            for &(k, d) in &rbs {
                sigma.insert(k, Some(d));
            }
            let (repl, r) = repair::repair(
                &RepairArgs {
                    orig_es: &ld.es,
                    new_es: &stage1,
                    spss: &ld.spss,
                    ppss: &ld.ppss,
                    pics,
                    new_pics: &npics,
                    drop: &drop_dec,
                    workers: self.workers,
                    sigma: Some(&sigma),
                    work: &self.work,
                    orig_of_out: Some(&orig_of_out),
                    drift: self.drift(),
                },
                &self.log,
            )?;
            rep = r;
            if !repl.is_empty() {
                out_es = self.work.join("out.h264");
                stream::write_stream(&stage1, &npics, &nspss, &out_es, &WriteOpts { repl: Some(&repl), ..Default::default() })?;
            }
            self.log(&format!(
                "repair: {} pictures rewritten, {} PCM MBs, {:+} bytes; {} pictures still over the drift threshold, worst {}",
                rep["pictures_rewritten"],
                rep["pcm_mbs"],
                rep["bytes_added"].as_i64().unwrap_or(0),
                rep.get("pictures_over_threshold_after").cloned().unwrap_or(Value::from(0)),
                rep.get("worst_after").cloned().unwrap_or(Value::from(0))
            ));
            self.stage("repair", t);
        }
        self.set_repair(rep);
        self.rep.anchor_rebuilds = Some(AnchorRebuilds::Explicit(ExplicitRebuilds {
            pairs: rbs.len(),
            leading_pictures: pl.rebuilds.iter().map(|r| r.b_dec.len()).sum(),
            pairs_detail: pl.rebuilds.clone(),
        }));
        Ok(Written { out_es, orig_of_out })
    }

    // ------------------------------------------------------------------
    fn remux(&mut self, ld: &Loaded, pl: &Planned, w: &Written) -> Result<()> {
        let t = Instant::now();
        let output = self.a.output.clone().ok_or_else(|| anyhow!("-o/--output is required"))?;
        let mut tmp = output.clone().into_os_string();
        tmp.push(".partial.mkv");
        let tmp = PathBuf::from(tmp);
        if tmp.exists() {
            std::fs::remove_file(&tmp)?;
        }
        self.log(&format!("muxing {}", output.display()));
        let fps_out = (ld.fps_out.num as u64, ld.fps_out.den as u64);
        let warnings = mux::mux(&self.a.input, &w.out_es, &tmp, fps_out, &|s: &str| self.log.log(s))?;
        std::fs::rename(&tmp, &output)?;
        let n_out = ld.pics.len() - pl.drop.len();
        self.rep.output_stream = Some(OutputStream {
            fps: fps_str(ld.fps_out),
            exact: ld.fps_out.to_string(),
            frames: n_out,
            mkvmerge_messages: warnings,
        });
        let dc: DurationCheck = mux::duration_check(&self.a.input, &output, n_out, fps_out)?;
        self.log(&format!(
            "durations: video {} s, audio {} s (input video {} s, audio {} s); A/V gap change {} s",
            pyopt(dc.output_video_s),
            pyopt(dc.output_audio_s),
            pyopt(dc.input_video_s),
            pyopt(dc.input_audio_s),
            pyopt(dc.av_gap_change_s)
        ));
        self.rep.durations = Some(Pairs(dc.entries().into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
        self.stage("mux", t);
        Ok(())
    }

    // ------------------------------------------------------------------
    fn verify(&mut self, ld: &Loaded, pl: &Planned, w: &Written) -> Result<()> {
        let t = Instant::now();
        self.log("verifying: decoding output and original");
        let (_, _, opics) = stream::load(&w.out_es)?;
        let pics = &ld.pics;
        // hard-drop neighbourhoods: frames after a dropped reference twin, up
        // to the end of its IDR period
        let mut hood: HashSet<usize> = HashSet::new();
        let mut period_end: HashMap<usize, usize> = HashMap::new();
        for p in pics {
            let e = period_end.entry(p.period).or_insert(0);
            *e = (*e).max(p.disp);
        }
        for h in &pl.hard {
            let d = &pics[h.dropped_dec];
            if ld.explicit {
                // every later picture in decode order of D's period can read it
                hood.extend(pics.iter().filter(|p| p.period == d.period && p.dec > d.dec).map(|p| p.disp));
            } else {
                hood.extend(d.disp + 1..=period_end[&d.period]);
            }
        }
        // leading B pictures of rebuilt pairs (they read I2 where they read K)
        let lead: HashSet<usize> =
            pl.rebuilds.iter().flat_map(|r| r.b_dec.iter().map(|&b| pics[b].disp)).filter(|d| !pl.drop.contains(d)).collect();
        let dec_by_disp = display_to_dec(pics);

        let mut writer = match &self.a.mb_report {
            Some(p) => {
                let sps = &ld.spss[&0];
                Some(MbReportWriter::create(
                    p,
                    &MbReportMeta {
                        input: abspath(&self.a.input),
                        output: self.a.output.as_deref().map(abspath),
                        fps_in: ld.fps_in,
                        fps_out: ld.fps_out,
                        mb_w: sps.mb_w,
                        mb_h: sps.mb_h,
                        crop: sps.crop.map(|c| 2 * c as usize),
                        threshold: self.a.mb_threshold,
                    },
                )?)
            }
            None => None,
        };
        let has_writer = writer.is_some();
        let mut sink = |f: &MbFrame| -> Result<()> {
            let p = &pics[dec_by_disp[f.frame.display]];
            let info = MbFrameInfo {
                typ: p.typ().as_str(),
                idr: p.idr,
                rebuilt_leading_b: lead.contains(&f.frame.display),
                hard_drop_neighbourhood: hood.contains(&f.frame.display),
            };
            writer.as_mut().expect("the sink exists only with a writer").frame(f, &info)
        };
        let mb = if has_writer { Some(MbOptions { threshold: self.a.mb_threshold, sink: &mut sink }) } else { None };
        let res = verify::verify(&ld.es, &w.out_es, &ld.spss, pics, &opics, self.workers, &self.log, Some(&w.orig_of_out), mb)?;
        if let Some(wr) = writer {
            let n = wr.finish()?;
            self.log(&format!(
                "mb report: {n} differing frames (macroblocks >= {}) written to {}",
                self.a.mb_threshold,
                self.a.mb_report.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
            ));
        }
        let in_lead: Vec<_> = res.differing_frames.iter().filter(|d| lead.contains(&d.display)).collect();
        let rebuild_leading_b = LeadingB {
            pictures: lead.len(),
            differing: in_lead.len(),
            worst: in_lead.iter().map(|d| d.worst).max().unwrap_or(0),
            mbs_ge3: in_lead.iter().map(|d| d.mbs_ge3).sum(),
            worst_luma: in_lead.iter().map(|d| d.worst_luma).max().unwrap_or(0),
            mbs_ge3_luma: in_lead.iter().map(|d| d.mbs_ge3_luma).sum(),
        };
        let outside: Vec<_> =
            res.differing_frames.iter().filter(|d| !hood.contains(&d.display) && !lead.contains(&d.display)).collect();
        let ok = outside.is_empty() && res.errors.is_empty() && res.frames_compared == res.frames_output;
        self.log(&format!(
            "verify: {} frames compared, {} bit-exact, {} differ (worst {}, {} MBs >= 3; luma worst {}, {} MBs >= 3), \
             {} differing outside hard-drop neighbourhoods and rebuilt leading B pictures (leading B: {} differ, worst {}, \
             {} MBs >= 3); errors {}",
            res.frames_compared,
            res.bit_exact,
            res.differing,
            res.worst_difference,
            res.mbs_ge3,
            res.worst_luma,
            res.mbs_ge3_luma,
            outside.len(),
            rebuild_leading_b.differing,
            rebuild_leading_b.worst,
            rebuild_leading_b.mbs_ge3,
            res.errors.len()
        ));
        let section = VerifySection {
            differing_outside_hard_neighbourhoods: outside.len(),
            worst_outside: outside.iter().map(|d| d.worst).max().unwrap_or(0),
            ok,
            rebuild_leading_b,
            base: res.clone(),
        };
        self.rep.verify = Some(section);
        self.stage("verify", t);
        Ok(())
    }

    // ------------------------------------------------------------------
    /// The stages after the logging of the banner; returns the exit code.
    fn execute(&mut self) -> Result<i32> {
        let ld = self.analyse()?;
        let pl = self.plan(&ld)?;
        if self.a.dry_run {
            self.rep.status = Some("dry_run");
            return Ok(0);
        }
        let w = self.write(&ld, &pl)?;
        self.remux(&ld, &pl, &w)?;
        if self.a.verify {
            self.verify(&ld, &pl, &w)?;
        }
        let ok = !self.a.verify || self.rep.verify.as_ref().is_some_and(|v| v.ok);
        self.rep.status = Some(if ok { "ok" } else { "verify_failed" });
        Ok(if ok { 0 } else { 4 })
    }
}

/// Decode index of every display index.
fn display_to_dec(pics: &[Picture]) -> Vec<usize> {
    let mut v = vec![0; pics.len()];
    for p in pics {
        v[p.disp] = p.dec;
    }
    v
}

/// Files this tool creates in the work directory (`run.log` is kept).
pub const WORK_FILES: [&str; 5] = ["in.h264", "dedup.h264", "out.h264", "rebuilt.h264", "features.json"];

/// Remove only what this tool created, then the directory if empty.
pub fn cleanup(work: &Path) {
    for f in WORK_FILES {
        let p = work.join(f);
        if p.exists() {
            let _ = std::fs::remove_file(p);
        }
    }
    let _ = std::fs::remove_dir(work);
}

/// Run the whole pipeline; returns the process exit code.
pub fn run(mut args: Args) -> i32 {
    if args.work.is_none() {
        let stem = args.input.file_stem().map(PathBuf::from).unwrap_or_default();
        args.work = Some(Path::new("work").join(stem));
    }
    args.workers = Some(args.workers.unwrap_or_else(system::default_workers).max(1));
    let mut r = match Run::new(args) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e:#}");
            return 1;
        }
    };
    let (total, _) = system::physical_ram();
    r.log(&format!(
        "frame-dedup {}: {} workers, {} logical CPUs, {:.1} GB RAM, below-normal priority",
        crate::VERSION,
        r.workers,
        r.rep.resources.cpu_count,
        total as f64 / (1u64 << 30) as f64
    ));
    let mut other: Option<anyhow::Error> = None;
    let mut code = match r.execute() {
        Ok(c) => c,
        Err(e) => {
            if let Some(nc) = e.downcast_ref::<NoCadence>() {
                r.rep.status = Some("no_cadence");
                r.rep.message = Some(nc.0.clone());
                r.log(&format!("FAIL: {}", nc.0));
                2
            } else if let Some(u) = e.downcast_ref::<Unsupported>() {
                r.rep.status = Some("unsupported");
                r.rep.message = Some(u.0.clone());
                r.log(&format!("FAIL: {}", u.0));
                3
            } else {
                other = Some(e);
                1
            }
        }
    };
    // what Python's `finally` does
    let total_s = py_round(r.t0.elapsed().as_secs_f64(), 1);
    let mut times = std::mem::take(&mut r.times);
    times.push(("total".into(), Value::from(total_s)));
    r.rep.runtime_s = Some(Pairs(times));
    let peak = (system::peak_rss() as f64 / (1u64 << 20) as f64).round_ties_even() as u64;
    r.rep.resources.main_peak_rss_mb = Some(peak);
    r.log(&format!(
        "done: status {}, main process peak RSS {peak} MB, total {} s",
        r.rep.status.unwrap_or("None"),
        pyf(total_s)
    ));
    if let Some(p) = &r.a.report {
        if let Err(e) = write_report(p, &r.rep) {
            eprintln!("error: {e:#}");
            if code == 0 {
                code = 1;
            }
        }
    }
    if !r.a.keep_work {
        cleanup(&r.work);
    }
    if let Some(e) = other {
        eprintln!("error: {e:#}");
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fps_strings() {
        assert_eq!(fps_str(Ratio::new(30000, 1001)), "29.97");
        assert_eq!(fps_str(Ratio::int(30)), "30");
        assert_eq!(fps_str(Ratio::new(24000, 1001)), "23.976");
        assert_eq!(fps_str(Ratio::int(25)), "25");
    }

    #[test]
    fn python_float_text() {
        assert_eq!(pyf(5.0), "5.0");
        assert_eq!(pyf(805.9), "805.9");
        assert_eq!(pyf(-0.35), "-0.35");
        assert_eq!(pyopt(None), "None");
        assert_eq!(pyopt(Some(1300.007)), "1300.007");
    }

    #[test]
    fn ordered_keys_first() {
        let o = Ordered { v: serde_json::json!({"b": 1, "a": 2, "z": 3}), order: &["z", "b", "missing"] };
        assert_eq!(serde_json::to_string(&o).unwrap(), r#"{"z":3,"b":1,"a":2}"#);
    }

    #[test]
    fn report_indent_is_one_space() {
        let p = Pairs(vec![("x".into(), serde_json::json!([1, 2])), ("y".into(), serde_json::json!({}))]);
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::PrettyFormatter::with_indent(b" "));
        p.serialize(&mut ser).unwrap();
        assert_eq!(String::from_utf8(buf).unwrap(), "{
 \"x\": [
  1,
  2
 ],
 \"y\": {}
}");
    }

    #[test]
    fn cleanup_removes_only_our_files() {
        let d = std::env::temp_dir().join(format!("fd_cleanup_test_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        for f in WORK_FILES.iter().chain(&["run.log", "other.txt"]) {
            std::fs::write(d.join(f), b"x").unwrap();
        }
        cleanup(&d);
        assert!(d.join("run.log").exists() && d.join("other.txt").exists());
        assert!(WORK_FILES.iter().all(|f| !d.join(f).exists()));
        std::fs::remove_dir_all(&d).unwrap();
    }
}

//! Remux with mkvmerge and the duration check. Port of `frame_dedup/mux.py`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Map, Value};

use crate::system;

/// The first `n` characters (Python `s[:n]`).
fn head(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The last `n` characters (Python `s[-n:]`).
fn tail(s: &str, n: usize) -> String {
    let c = s.chars().count();
    s.chars().skip(c.saturating_sub(n)).collect()
}

/// Python's `round(x, ndigits)`: the correctly rounded (half to even) decimal
/// of the exact binary value. Rust's fixed-precision formatting is exact too.
pub fn py_round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

/// `mkvmerge -J <mkv>` parsed.
pub fn identify(mkv: &Path) -> Result<Value> {
    let r = system::command("mkvmerge")
        .arg("-J")
        .arg(mkv)
        .output()
        .map_err(|e| anyhow!("could not run mkvmerge: {e}"))?;
    let out = String::from_utf8_lossy(&r.stdout);
    let err = String::from_utf8_lossy(&r.stderr);
    // Python: returncode not in (0, 1)
    if !matches!(r.status.code(), Some(0) | Some(1)) {
        bail!("mkvmerge -J failed: {} {}", head(&out, 300), head(&err, 300));
    }
    Ok(serde_json::from_str(&out)?)
}

/// Python truthiness of a JSON value used as an optional string.
fn truthy_str(v: Option<&Value>) -> Option<String> {
    match v {
        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// `int(bool)` of a JSON flag (`0`/`1`).
fn flag(v: &Value) -> i64 {
    match v {
        Value::Bool(b) => *b as i64,
        Value::Number(n) => (n.as_f64().unwrap_or(0.0) != 0.0) as i64,
        _ => 0,
    }
}

/// The mkvmerge command line (program name first) for muxing the video in
/// `es` plus everything except video from `in_mkv`; `info` is
/// `identify(in_mkv)`. `fps_out` is the exact output rate (numerator,
/// denominator), reduced like a Python `Fraction`.
pub fn mux_command(info: &Value, in_mkv: &Path, es: &Path, out_mkv: &Path, fps_out: (u64, u64)) -> Vec<String> {
    let empty = Vec::new();
    let tracks = info["tracks"].as_array().unwrap_or(&empty);
    let vp = tracks.iter().find(|t| t["type"] == "video").map(|t| &t["properties"]);
    let get = |k: &str| vp.and_then(|p| p.get(k));
    let rate = if fps_out.1 != 1 { format!("{}/{}p", fps_out.0, fps_out.1) } else { format!("{}p", fps_out.0) };
    let mut cmd: Vec<String> = vec!["mkvmerge".into(), "-q".into(), "-o".into(), out_mkv.to_string_lossy().into_owned()];
    if let Some(title) = truthy_str(info.get("container").and_then(|c| c.get("properties")).and_then(|p| p.get("title"))) {
        cmd.extend(["--title".into(), title]);
    }
    if let Some(lang) = truthy_str(get("language_ietf")).or_else(|| truthy_str(get("language"))) {
        cmd.extend(["--language".into(), format!("0:{lang}")]);
    }
    if let Some(n) = truthy_str(get("track_name")) {
        cmd.extend(["--track-name".into(), format!("0:{n}")]);
    }
    if let Some(v) = get("default_track") {
        cmd.extend(["--default-track-flag".into(), format!("0:{}", flag(v))]);
    }
    if let Some(v) = get("forced_track") {
        cmd.extend(["--forced-display-flag".into(), format!("0:{}", flag(v))]);
    }
    if let Some(d) = truthy_str(get("display_dimensions")) {
        cmd.extend(["--display-dimensions".into(), format!("0:{d}")]);
    }
    cmd.extend([
        "--default-duration".into(),
        format!("0:{rate}"),
        es.to_string_lossy().into_owned(),
        "--no-video".into(),
        in_mkv.to_string_lossy().into_owned(),
    ]);
    cmd
}

/// Python's `str.splitlines()` on the line breaks mkvmerge can produce.
fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\n' || b[i] == b'\r' {
            out.push(&s[start..i]);
            if b[i] == b'\r' && b.get(i + 1) == Some(&b'\n') {
                i += 1;
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// Video from `es` first, then everything except video from the input (audio,
/// subtitles, chapters, tags, attachments). Returns mkvmerge's messages.
pub fn mux(in_mkv: &Path, es: &Path, out_mkv: &Path, fps_out: (u64, u64), log: &dyn Fn(&str)) -> Result<Vec<String>> {
    let info = identify(in_mkv)?;
    let cmd = mux_command(&info, in_mkv, es, out_mkv, fps_out);
    let line: Vec<String> = cmd.iter().map(|c| if c.contains(' ') { format!("\"{c}\"") } else { c.clone() }).collect();
    log(&format!("  {}", line.join(" ")));
    let r = system::command(&cmd[0])
        .args(&cmd[1..])
        .output()
        .map_err(|e| anyhow!("could not run mkvmerge: {e}"))?;
    let out = String::from_utf8_lossy(&r.stdout).into_owned();
    let err = String::from_utf8_lossy(&r.stderr).into_owned();
    // Python: returncode >= 2 (a killed process has no code; treat as failure)
    if r.status.code().is_none_or(|c| c >= 2) {
        bail!("mkvmerge failed: {} {}", tail(&out, 500), tail(&err, 500));
    }
    let all = format!("{out}{err}");
    Ok(splitlines(&all).into_iter().filter(|l| !l.trim().is_empty()).map(String::from).collect())
}

/// `{codec_type}{index}` -> seconds, from the per-track `DURATION` statistics
/// tags (`H:MM:SS.nnnnnnnnn`). Sorted by key like Python's `sorted(d)`.
pub fn track_durations(mkv: &Path) -> Result<BTreeMap<String, f64>> {
    let r = system::command("ffprobe")
        .args(["-v", "error", "-show_entries", "stream=index,codec_type:stream_tags=DURATION", "-of", "json"])
        .arg(mkv)
        .output()
        .map_err(|e| anyhow!("could not run ffprobe: {e}"))?;
    let text = String::from_utf8_lossy(&r.stdout);
    let text = if text.is_empty() { "{}" } else { &text };
    let v: Value = serde_json::from_str(text).context("ffprobe output")?;
    let mut out = BTreeMap::new();
    for s in v.get("streams").and_then(Value::as_array).unwrap_or(&Vec::new()) {
        let Some(d) = s.get("tags").and_then(|t| t.get("DURATION")).and_then(Value::as_str) else { continue };
        if d.is_empty() {
            continue;
        }
        let parts: Vec<&str> = d.split(':').collect();
        let [h, m, sec] = parts[..] else { bail!("bad DURATION tag {d:?}") };
        let secs = h.parse::<i64>()? * 3600 + m.parse::<i64>()? * 60;
        let key = format!("{}{}", s["codec_type"].as_str().unwrap_or(""), s["index"]);
        out.insert(key, secs as f64 + sec.parse::<f64>()?);
    }
    Ok(out)
}

/// Result of [`duration_check`]; field order is the Python dict's insertion order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DurationCheck {
    pub input_video_s: Option<f64>,
    pub input_audio_s: Option<f64>,
    pub output_video_s: Option<f64>,
    pub output_audio_s: Option<f64>,
    pub output_video_frames_s: f64,
    pub frame_s: f64,
    pub output_av_diff_s: Option<f64>,
    pub output_av_diff_frames: Option<f64>,
    pub input_av_diff_s: Option<f64>,
    pub video_duration_change_s: Option<f64>,
    pub within_one_frame_of_input: Option<bool>,
    pub within_one_frame_of_audio: Option<bool>,
    pub av_gap_change_s: Option<f64>,
    pub av_gap_change_within_one_frame: Option<bool>,
}

fn num(x: f64) -> Value {
    serde_json::Number::from_f64(x).map(Value::Number).unwrap_or(Value::Null)
}

impl DurationCheck {
    /// The dict entries in Python's insertion order (absent keys omitted;
    /// the four leading durations are always present, as `null` if unknown).
    pub fn entries(&self) -> Vec<(&'static str, Value)> {
        let opt = |x: Option<f64>| x.map(num).unwrap_or(Value::Null);
        let mut e = vec![
            ("input_video_s", opt(self.input_video_s)),
            ("input_audio_s", opt(self.input_audio_s)),
            ("output_video_s", opt(self.output_video_s)),
            ("output_audio_s", opt(self.output_audio_s)),
            ("output_video_frames_s", num(self.output_video_frames_s)),
            ("frame_s", num(self.frame_s)),
        ];
        let mut add = |k: &'static str, v: Option<Value>| {
            if let Some(v) = v {
                e.push((k, v));
            }
        };
        add("output_av_diff_s", self.output_av_diff_s.map(num));
        add("output_av_diff_frames", self.output_av_diff_frames.map(num));
        add("input_av_diff_s", self.input_av_diff_s.map(num));
        add("video_duration_change_s", self.video_duration_change_s.map(num));
        add("within_one_frame_of_input", self.within_one_frame_of_input.map(Value::Bool));
        add("within_one_frame_of_audio", self.within_one_frame_of_audio.map(Value::Bool));
        add("av_gap_change_s", self.av_gap_change_s.map(num));
        add("av_gap_change_within_one_frame", self.av_gap_change_within_one_frame.map(Value::Bool));
        e
    }

    /// As a JSON object (keys sorted by serde_json's map; use `entries()` to
    /// keep Python's order when writing the report).
    pub fn to_value(&self) -> Value {
        Value::Object(self.entries().into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Map<_, _>>())
    }
}

/// Compare video duration with the first audio track's, before and after.
pub fn duration_check(in_mkv: &Path, out_mkv: &Path, n_out: usize, fps_out: (u64, u64)) -> Result<DurationCheck> {
    let din = track_durations(in_mkv)?;
    let dout = track_durations(out_mkv)?;
    Ok(compute_duration_check(&din, &dout, n_out, fps_out))
}

/// The pure part of [`duration_check`].
pub fn compute_duration_check(
    din: &BTreeMap<String, f64>,
    dout: &BTreeMap<String, f64>,
    n_out: usize,
    fps_out: (u64, u64),
) -> DurationCheck {
    // first key (sorted) starting with `kind`
    let pick = |d: &BTreeMap<String, f64>, kind: &str| d.iter().find(|(k, _)| k.starts_with(kind)).map(|(_, v)| *v);
    let frame = 1.0 / (fps_out.0 as f64 / fps_out.1 as f64);
    let mut r = DurationCheck {
        input_video_s: pick(din, "video"),
        input_audio_s: pick(din, "audio"),
        output_video_s: pick(dout, "video"),
        output_audio_s: pick(dout, "audio"),
        output_video_frames_s: py_round(n_out as f64 * frame, 4),
        frame_s: py_round(frame, 5),
        ..Default::default()
    };
    if let (Some(v), Some(a)) = (r.output_video_s, r.output_audio_s) {
        let d = py_round(v - a, 4);
        r.output_av_diff_s = Some(d);
        r.output_av_diff_frames = Some(py_round(d / frame, 2));
    }
    if let (Some(v), Some(a)) = (r.input_video_s, r.input_audio_s) {
        r.input_av_diff_s = Some(py_round(v - a, 4));
    }
    if let (Some(i), Some(o)) = (r.input_video_s, r.output_video_s) {
        let c = py_round(o - i, 4);
        r.video_duration_change_s = Some(c);
        r.within_one_frame_of_input = Some(c.abs() <= frame);
    }
    if let Some(d) = r.output_av_diff_s {
        r.within_one_frame_of_audio = Some(d.abs() <= frame);
    }
    if let (Some(o), Some(i)) = (r.output_av_diff_s, r.input_av_diff_s) {
        // the input may already end its video and audio a little apart; what
        // the tool must preserve is that gap
        let g = py_round(o - i, 4);
        r.av_gap_change_s = Some(g);
        r.av_gap_change_within_one_frame = Some(g.abs() <= frame);
    }
    r
}

/// `frame-dedup dev mux-cmd <in.mkv> <es> <out.mkv> <num> <den>`: the mkvmerge
/// command line as a JSON list. `dev duration-check <in> <out> <n_out> <num> <den>`.
pub fn dev_main(cmd: &str, args: &[String]) -> Result<()> {
    use std::path::PathBuf;
    let p = |i: usize| -> Result<PathBuf> { Ok(PathBuf::from(args.get(i).ok_or_else(|| anyhow!("missing argument {i}"))?)) };
    let n = |i: usize| -> Result<u64> { Ok(args.get(i).ok_or_else(|| anyhow!("missing argument {i}"))?.parse()?) };
    match cmd {
        "mux-cmd" => {
            let info = identify(&p(0)?)?;
            let c = mux_command(&info, &p(0)?, &p(1)?, &p(2)?, (n(3)?, n(4)?));
            println!("{}", serde_json::to_string(&c)?);
        }
        "mux" => {
            let msgs = mux(&p(0)?, &p(1)?, &p(2)?, (n(3)?, n(4)?), &|s| println!("{s}"))?;
            println!("{}", serde_json::to_string(&msgs)?);
        }
        "duration-check" => {
            let d = duration_check(&p(0)?, &p(1)?, n(2)? as usize, (n(3)?, n(4)?))?;
            println!("{}", serde_json::to_string(&d.to_value())?);
        }
        _ => bail!("unknown mux dev command {cmd:?}"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_matches_python() {
        // Python: round(2.675, 2) == 2.67 (binary value is below the tie), round(0.125, 2) == 0.12,
        // round(0.375, 2) == 0.38, round(-0.00001, 4) == -0.0, round(1300.00749999, 3) == 1300.007
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(0.125, 2), 0.12);
        assert_eq!(py_round(0.375, 2), 0.38);
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(py_round(3.5, 0), 4.0);
        assert!(py_round(-0.00001, 4).is_sign_negative());
        assert_eq!(py_round(1.0 / (24000.0 / 1001.0), 5), 0.04171);
    }

    #[test]
    fn splitlines_like_python() {
        assert_eq!(splitlines("a\nb\r\nc\rd\n"), vec!["a", "b", "c", "d"]);
        assert_eq!(splitlines("a\n\nb"), vec!["a", "", "b"]);
        assert!(splitlines("").is_empty());
    }

    #[test]
    fn check_matches_report() {
        // numbers from work/sample_report.json
        let mk = |v: f64, a: f64| BTreeMap::from([("audio1".to_string(), a), ("video0".to_string(), v)]);
        let r = compute_duration_check(&mk(1300.008, 1300.096), &mk(1300.007, 1300.096), 31169, (24000, 1001));
        assert_eq!(r.frame_s, 0.04171);
        assert_eq!(r.output_video_frames_s, 1300.007);
        assert_eq!(r.output_av_diff_s, Some(-0.089));
        assert_eq!(r.output_av_diff_frames, Some(-2.13));
        assert_eq!(r.av_gap_change_s, Some(-0.001));
        assert_eq!(r.within_one_frame_of_audio, Some(false));
    }

    #[test]
    fn command_line() {
        let info: Value = serde_json::from_str(
            r#"{"container":{"properties":{"title":"My Title"}},"tracks":[{"type":"video","properties":
            {"language":"eng","default_track":true,"forced_track":false,"display_dimensions":"1920x1080"}}]}"#,
        )
        .unwrap();
        let c = mux_command(&info, Path::new("in.mkv"), Path::new("a.h264"), Path::new("o.mkv"), (24000, 1001));
        let want = "mkvmerge -q -o o.mkv --title My Title --language 0:eng --default-track-flag 0:1 \
                    --forced-display-flag 0:0 --display-dimensions 0:1920x1080 --default-duration 0:24000/1001p \
                    a.h264 --no-video in.mkv";
        assert_eq!(c.join(" "), want);
        let c = mux_command(&info, Path::new("in.mkv"), Path::new("a.h264"), Path::new("o.mkv"), (25, 1));
        assert!(c.contains(&"0:25p".to_string()));
    }
}

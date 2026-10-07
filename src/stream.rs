//! Annex-B stream handling: extraction, loading, scope check, display order
//! and rewriting. Port of `frame_dedup/stream.py`.
//!
//! This file has the loading side (`load`, `check_scope`, `input_rate`,
//! `au_offsets`); the writers are still to be ported.

use std::fmt;
use std::fs::File;
use std::io::Write as _;
use std::path::Path;

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use serde_json::{Map, Value, json};

use crate::bits::{escape, unescape};
use crate::h264hdr::{Picture, Ppss, Pps, SliceHeader, Sps, Spss, Vui, Hrd, iter_nals, parse_stream, patch_sps_timing, sps_frame_rate};
use crate::mv::dpb_walk;
use crate::records::canonical_text;

/// The input is outside what this tool can process (Python: `Unsupported`).
/// Carried inside an `anyhow::Error`; the pipeline downcasts to it for exit code 3.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unsupported {}

/// An `anyhow::Error` holding [`Unsupported`].
pub fn unsupported(msg: impl Into<String>) -> anyhow::Error {
    anyhow::Error::new(Unsupported(msg.into()))
}

/// Byte range of every access unit; parameter sets and other non-VCL NALs in
/// front of a picture belong to it.
pub fn au_offsets(pics: &mut [Picture], size: usize) {
    let mut prev = 0;
    for p in pics.iter_mut() {
        p.au_start = prev;
        let last = p.slices.last().expect("a picture has at least one slice");
        prev = last.offset + last.size;
        p.au_end = prev;
    }
    if let Some(last) = pics.last_mut() {
        last.au_end = size;
    }
}

/// Parse an Annex-B stream. Returns (spss, ppss, pics) with decode index,
/// POC, ref lists, display index and AU byte ranges attached.
pub fn load(es_path: &Path) -> Result<(Spss, Ppss, Vec<Picture>)> {
    let file = File::open(es_path).with_context(|| format!("cannot open {}", es_path.display()))?;
    let size = file.metadata()?.len() as usize;
    // an empty file cannot be mapped
    let map = if size > 0 { Some(unsafe { Mmap::map(&file)? }) } else { None };
    let data: &[u8] = map.as_deref().unwrap_or(&[]);
    let (spss, ppss, mut pics) = parse_stream(data)?;
    if pics.is_empty() {
        return Err(unsupported("no pictures in the video stream"));
    }
    check_scope(&spss, &ppss, &pics)?;
    if let Err(e) = dpb_walk(&spss, &mut pics) {
        return Err(unsupported(e.to_string()));
    }
    let mut period: i64 = -1;
    for p in pics.iter_mut() {
        if p.idr {
            period += 1;
        }
        p.period = period as usize;
    }
    // sorted(range(n), key=(period, poc)) is stable
    let mut order: Vec<usize> = (0..pics.len()).collect();
    order.sort_by_key(|&d| (pics[d].period, pics[d].poc));
    for (i, &d) in order.iter().enumerate() {
        pics[d].disp = i;
    }
    au_offsets(&mut pics, size);
    Ok((spss, ppss, pics))
}

pub fn check_scope(spss: &Spss, ppss: &Ppss, pics: &[Picture]) -> Result<()> {
    if !pics[0].idr {
        return Err(unsupported("stream does not start with an IDR picture"));
    }
    if spss.len() != 1 {
        return Err(unsupported(format!("{} distinct SPS ids; only one is supported", spss.len())));
    }
    let Some(s) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    if (s.profile_idc != 77 && s.profile_idc != 100) || s.chroma_format_idc != 1 || !s.frame_mbs_only {
        return Err(unsupported(format!(
            "needs Main/High 4:2:0 progressive H.264 (profile {}, chroma {}, frame_mbs_only {})",
            s.profile_idc, s.chroma_format_idc, s.frame_mbs_only as u8
        )));
    }
    if s.poc_type != 0 {
        return Err(unsupported("only POC type 0 is supported"));
    }
    for p in ppss.values() {
        if !p.cabac {
            return Err(unsupported("only CABAC streams are supported"));
        }
    }
    let multi = pics.iter().filter(|p| p.slices.len() > 1).count();
    if multi > 0 {
        return Err(unsupported(format!("{multi} pictures have more than one slice; only one slice per picture is supported")));
    }
    for p in pics {
        if p.slices[0].long_term_ref == Some(true) {
            return Err(unsupported("long-term references are not supported"));
        }
    }
    Ok(())
}

/// Input frame rate (num, den) from the SPS VUI timing.
pub fn input_rate(spss: &Spss) -> Result<(u64, u64)> {
    let Some(s) = spss.get(&0) else {
        bail!("KeyError: 0 (no SPS with id 0)");
    };
    match sps_frame_rate(s) {
        Some(fr) => Ok(fr),
        None => Err(unsupported("the SPS carries no VUI timing_info; cannot read the input frame rate")),
    }
}

// ---------------------------------------------------------------------------
// dev commands: `headers`, `escape`, `patch-sps` (oracle: port/oracle.py)
// ---------------------------------------------------------------------------

fn int(b: bool) -> Value {
    json!(b as u8)
}

fn hrd_json(h: &Hrd) -> Value {
    json!({
        "cpb_cnt": h.cpb_cnt, "bit_rate_scale": h.bit_rate_scale, "cpb_size_scale": h.cpb_size_scale,
        "sched": h.sched.iter().map(|&(a, b, c)| json!([a, b, c as u8])).collect::<Vec<_>>(),
        "init_cpb_delay_len": h.init_cpb_delay_len, "cpb_removal_delay_len": h.cpb_removal_delay_len,
        "dpb_output_delay_len": h.dpb_output_delay_len, "time_offset_len": h.time_offset_len,
    })
}

fn vui_json(v: &Vui) -> Value {
    let mut m = Map::new();
    if let Some(x) = v.aspect_ratio_idc {
        m.insert("aspect_ratio_idc".into(), json!(x));
    }
    if let Some((a, b)) = v.sar {
        m.insert("sar".into(), json!([a, b]));
    }
    if let Some(x) = v.overscan_appropriate {
        m.insert("overscan_appropriate".into(), int(x));
    }
    if let Some(x) = v.video_format {
        m.insert("video_format".into(), json!(x));
    }
    if let Some(x) = v.full_range {
        m.insert("full_range".into(), int(x));
    }
    if let Some((a, b, c)) = v.colour {
        m.insert("colour".into(), json!([a, b, c]));
    }
    if let Some((a, b)) = v.chroma_loc {
        m.insert("chroma_loc".into(), json!([a, b]));
    }
    m.insert(
        "timing".into(),
        match &v.timing {
            None => Value::Null,
            Some(t) => json!({
                "num_units_in_tick": t.num_units_in_tick, "time_scale": t.time_scale,
                "fixed_frame_rate": int(t.fixed_frame_rate), "bit": t.bit,
            }),
        },
    );
    m.insert("nal_hrd".into(), v.nal_hrd.as_ref().map_or(Value::Null, hrd_json));
    m.insert("vcl_hrd".into(), v.vcl_hrd.as_ref().map_or(Value::Null, hrd_json));
    if let Some(x) = v.low_delay_hrd {
        m.insert("low_delay_hrd".into(), int(x));
    }
    m.insert("pic_struct_present".into(), int(v.pic_struct_present));
    m.insert(
        "bitstream_restriction".into(),
        match &v.bitstream_restriction {
            None => Value::Null,
            Some(r) => json!({
                "mv_over_pic_boundaries": int(r.mv_over_pic_boundaries),
                "max_bytes_per_pic_denom": r.max_bytes_per_pic_denom, "max_bits_per_mb_denom": r.max_bits_per_mb_denom,
                "log2_max_mv_length_h": r.log2_max_mv_length_h, "log2_max_mv_length_v": r.log2_max_mv_length_v,
                "max_num_reorder_frames": r.max_num_reorder_frames, "max_dec_frame_buffering": r.max_dec_frame_buffering,
            }),
        },
    );
    m.insert("end_bit".into(), json!(v.end_bit));
    Value::Object(m)
}

/// The SPS as the Python dict (keys absent in Python are absent here).
pub fn sps_json(s: &Sps) -> Value {
    let mut m = Map::new();
    m.insert("profile_idc".into(), json!(s.profile_idc));
    m.insert("level_idc".into(), json!(s.level_idc));
    m.insert("id".into(), json!(s.id));
    m.insert("chroma_format_idc".into(), json!(s.chroma_format_idc));
    if s.chroma_format_idc == 3 {
        m.insert("separate_colour_plane".into(), int(s.separate_colour_plane));
    }
    m.insert("log2_max_frame_num".into(), json!(s.log2_max_frame_num));
    m.insert("poc_type".into(), json!(s.poc_type));
    if s.poc_type == 0 {
        m.insert("log2_max_poc_lsb".into(), json!(s.log2_max_poc_lsb));
    } else if s.poc_type == 1 {
        m.insert("delta_pic_order_always_zero".into(), int(s.delta_pic_order_always_zero));
    }
    m.insert("max_num_ref_frames".into(), json!(s.max_num_ref_frames));
    m.insert("gaps_allowed".into(), int(s.gaps_allowed));
    m.insert("mb_w".into(), json!(s.mb_w));
    m.insert("mb_h".into(), json!(s.mb_h));
    m.insert("frame_mbs_only".into(), int(s.frame_mbs_only));
    if !s.frame_mbs_only {
        m.insert("mbaff".into(), int(s.mbaff));
    }
    m.insert("direct_8x8_inference".into(), int(s.direct_8x8_inference));
    m.insert("crop".into(), json!(s.crop));
    m.insert("vui".into(), s.vui.as_ref().map_or(Value::Null, vui_json));
    Value::Object(m)
}

pub fn pps_json(p: &Pps) -> Value {
    let mut m = Map::new();
    m.insert("id".into(), json!(p.id));
    m.insert("sps_id".into(), json!(p.sps_id));
    m.insert("cabac".into(), int(p.cabac));
    m.insert("bottom_field_poc_present".into(), int(p.bottom_field_poc_present));
    m.insert("num_slice_groups".into(), json!(p.num_slice_groups));
    m.insert("num_ref_idx_l0_default".into(), json!(p.num_ref_idx_l0_default));
    m.insert("num_ref_idx_l1_default".into(), json!(p.num_ref_idx_l1_default));
    m.insert("weighted_pred".into(), int(p.weighted_pred));
    m.insert("weighted_bipred_idc".into(), json!(p.weighted_bipred_idc));
    m.insert("pic_init_qp".into(), json!(p.pic_init_qp));
    m.insert("chroma_qp_offset".into(), json!(p.chroma_qp_offset));
    m.insert("deblocking_ctrl_present".into(), int(p.deblocking_ctrl_present));
    m.insert("constrained_intra".into(), int(p.constrained_intra));
    m.insert("redundant_pic_cnt_present".into(), int(p.redundant_pic_cnt_present));
    m.insert("transform_8x8".into(), int(p.transform_8x8));
    m.insert("chroma_qp_offset2".into(), json!(p.chroma_qp_offset2));
    // Python has 'scaling_matrix' only when more_rbsp_data() was true (always 0 then);
    // the Rust struct cannot tell absent from 0, so the oracle also emits 0 when absent.
    m.insert("scaling_matrix".into(), int(p.scaling_matrix));
    Value::Object(m)
}

/// The slice header as the Python dict. `poc_type` is the SPS's (the POC
/// fields exist in the dict only for type 0).
pub fn slice_json(h: &SliceHeader, poc_type: u32) -> Value {
    let mut m = Map::new();
    m.insert("first_mb".into(), json!(h.first_mb));
    m.insert("slice_type".into(), json!(h.slice_type.as_str()));
    m.insert("pps_id".into(), json!(h.pps_id));
    m.insert("frame_num_bit".into(), json!(h.frame_num_bit));
    m.insert("frame_num".into(), json!(h.frame_num));
    m.insert("field_pic".into(), int(h.field_pic));
    if let Some(x) = h.bottom_field {
        m.insert("bottom_field".into(), int(x));
    }
    if let Some(x) = h.idr_pic_id_bit {
        m.insert("idr_pic_id_bit".into(), json!(x));
    }
    if let Some(x) = h.idr_pic_id {
        m.insert("idr_pic_id".into(), json!(x));
    }
    if poc_type == 0 {
        m.insert("poc_lsb_bit".into(), json!(h.poc_lsb_bit));
        m.insert("poc_lsb".into(), json!(h.poc_lsb));
    }
    if let Some(x) = h.delta_poc_bottom {
        m.insert("delta_poc_bottom".into(), json!(x));
    }
    if let Some(x) = h.direct_spatial {
        m.insert("direct_spatial".into(), int(x));
    }
    m.insert("num_ref_l0".into(), json!(h.num_ref_l0));
    m.insert("num_ref_l1".into(), json!(h.num_ref_l1));
    m.insert("rplm".into(), json!([h.rplm[0], h.rplm[1]]));
    m.insert("rplm_bit".into(), json!(h.rplm_bit));
    m.insert("rplm_end_bit".into(), json!(h.rplm_end_bit));
    m.insert("weighted".into(), json!(h.weighted)); // a Python bool, not an int
    m.insert(
        "mmco".into(),
        Value::Array(
            h.mmco
                .iter()
                .map(|c| {
                    let mut v = vec![json!(c.op)];
                    v.extend(c.args.iter().map(|&a| json!(a)));
                    Value::Array(v)
                })
                .collect(),
        ),
    );
    m.insert("drpm_bit".into(), json!(h.drpm_bit));
    if h.adaptive {
        m.insert("adaptive".into(), json!(1));
    }
    if let Some(x) = h.no_output_prior {
        m.insert("no_output_prior".into(), int(x));
    }
    if let Some(x) = h.long_term_ref {
        m.insert("long_term_ref".into(), int(x));
    }
    m.insert("drpm_end_bit".into(), json!(h.drpm_end_bit));
    m.insert("cabac_init_idc".into(), json!(h.cabac_init_idc));
    m.insert("qp".into(), json!(h.qp));
    m.insert("deblock_idc".into(), json!(h.deblock_idc));
    m.insert("alpha_off".into(), json!(h.alpha_off));
    m.insert("beta_off".into(), json!(h.beta_off));
    m.insert("hdr_bits".into(), json!(h.hdr_bits));
    m.insert("data_bit".into(), json!(h.data_bit));
    m.insert("nal_type".into(), json!(h.nal_type));
    m.insert("ref_idc".into(), json!(h.ref_idc));
    m.insert("offset".into(), json!(h.offset));
    m.insert("size".into(), json!(h.size));
    Value::Object(m)
}

/// `frame-dedup dev headers <file.h264>`: the SPSs and PPSs on one line, then
/// one line per picture (JSONL; oracle: `python -m port.oracle headers`).
pub fn dev_headers(args: &[String]) -> Result<()> {
    let [path] = args else {
        bail!("usage: frame-dedup dev headers <file.h264>");
    };
    let (spss, ppss, pics) = load(Path::new(path))?;
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, out.lock());
    let mut first = Map::new();
    first.insert("spss".into(), Value::Object(spss.iter().map(|(k, s)| (k.to_string(), sps_json(s))).collect()));
    first.insert("ppss".into(), Value::Object(ppss.iter().map(|(k, p)| (k.to_string(), pps_json(p))).collect()));
    writeln!(out, "{}", canonical_text(&Value::Object(first)))?;
    for p in &pics {
        let poc_type = spss[&ppss[&p.slices[0].pps_id].sps_id].poc_type;
        let line = json!({
            "dec": p.dec, "poc": p.poc, "frame_num": p.frame_num, "lists": [p.lists[0], p.lists[1]],
            "dpb_after": p.dpb_after, "period": p.period, "disp": p.disp, "au_start": p.au_start,
            "au_end": p.au_end, "idr": p.idr, "ref_idc": p.ref_idc, "type": p.typ().as_str(),
            "sl": slice_json(&p.slices[0], poc_type),
        });
        writeln!(out, "{}", canonical_text(&line))?;
    }
    out.flush()?;
    Ok(())
}

/// zlib.crc32
pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, t) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *t = c;
    }
    let mut c = !0u32;
    for &b in data {
        c = table[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn with_mmap<T>(path: &str, f: impl FnOnce(&[u8]) -> Result<T>) -> Result<T> {
    let file = File::open(path).with_context(|| format!("cannot open {path}"))?;
    let map = unsafe { Mmap::map(&file)? };
    f(&map)
}

/// `frame-dedup dev escape <file.h264> <n>`: for the first n slice NALs,
/// unescape / escape results (length and CRC-32). Oracle: `python -m port.oracle escape`.
pub fn dev_escape(args: &[String]) -> Result<()> {
    let [path, n] = args else {
        bail!("usage: frame-dedup dev escape <file.h264> <n>");
    };
    let n: usize = n.parse()?;
    with_mmap(path, |data| {
        let mut count = 0;
        for (s, e) in iter_nals(data) {
            let t = data[s] & 0x1f;
            if t != 1 && t != 5 {
                continue;
            }
            if count >= n {
                break;
            }
            count += 1;
            let body = &data[s + 1..e];
            let rbsp = unescape(body);
            let esc = escape(&rbsp);
            let line = json!({
                "offset": s, "unesc": [rbsp.len(), crc32(&rbsp)], "esc": [esc.len(), crc32(&esc)],
                "same": esc == body,
                "esc_raw": [escape(body).len(), crc32(&escape(body))],
                "unesc_raw": [unescape(&esc).len(), crc32(&unescape(&esc))],
            });
            println!("{}", canonical_text(&line));
        }
        Ok(())
    })
}

/// `frame-dedup dev patch-sps <file.h264>`: every SPS NAL, patched to
/// 1001/48000 and to 1/48 (hex). Oracle: `python -m port.oracle patch-sps`.
pub fn dev_patch_sps(args: &[String]) -> Result<()> {
    let [path] = args else {
        bail!("usage: frame-dedup dev patch-sps <file.h264>");
    };
    with_mmap(path, |data| {
        for (s, e) in iter_nals(data) {
            if data[s] & 0x1f != 7 {
                continue;
            }
            let nal = &data[s..e];
            let a = patch_sps_timing(nal, 1001, 48000)?;
            let b = patch_sps_timing(nal, 1, 48)?;
            println!("{}", canonical_text(&json!({"offset": s, "nal": hex(nal), "p1001": hex(&a), "p1_48": hex(&b)})));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::{escape, unescape};

    #[test]
    fn unsupported_downcasts() {
        let e = unsupported("nope");
        assert_eq!(e.downcast_ref::<Unsupported>().map(|u| u.0.as_str()), Some("nope"));
        assert_eq!(e.to_string(), "nope");
        let wrapped = e.context("while loading");
        assert!(wrapped.downcast_ref::<Unsupported>().is_some());
    }

    #[test]
    fn crc32_known() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// Mirror of the Python corpus (LCG, zero-heavy bytes): the chained CRC of
    /// the escape / unescape outputs was computed with `frame_dedup.h264hdr`.
    #[test]
    fn escape_corpus_matches_python() {
        let mut s = 12345u32;
        let mut chain = 0u32;
        for _ in 0..20000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let len = (s >> 24) as usize % 40;
            let mut v = Vec::new();
            for _ in 0..len {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push([0, 0, 0, 1, 2, 3, 4][(s >> 20) as usize % 7]);
            }
            let mut buf = Vec::new();
            buf.extend_from_slice(&chain.to_le_bytes());
            buf.extend_from_slice(&escape(&v));
            buf.push(0xaa);
            buf.extend_from_slice(&unescape(&v));
            chain = crc32(&buf);
        }
        assert_eq!(chain, ESCAPE_CORPUS_CRC);
    }

    const ESCAPE_CORPUS_CRC: u32 = 0xf9a4_7774;
}

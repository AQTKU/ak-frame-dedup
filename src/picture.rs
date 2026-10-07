//! Picture-level decode and encode of slice data. Port of
//! `roundtrip_test.decode_picture` / `encode_picture`, plus the `dev`
//! commands that check the CABAC codec against the Python oracle
//! (`records`, `roundtrip`, `encode-edited`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use serde_json::{Value, json};

use crate::bits::{BitWriter, escape, unescape};
use crate::cabac::{decode_slice, encode_slice};
use crate::h264hdr::{Picture, Ppss, SliceHeader, SliceType, Spss};
use crate::records::{Coefs, Kind, Mb, Mvds, Part, canonical_text};

/// A parsed picture (Python: the dict returned by `decode_picture`).
#[derive(Clone, Debug)]
pub struct Parsed {
    /// The NAL header byte.
    pub nal_header: u8,
    /// Size of the original NAL in bytes (Python `len(dd['nal'])`).
    pub nal_len: usize,
    /// The slice header the records are encoded under.
    pub sl: SliceHeader,
    /// RBSP bytes before slice_data(): `rbsp[..sl.data_bit / 8]` (header plus
    /// cabac_alignment_one_bits). Python keeps these as `bits[:data_bit]`.
    pub head: Vec<u8>,
    pub mbs: Vec<Mb>,
    /// Bit position just past the last CABAC bit read (Python `end`).
    pub end_bit: usize,
}

/// The first slice's NAL (header byte included) and its RBSP (without the
/// header byte). Python `roundtrip_test.load_slice`.
pub fn load_slice<'a>(es: &'a [u8], sl: &SliceHeader) -> Result<(&'a [u8], Vec<u8>)> {
    let nal = es
        .get(sl.offset..sl.offset + sl.size)
        .ok_or_else(|| anyhow!("slice NAL at {}+{} lies outside the stream", sl.offset, sl.size))?;
    Ok((nal, unescape(&nal[1..])))
}

/// Parse the slice data of `pic` from the Annex-B buffer `es`.
pub fn decode_picture(es: &[u8], spss: &Spss, ppss: &Ppss, pic: &Picture) -> Result<Parsed> {
    let sl = &pic.slices[0];
    let (nal, rbsp) = load_slice(es, sl)?;
    let pps = ppss.get(&sl.pps_id).ok_or_else(|| anyhow!("KeyError: pps {}", sl.pps_id))?;
    let sps = spss.get(&pps.sps_id).ok_or_else(|| anyhow!("KeyError: sps {}", pps.sps_id))?;
    if sl.data_bit % 8 != 0 {
        bail!("slice data starts at bit {}, which is not byte-aligned", sl.data_bit);
    }
    let head = rbsp.get(..sl.data_bit / 8).ok_or_else(|| anyhow!("slice header runs past the end of the NAL"))?.to_vec();
    let (mbs, end_bit) = decode_slice(sps, pps, sl, &rbsp)?;
    Ok(Parsed { nal_header: nal[0], nal_len: nal.len(), sl: sl.clone(), head, mbs, end_bit })
}

/// Encode `mbs` under `p`'s header. Returns the RBSP bytes (zero-padded to a
/// whole number of bytes) and the number of bits before the padding.
fn encode_rbsp(p: &Parsed, spss: &Spss, ppss: &Ppss, mbs: &mut [Mb]) -> Result<(Vec<u8>, usize)> {
    let pps = ppss.get(&p.sl.pps_id).ok_or_else(|| anyhow!("KeyError: pps {}", p.sl.pps_id))?;
    let sps = spss.get(&pps.sps_id).ok_or_else(|| anyhow!("KeyError: sps {}", pps.sps_id))?;
    let mut w = BitWriter::with_capacity(p.head.len() + p.nal_len);
    w.put_bytes(&p.head);
    let w = encode_slice(sps, pps, &p.sl, mbs, w)?;
    let bits = w.len_bits();
    Ok((w.finish(), bits))
}

/// Encode `mbs` under `p`'s header. Returns the NAL (header byte + escaped
/// RBSP, no start code), byte-identical to Python's `encode_picture`. The
/// records are updated the way the Python encoder updates them (derived
/// fields such as `dc_l`, `dc_c`, `part` for B).
///
/// Panics on records the Python encoder would raise on (missing fields);
/// use [`try_encode_picture`] to get an error instead.
pub fn encode_picture(p: &Parsed, spss: &Spss, ppss: &Ppss, mbs: &mut [Mb]) -> Vec<u8> {
    try_encode_picture(p, spss, ppss, mbs).unwrap_or_else(|e| panic!("encode_picture: {e:#}"))
}

/// [`encode_picture`], returning an error for a malformed record.
pub fn try_encode_picture(p: &Parsed, spss: &Spss, ppss: &Ppss, mbs: &mut [Mb]) -> Result<Vec<u8>> {
    let (rbsp, _) = encode_rbsp(p, spss, ppss, mbs)?;
    let mut nal = Vec::with_capacity(rbsp.len() + rbsp.len() / 64 + 8);
    nal.push(p.nal_header);
    nal.extend_from_slice(&escape(&rbsp));
    Ok(nal)
}

// ===========================================================================
// dev commands
// ===========================================================================

/// SHA-256 (dev checks only).
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut block = |chunk: &[u8]| {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    };
    let mut chunks = data.chunks_exact(64);
    for c in &mut chunks {
        block(c);
    }
    let rem = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let total = if rem.len() < 56 { 64 } else { 128 };
    tail[total - 8..total].copy_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for c in tail[..total].chunks_exact(64) {
        block(c);
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Run `work(i)` for `i in 0..n` on `threads` threads and hand the results
/// to `sink` in index order.
fn par_ordered<R: Send>(n: usize, threads: usize, work: impl Fn(usize) -> R + Sync, mut sink: impl FnMut(usize, R)) {
    if threads <= 1 {
        for i in 0..n {
            let r = work(i);
            sink(i, r);
        }
        return;
    }
    let next = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(usize, R)>();
    std::thread::scope(|s| {
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, work) = (&next, &work);
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    if tx.send((i, work(i))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut pending = BTreeMap::new();
        let mut want = 0;
        for (i, r) in rx {
            pending.insert(i, r);
            while let Some(r) = pending.remove(&want) {
                sink(want, r);
                want += 1;
            }
        }
    });
}

struct Opts {
    path: String,
    pos: Vec<String>,
    from: usize,
    count: Option<usize>,
    full: bool,
    threads: usize,
}

fn parse_opts(args: &[String], usage: &str) -> Result<Opts> {
    let mut o = Opts { path: String::new(), pos: vec![], from: 0, count: None, full: false, threads: 4 };
    let mut it = args.iter();
    let mut first = true;
    while let Some(a) = it.next() {
        let mut num = |name: &str| -> Result<usize> {
            it.next().ok_or_else(|| anyhow!("{name} needs a value"))?.parse().with_context(|| format!("bad {name}"))
        };
        match a.as_str() {
            "--from" => o.from = num("--from")?,
            "--count" => o.count = Some(num("--count")?),
            "--threads" => o.threads = num("--threads")?.max(1),
            "--full" => o.full = true,
            _ if first => {
                o.path = a.clone();
                first = false;
            }
            _ => o.pos.push(a.clone()),
        }
    }
    if first {
        bail!("usage: {usage}");
    }
    Ok(o)
}

struct Loaded {
    map: Option<Mmap>,
    spss: Spss,
    ppss: Ppss,
    pics: Vec<Picture>,
}

impl Loaded {
    fn open(path: &str) -> Result<Loaded> {
        let (spss, ppss, pics) = crate::stream::load(Path::new(path))?;
        let file = File::open(path).with_context(|| format!("cannot open {path}"))?;
        let map = Some(unsafe { Mmap::map(&file)? });
        Ok(Loaded { map, spss, ppss, pics })
    }

    fn es(&self) -> &[u8] {
        self.map.as_deref().unwrap_or(&[])
    }

    /// The pictures `from..from + count`.
    fn range(&self, o: &Opts) -> std::ops::Range<usize> {
        let a = o.from.min(self.pics.len());
        let b = o.count.map_or(self.pics.len(), |c| (a + c).min(self.pics.len()));
        a..b
    }
}

/// `dev records <file.h264> [--from N] [--count N] [--full] [--threads N]`
pub fn dev_records(args: &[String]) -> Result<()> {
    let o = parse_opts(args, "frame-dedup dev records <file.h264> [--from N] [--count N] [--full] [--threads N]")?;
    let ld = Loaded::open(&o.path)?;
    let range = ld.range(&o);
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    let t0 = Instant::now();
    let mut err = None;
    par_ordered(
        range.len(),
        o.threads,
        |k| -> Result<(Value, f64)> {
            let pic = &ld.pics[range.start + k];
            let t = Instant::now();
            let d = decode_picture(ld.es(), &ld.spss, &ld.ppss, pic).with_context(|| format!("picture {}", pic.dec))?;
            let secs = t.elapsed().as_secs_f64();
            let list = Value::Array(d.mbs.iter().map(Mb::canonical).collect());
            let digest = hex(&sha256(canonical_text(&list).as_bytes()));
            let mut line = json!({"dec": pic.dec, "type": pic.typ().as_str(), "n_mbs": d.mbs.len(), "digest": digest});
            if o.full {
                line["mbs"] = list;
            }
            Ok((line, secs))
        },
        |_, r| match r {
            Ok((line, _)) if err.is_none() => {
                let _ = writeln!(out, "{}", canonical_text(&line));
            }
            Ok(_) => {}
            Err(e) => {
                err.get_or_insert(e);
            }
        },
    );
    out.flush()?;
    eprintln!("records: {} pictures in {:.2}s", range.len(), t0.elapsed().as_secs_f64());
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Bit `i` of `b` (MSB first).
fn bit(b: &[u8], i: usize) -> u8 {
    (b[i >> 3] >> (7 - (i & 7))) & 1
}

struct RtResult {
    line: Value,
    ok: bool,
    dec_s: f64,
    enc_s: f64,
}

fn roundtrip_one(ld: &Loaded, pic: &Picture) -> Result<RtResult> {
    let t0 = Instant::now();
    let mut d = decode_picture(ld.es(), &ld.spss, &ld.ppss, pic)?;
    let t1 = Instant::now();
    let data_bit = d.sl.data_bit;
    let (nal, rbsp) = load_slice(ld.es(), &d.sl)?;
    // the slice must end on its stop bit, then zero alignment, then only
    // cabac_zero_words (0x0000) if any
    let end = d.end_bit;
    let nbits = rbsp.len() * 8;
    let tail_bits = nbits - end;
    let stop_ok = end > 0 && bit(&rbsp, end - 1) == 1 && (end..nbits).all(|i| bit(&rbsp, i) == 0);
    // x264 streams (the Amazon file) have the stop bit one bit after where the decoder stops
    let late_stop = !stop_ok && end < nbits && bit(&rbsp, end) == 1 && (end + 1..nbits).all(|i| bit(&rbsp, i) == 0);
    let mut mbs = std::mem::take(&mut d.mbs);
    let (enc, enc_total_bits) = encode_rbsp(&d, &ld.spss, &ld.ppss, &mut mbs)?;
    let mut nal2 = vec![d.nal_header];
    nal2.extend_from_slice(&escape(&enc));
    let t2 = Instant::now();
    let same_bits = enc_total_bits == end && (data_bit..end).all(|i| bit(&enc, i) == bit(&rbsp, i));
    let same_nal = nal2 == nal;
    let ok = stop_ok && same_bits && same_nal;
    let line = json!({
        "dec": pic.dec, "type": pic.typ().as_str(), "nal_len": nal.len(), "stop_bit_ok": stop_ok,
        "tail_bits": tail_bits, "bits_equal": same_bits, "nal_equal": same_nal, "late_stop_bit": late_stop,
    });
    Ok(RtResult { line, ok, dec_s: (t1 - t0).as_secs_f64(), enc_s: (t2 - t1).as_secs_f64() })
}

/// `dev roundtrip <file.h264> [--from N] [--count N] [--threads N]`: decode every
/// picture, re-encode the unchanged records, compare with the original NAL.
pub fn dev_roundtrip(args: &[String]) -> Result<()> {
    let o = parse_opts(args, "frame-dedup dev roundtrip <file.h264> [--from N] [--count N] [--threads N]")?;
    let ld = Loaded::open(&o.path)?;
    let range = ld.range(&o);
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    let t0 = Instant::now();
    let (mut n_ok, mut n_bad, mut dec_s, mut enc_s) = (0usize, 0usize, 0.0, 0.0);
    let mut failed: Vec<Value> = vec![];
    par_ordered(
        range.len(),
        o.threads,
        |k| {
            let pic = &ld.pics[range.start + k];
            roundtrip_one(&ld, pic).unwrap_or_else(|e| RtResult {
                line: json!({"dec": pic.dec, "type": pic.typ().as_str(), "error": format!("{e:#}")}),
                ok: false,
                dec_s: 0.0,
                enc_s: 0.0,
            })
        },
        |_, r| {
            let _ = writeln!(out, "{}", canonical_text(&r.line));
            dec_s += r.dec_s;
            enc_s += r.enc_s;
            if r.ok {
                n_ok += 1;
            } else {
                n_bad += 1;
                failed.push(r.line["dec"].clone());
            }
        },
    );
    let summary = json!({"summary": true, "pictures": range.len(), "ok": n_ok, "failed": n_bad, "failed_dec": failed});
    writeln!(out, "{}", canonical_text(&summary))?;
    out.flush()?;
    let n = range.len().max(1) as f64;
    eprintln!(
        "roundtrip: {} pictures, {} ok, {} failed; {:.2}s wall; mean decode {:.1} ms, mean encode {:.1} ms",
        range.len(),
        n_ok,
        n_bad,
        t0.elapsed().as_secs_f64(),
        dec_s / n * 1e3,
        enc_s / n * 1e3
    );
    if n_bad > 0 {
        bail!("{n_bad} pictures did not round-trip");
    }
    Ok(())
}

/// SplitMix64, the generator of the deterministic edit spec (port/oracle.py has the same).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// An I_PCM record like `patch.pcm_record` with deterministic samples.
fn edit_pcm_record(x: u16, y: u16, seed: u64) -> Mb {
    let mut mb = Mb::new(x, y);
    mb.kind = Kind::Pcm;
    mb.intra = true;
    mb.pcm = true;
    mb.cbp = 0x2f;
    let mut s = Box::new([0u8; 384]);
    for (i, v) in s.iter_mut().enumerate() {
        *v = ((x as u64 * 31 + y as u64 * 17 + i as u64 * 7 + seed) & 255) as u8;
    }
    mb.samples = Some(s);
    mb
}

/// Counts of the edits made: (pcm, skip made explicit, empty chroma DC, qpd changed).
type EditCounts = [usize; 4];

/// The deterministic edit spec of `dev encode-edited` (port/oracle.py `encode_edited`
/// does the same on Python records): exercises encode paths that a plain
/// round trip never hits.
fn apply_edit(mbs: &mut [Mb], t: SliceType, w: usize, seed: u64) -> EditCounts {
    let mut rng = SplitMix64(seed);
    let period = 6 + seed % 29;
    let mut counts = [0; 4];
    for (i, mb) in mbs.iter_mut().enumerate() {
        let r = rng.next();
        let (x, y) = ((i % w) as u16, (i / w) as u16);
        if r % period == 0 {
            *mb = edit_pcm_record(x, y, seed);
            counts[0] += 1;
        } else if t == SliceType::P && mb.skip && (r >> 20) % 6 == 0 {
            // patch.to_explicit(mb, 'P', None)
            mb.skip = false;
            mb.kind = Kind::P;
            mb.part = Some(Part::P16x16);
            mb.refs = Some([vec![Some(0)], vec![None]]);
            mb.mvds = Some(Mvds::Parts([vec![Some([0, 0])], vec![None]]));
            mb.direct16 = false;
            mb.cbp = 0;
            mb.t8 = false;
            mb.qpd = 0;
            mb.coef = Some(Box::<Coefs>::default());
            counts[1] += 1;
        } else if t == SliceType::B && mb.skip && (r >> 20) % 6 == 0 {
            // a skipped B macroblock as explicit B_Direct_16x16
            mb.skip = false;
            mb.kind = Kind::B;
            mb.btype = Some(0);
            mb.part = Some(Part::Direct);
            mb.direct16 = true;
            mb.cbp = 0;
            mb.t8 = false;
            mb.qpd = 0;
            mb.coef = Some(Box::<Coefs>::default());
            counts[1] += 1;
        } else if !mb.skip && mb.cbp == 0 && mb.kind != Kind::I16 && (r >> 32) % 4 == 0 {
            // patch.fix_qp: an empty coded chroma DC so the MB can carry a qp delta
            mb.cbp |= 0x10;
            if mb.coef.is_none() {
                mb.coef = Some(Box::<Coefs>::default());
            }
            mb.qpd = ((r >> 40) % 5) as i32 - 2;
            counts[2] += 1;
        } else if !mb.skip && (mb.cbp != 0 || mb.kind == Kind::I16) && (r >> 48) % 9 == 0 {
            mb.qpd = ((r >> 52) % 7) as i32 - 3;
            counts[3] += 1;
        }
    }
    counts
}

/// `dev encode-edited <file.h264> <dec> <seed> [<dec> <seed> ...]`: decode picture `dec`,
/// apply the deterministic edit, encode, print the NAL's length and SHA-256.
pub fn dev_encode_edited(args: &[String]) -> Result<()> {
    const USAGE: &str = "frame-dedup dev encode-edited <file.h264> <dec> <seed> [<dec> <seed> ...]";
    let o = parse_opts(args, USAGE)?;
    if o.pos.is_empty() || o.pos.len() % 2 != 0 {
        bail!("usage: {USAGE}");
    }
    let ld = Loaded::open(&o.path)?;
    for pair in o.pos.chunks(2) {
        let dec: usize = pair[0].parse().context("bad dec")?;
        let seed: u64 = pair[1].parse().context("bad seed")?;
        let pic = ld.pics.get(dec).ok_or_else(|| anyhow!("no picture {dec}"))?;
        let mut d = decode_picture(ld.es(), &ld.spss, &ld.ppss, pic)?;
        let w = ld.spss[&ld.ppss[&d.sl.pps_id].sps_id].mb_w;
        let mut mbs = std::mem::take(&mut d.mbs);
        let counts = apply_edit(&mut mbs, d.sl.slice_type, w, seed);
        let nal = try_encode_picture(&d, &ld.spss, &ld.ppss, &mut mbs)?;
        let line = json!({
            "dec": dec, "seed": seed, "type": pic.typ().as_str(), "len": nal.len(), "sha256": hex(&sha256(&nal)),
            "pcm": counts[0], "explicit": counts[1], "empty_dc": counts[2], "qpd": counts[3],
        });
        println!("{}", canonical_text(&line));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_vectors() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let long = vec![b'a'; 1000];
        assert_eq!(hex(&sha256(&long)), "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3");
    }

    #[test]
    fn splitmix_reference() {
        // First outputs of SplitMix64 seeded with 0 (the reference implementation's test vector).
        let mut r = SplitMix64(0);
        assert_eq!(r.next(), 0xe220a8397b1dcdaf);
        assert_eq!(r.next(), 0x6e789e6aa1b965f4);
    }
}

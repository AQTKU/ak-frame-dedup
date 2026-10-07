//! CABAC slice-data codec (High profile, 4:2:0, 8-bit, progressive).
//! Port of `frame_dedup/h264cabac.py` (`Coder`, `SliceCodec`).
//!
//! One code path decodes bins from a bitstream or encodes them from the
//! records; keep that design, it is why the round trip is byte-exact.
//!
//! Python passes `val=None` to decode and a value to encode. Here both
//! `Coder` and `SliceCodec` carry a `const DEC: bool`: every syntax element is
//! coded by one function, and the value expressions that only make sense when
//! encoding sit inside `ev!(..)`, which is a constant `0` when decoding.
//!
//! Scope: single-slice pictures, frame_mbs_only, no MBAFF, no FMO, 4:2:0,
//! 8-bit. That is what this project's streams use.

use anyhow::{Result, anyhow, bail};

use crate::bits::BitWriter;
use crate::h264hdr::{Pps, SliceHeader, SliceType, Sps};
use crate::records::{
    B_SUB, B_TYPES, BLK_XY, Coefs, Kind, Mb, Mvds, P_SUB, P_TYPES, Part, Pred, SubShape, mb_parts,
    sub_parts,
};

#[path = "cabac_tables.rs"]
mod tables;
use tables::{INIT_I, INIT_PB, LAST_8X8, RANGE_LPS, TRANS_LPS};

const SIG_8X8: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, 4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7, 7, 6, 11,
    12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, 12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13, 11, 14, 10, 12,
];

// ctxBlockCat: 0 luma DC (I16), 1 luma AC (I16), 2 luma 4x4, 3 chroma DC,
// 4 chroma AC, 5 luma 8x8
const CBF_BASE: [usize; 5] = [85, 89, 93, 97, 101];
const SIG_BASE: [usize; 6] = [105, 120, 134, 149, 152, 402];
const LAST_BASE: [usize; 6] = [166, 181, 195, 210, 213, 417];
const ABS_BASE: [usize; 6] = [227, 237, 247, 257, 266, 426];
const MAX_COEFF: [usize; 6] = [16, 15, 16, 4, 15, 64];
const ABS1_CTX: [usize; 8] = [1, 2, 3, 4, 0, 0, 0, 0];
const ABSGT1_CTX: [usize; 8] = [5, 5, 5, 5, 6, 7, 8, 9];
const TRANS_EQ1: [usize; 8] = [1, 2, 3, 3, 4, 5, 6, 7];
const TRANS_GT1: [usize; 8] = [4, 4, 4, 4, 5, 6, 7, 7];

/// Python `TRANS_MPS`: min(s + 1, 62) for s < 63, and 63 stays 63.
#[inline]
fn trans_mps(s: usize) -> u8 {
    if s < 63 { (s + 1).min(62) as u8 } else { 63 }
}

/// A value expression that is only evaluated when encoding (decode passes
/// `None` in the Python); `0` when decoding.
macro_rules! ev {
    ($e:expr) => {
        if DEC { 0 } else { $e }
    };
}

fn missing(what: &str) -> anyhow::Error {
    anyhow!("KeyError: macroblock record has no {what}")
}

// ---------------------------------------------------------------------------
// The arithmetic engine
// ---------------------------------------------------------------------------

/// CABAC arithmetic engine. `DEC` reads bits from `data`, otherwise it
/// writes them to `out`.
pub struct Coder<'a, const DEC: bool> {
    st: [u8; 1024],
    mps: [u8; 1024],
    // decoder
    data: &'a [u8],
    nbits: usize,
    /// Bit position in `data` (Python `Coder.pos`).
    pub pos: usize,
    off: u32,
    /// Set when the decoder tried to read past the end of the data; the bits
    /// read from then on are zeros. Python raises IndexError at that point.
    pub err: bool,
    // encoder
    low: u32,
    first: bool,
    outstanding: u32,
    pub out: BitWriter,
    rng: u32,
}

impl<'a, const DEC: bool> Coder<'a, DEC> {
    /// `data` and `pos` (a bit position) are used when decoding, `out` (which
    /// already holds the bits before slice_data(), byte-aligned) when encoding.
    pub fn new(slice_qp: i32, init_idc: u32, slice_is_i: bool, data: &'a [u8], pos: usize, out: BitWriter) -> Result<Self> {
        let tab: &[(i8, i8); 1024] = if slice_is_i {
            &INIT_I
        } else {
            INIT_PB.get(init_idc as usize).ok_or_else(|| anyhow!("IndexError: cabac_init_idc {init_idc}"))?
        };
        let qp = slice_qp.clamp(0, 51);
        let mut st = [0u8; 1024];
        let mut mps = [0u8; 1024];
        for (i, &(m, n)) in tab.iter().enumerate() {
            let pre = (((m as i32 * qp) >> 4) + n as i32).clamp(1, 126);
            if pre <= 63 {
                st[i] = (63 - pre) as u8;
                mps[i] = 0;
            } else {
                st[i] = (pre - 64) as u8;
                mps[i] = 1;
            }
        }
        let mut c = Coder {
            st,
            mps,
            data,
            nbits: data.len() * 8,
            pos,
            off: 0,
            err: false,
            low: 0,
            first: true,
            outstanding: 0,
            out,
            rng: 510,
        };
        if DEC {
            c.init_dec();
        } else {
            c.init_enc();
        }
        Ok(c)
    }

    /// Read `n <= 9` bits (zeros after the end, with `err` set).
    #[inline]
    fn read(&mut self, n: u32) -> u32 {
        let end = self.pos + n as usize;
        if end > self.nbits {
            self.err = true;
            self.pos = end;
            return 0;
        }
        let b = self.pos >> 3;
        let w = ((self.data[b] as u32) << 8) | self.data.get(b + 1).copied().unwrap_or(0) as u32;
        let sh = 16 - (self.pos & 7) as u32 - n;
        self.pos = end;
        (w >> sh) & ((1 << n) - 1)
    }

    fn init_dec(&mut self) {
        self.rng = 510;
        self.off = self.read(9);
    }

    fn init_enc(&mut self) {
        self.low = 0;
        self.rng = 510;
        self.first = true;
        self.outstanding = 0;
    }

    #[inline]
    fn put(&mut self, b: u32) {
        if self.first {
            self.first = false;
        } else {
            self.out.put_bit(b != 0);
        }
        while self.outstanding > 0 {
            self.out.put_bit(b == 0);
            self.outstanding -= 1;
        }
    }

    #[inline]
    fn renorm_e(&mut self) {
        while self.rng < 256 {
            if self.low < 256 {
                self.put(0);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put(1);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.rng <<= 1;
            self.low <<= 1;
        }
    }

    /// Code one context-coded bin. `val` is ignored when decoding.
    #[inline]
    pub fn bin(&mut self, ctx: usize, val: u32) -> u32 {
        let s = self.st[ctx] as usize;
        let m = self.mps[ctx] as u32;
        let lps = RANGE_LPS[s][((self.rng >> 6) & 3) as usize] as u32;
        self.rng -= lps;
        let val = if DEC {
            let v = if self.off >= self.rng {
                self.off -= self.rng;
                self.rng = lps;
                1 - m
            } else {
                m
            };
            if self.rng < 256 {
                let n = self.rng.leading_zeros() - 23;
                self.rng <<= n;
                self.off = (self.off << n) | self.read(n);
            }
            v
        } else {
            if val != m {
                self.low += self.rng;
                self.rng = lps;
            }
            self.renorm_e();
            val
        };
        if val != m {
            if s == 0 {
                self.mps[ctx] = (1 - m) as u8;
            }
            self.st[ctx] = TRANS_LPS[s];
        } else {
            self.st[ctx] = trans_mps(s);
        }
        val
    }

    #[inline]
    pub fn bypass(&mut self, val: u32) -> u32 {
        if DEC {
            self.off = (self.off << 1) | self.read(1);
            if self.off >= self.rng {
                self.off -= self.rng;
                return 1;
            }
            return 0;
        }
        self.low <<= 1;
        if val != 0 {
            self.low += self.rng;
        }
        if self.low >= 1024 {
            self.put(1);
            self.low -= 1024;
        } else if self.low < 512 {
            self.put(0);
        } else {
            self.low -= 512;
            self.outstanding += 1;
        }
        val
    }

    /// end_of_slice_flag / the I_PCM bin. Encoding a 1 flushes the engine
    /// and writes the stop bit.
    pub fn term(&mut self, val: u32) -> u32 {
        self.rng -= 2;
        if DEC {
            if self.off >= self.rng {
                return 1;
            }
            if self.rng < 256 {
                let n = self.rng.leading_zeros() - 23;
                self.rng <<= n;
                self.off = (self.off << n) | self.read(n);
            }
            return 0;
        }
        if val != 0 {
            self.low += self.rng;
            self.rng = 2;
            self.renorm_e();
            self.put((self.low >> 9) & 1);
            self.out.put_bit((self.low >> 8) & 1 != 0);
            self.out.put_bit(true);
        } else {
            self.renorm_e();
        }
        val
    }

    /// Raw bits around an I_PCM macroblock: after the mb_type, align, then
    /// read (decode: fills `samples`) or write (encode: needs `samples`) 384
    /// bytes, and restart the engine.
    pub fn pcm(&mut self, samples: &mut Option<Box<[u8; 384]>>) -> Result<()> {
        if DEC {
            self.pos = (self.pos + 7) & !7;
            if self.pos + 384 * 8 > self.nbits {
                self.err = true;
                bail!("IndexError: bitstream read past the end (I_PCM samples)");
            }
            let mut s = Box::new([0u8; 384]);
            s.copy_from_slice(&self.data[self.pos >> 3..(self.pos >> 3) + 384]);
            self.pos += 384 * 8;
            *samples = Some(s);
            self.init_dec();
        } else {
            let s = samples.as_ref().ok_or_else(|| missing("samples"))?;
            self.out.align_zeros();
            self.out.put_bytes(&s[..]);
            self.init_enc();
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The slice codec
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum T {
    I,
    P,
    B,
}

/// Which coefficient block of a macroblock (Python `coef` dict keys).
#[derive(Clone, Copy)]
enum Key {
    Ldc,
    L8(usize),
    L4(usize),
    Cdc(usize),
    Cac(usize, usize),
}

fn coef_slot(c: &mut Coefs, key: Key) -> &mut Option<Vec<i32>> {
    match key {
        Key::Ldc => &mut c.ldc,
        Key::L8(i) => &mut c.l8[i],
        Key::L4(i) => &mut c.l4[i],
        Key::Cdc(i) => &mut c.cdc[i],
        Key::Cac(comp, i) => &mut c.cac[comp][i],
    }
}

/// The left (A) and upper (B) neighbour macroblocks, when available.
#[derive(Clone, Copy)]
struct Nb<'m> {
    a: Option<&'m Mb>,
    b: Option<&'m Mb>,
}

/// Parse or write the slice_data() of one picture.
pub struct SliceCodec<'a, const DEC: bool> {
    c: Coder<'a, DEC>,
    w: usize,
    h: usize,
    w4: usize,
    transform_8x8: bool,
    t: T,
    n_ref: [u32; 2],
    lists: usize,
    // 4x4-granular neighbour state used for context derivation, indexed y * w4 + x
    ref_: [Vec<i8>; 2],
    mvd: [Vec<[u8; 2]>; 2],
    dir: Vec<u8>,
    cbf_l: Vec<u8>,
    /// Chroma coded_block_flags, 2 per MB each way: index cy * (2 * w) + cx.
    cbf_c: [Vec<u8>; 2],
    t8: Vec<u8>,
    /// Intra4x4/8x8 modes, -1 = other
    imode: Vec<i8>,
    prev_qpd: i32,
}

/// Decode the slice data that starts at bit `data_bit` of `rbsp`. Returns the
/// macroblock records and the bit position just past the last bit read
/// (Python `Coder.pos` after `run`).
pub fn decode_slice(sps: &Sps, pps: &Pps, sl: &SliceHeader, rbsp: &[u8]) -> Result<(Vec<Mb>, usize)> {
    let c = Coder::<true>::new(sl.qp, sl.cabac_init_idc, sl.slice_type == SliceType::I, rbsp, sl.data_bit, BitWriter::new())?;
    let mut codec = SliceCodec::<true>::new(c, sps, pps, sl)?;
    let n = codec.w * codec.h;
    let mut mbs: Vec<Mb> = (0..n).map(|i| Mb::new((i % codec.w) as u16, (i / codec.w) as u16)).collect();
    codec.run(&mut mbs)?;
    Ok((mbs, codec.c.pos))
}

/// Encode `mbs` as slice data appended to `out` (which holds the bits before
/// slice_data(), byte-aligned). The records are updated the way the Python
/// encoder updates them. The stop bit is the last bit written; the bits are
/// not padded.
pub fn encode_slice(sps: &Sps, pps: &Pps, sl: &SliceHeader, mbs: &mut [Mb], out: BitWriter) -> Result<BitWriter> {
    let c = Coder::<false>::new(sl.qp, sl.cabac_init_idc, sl.slice_type == SliceType::I, &[], 0, out)?;
    let mut codec = SliceCodec::<false>::new(c, sps, pps, sl)?;
    codec.run(mbs)?;
    Ok(codec.c.out)
}

impl<'a, const DEC: bool> SliceCodec<'a, DEC> {
    fn new(c: Coder<'a, DEC>, sps: &Sps, pps: &Pps, hdr: &SliceHeader) -> Result<Self> {
        let (w, h) = (sps.mb_w, sps.mb_h);
        let t = match hdr.slice_type {
            SliceType::I => T::I,
            SliceType::P => T::P,
            SliceType::B => T::B,
            other => bail!("unsupported slice type {}", other.as_str()),
        };
        let (w4, h4) = (w * 4, h * 4);
        Ok(SliceCodec {
            c,
            w,
            h,
            w4,
            transform_8x8: pps.transform_8x8,
            t,
            n_ref: [hdr.num_ref_l0, hdr.num_ref_l1],
            lists: match t {
                T::B => 2,
                T::P => 1,
                T::I => 0,
            },
            ref_: [vec![-1; w4 * h4], vec![-1; w4 * h4]],
            mvd: [vec![[0; 2]; w4 * h4], vec![[0; 2]; w4 * h4]],
            dir: vec![0; w4 * h4],
            cbf_l: vec![0; w4 * h4],
            cbf_c: [vec![0; w * h * 4], vec![0; w * h * 4]],
            t8: vec![0; w4 * h4],
            imode: vec![-1; w4 * h4],
            prev_qpd: 0,
        })
    }

    // ------------------------------------------------------------------ top level
    /// Decode into (DEC) or encode the given list of macroblock records.
    fn run(&mut self, mbs: &mut [Mb]) -> Result<()> {
        let n = self.w * self.h;
        if mbs.len() < n {
            bail!("IndexError: {} macroblock records for a picture of {n}", mbs.len());
        }
        self.prev_qpd = 0;
        for addr in 0..n {
            let (mbx, mby) = (addr % self.w, addr / self.w);
            let (done, rest) = mbs.split_at_mut(addr);
            let mb = &mut rest[0];
            let nb = Nb {
                a: if mbx > 0 { Some(&done[addr - 1]) } else { None },
                b: if mby > 0 { Some(&done[addr - self.w]) } else { None },
            };
            mb.x = mbx as u16;
            mb.y = mby as u16;
            if self.t != T::I {
                let inc = nb.a.is_some_and(|m| !m.skip) as usize + nb.b.is_some_and(|m| !m.skip) as usize;
                let ctx = if self.t == T::B { 24 } else { 11 } + inc;
                mb.skip = self.c.bin(ctx, ev!(mb.skip as u32)) != 0;
            } else {
                mb.skip = false;
            }
            if mb.skip {
                self.skip_mb(mb);
            } else {
                self.macroblock(mb, nb)?;
            }
            let end = self.c.term(ev!((addr == n - 1) as u32));
            if self.c.err {
                bail!("IndexError: bitstream read past the end at mb {addr}");
            }
            if DEC && end != 0 && addr != n - 1 {
                bail!("early end of slice at mb {addr}");
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ skip
    fn skip_mb(&mut self, mb: &mut Mb) {
        mb.kind = Kind::Skip;
        mb.intra = false;
        mb.pcm = false;
        mb.i16 = false;
        mb.cbp = 0;
        mb.t8 = false;
        mb.cmode = 0;
        mb.qpd = 0;
        mb.direct16 = self.t == T::B;
        let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
        for y in y0..y0 + 4 {
            for x in x0..x0 + 4 {
                let i = y * self.w4 + x;
                self.ref_[0][i] = if self.t == T::P { 0 } else { -1 };
                self.ref_[1][i] = -1;
                self.mvd[0][i] = [0; 2];
                self.mvd[1][i] = [0; 2];
                self.dir[i] = (self.t == T::B) as u8;
                self.cbf_l[i] = 0;
                self.t8[i] = 0;
                self.imode[i] = -1;
            }
        }
        self.set_cbf_c(mb, 0);
        mb.dc_l = false;
        mb.dc_c = [false, false];
        self.prev_qpd = 0;
    }

    fn set_cbf_c(&mut self, mb: &Mb, v: u8) {
        let cw = self.w * 2;
        for comp in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    self.cbf_c[comp][(mb.y as usize * 2 + y) * cw + mb.x as usize * 2 + x] = v;
                }
            }
        }
    }

    // ------------------------------------------------------------------ mb_type
    fn mb_type(&mut self, mb: &mut Mb, nb: Nb) -> Result<()> {
        let (a, b) = (nb.a, nb.b);
        match self.t {
            T::I => {
                let f = |n: Option<&Mb>| n.is_some_and(|m| matches!(m.kind, Kind::I16 | Kind::Pcm)) as usize;
                let inc = f(a) + f(b);
                return self.intra_type(mb, 3, inc, true);
            }
            T::P => {
                let intra = self.c.bin(14, ev!(mb.kind.is_intra() as u32));
                if intra != 0 {
                    return self.intra_type(mb, 17, 0, false);
                }
                if DEC {
                    let t = if self.c.bin(15, 0) == 0 {
                        3 * self.c.bin(16, 0) as usize
                    } else {
                        2 - self.c.bin(17, 0) as usize
                    };
                    mb.kind = Kind::P;
                    mb.part = Some(P_TYPES[t]);
                } else {
                    let part = mb.part.ok_or_else(|| missing("part"))?;
                    let t = P_TYPES
                        .iter()
                        .position(|&p| p == part)
                        .ok_or_else(|| anyhow!("ValueError: {} is not a P partitioning", part.as_str()))?;
                    if t == 0 || t == 3 {
                        self.c.bin(15, 0);
                        self.c.bin(16, (t == 3) as u32);
                    } else {
                        self.c.bin(15, 1);
                        self.c.bin(17, (t == 1) as u32);
                    }
                }
                return Ok(());
            }
            T::B => {}
        }
        // B slice
        let f = |n: Option<&Mb>| n.is_some_and(|m| !m.direct16) as usize;
        let inc = f(a) + f(b);
        if DEC {
            let c = &mut self.c;
            let t: u32 = if c.bin(27 + inc, 0) == 0 {
                0
            } else if c.bin(30, 0) == 0 {
                1 + c.bin(32, 0)
            } else {
                let mut bits = c.bin(31, 0) << 3;
                bits |= c.bin(32, 0) << 2;
                bits |= c.bin(32, 0) << 1;
                bits |= c.bin(32, 0);
                if bits < 8 {
                    bits + 3
                } else if bits == 13 {
                    return self.intra_type(mb, 32, 0, false);
                } else if bits == 14 {
                    11
                } else if bits == 15 {
                    22
                } else {
                    ((bits << 1) | c.bin(32, 0)) - 4
                }
            };
            mb.kind = Kind::B;
            mb.btype = Some(t as u8);
        } else {
            if mb.kind.is_intra() {
                self.c.bin(27 + inc, 1);
                self.c.bin(30, 1);
                for (k, v) in [(31, 1), (32, 1), (32, 0), (32, 1)] {
                    self.c.bin(k, v);
                }
                return self.intra_type(mb, 32, 0, false);
            }
            let t = mb.btype.ok_or_else(|| missing("btype"))? as u32;
            let c = &mut self.c;
            if t == 0 {
                c.bin(27 + inc, 0);
            } else if t <= 2 {
                c.bin(27 + inc, 1);
                c.bin(30, 0);
                c.bin(32, t - 1);
            } else {
                c.bin(27 + inc, 1);
                c.bin(30, 1);
                let (bits, extra) = if t <= 10 {
                    (t - 3, None)
                } else if t == 11 {
                    (14, None)
                } else if t == 22 {
                    (15, None)
                } else {
                    let v = t + 4;
                    (v >> 1, Some(v & 1))
                };
                c.bin(31, (bits >> 3) & 1);
                c.bin(32, (bits >> 2) & 1);
                c.bin(32, (bits >> 1) & 1);
                c.bin(32, bits & 1);
                if let Some(e) = extra {
                    c.bin(32, e);
                }
            }
        }
        let bt = mb.btype.ok_or_else(|| missing("btype"))? as usize;
        let (part, _) = B_TYPES.get(bt).ok_or_else(|| anyhow!("IndexError: btype {bt}"))?;
        mb.part = Some(*part);
        Ok(())
    }

    /// I mb_type binarization (prefix-free part), shared by I/P/B slices.
    fn intra_type(&mut self, mb: &mut Mb, base: usize, inc: usize, islice: bool) -> Result<()> {
        let c = &mut self.c;
        let o = if islice { 2 } else { 0 };
        let i1 = islice as usize;
        let ctx0 = if islice { base + inc } else { base };
        if DEC {
            if c.bin(ctx0, 0) == 0 {
                mb.kind = Kind::I4;
                return Ok(());
            }
            if c.term(0) != 0 {
                mb.kind = Kind::Pcm;
                return Ok(());
            }
            let luma = c.bin(base + 1 + o, 0);
            let mut chroma = 0;
            if c.bin(base + 2 + o, 0) != 0 {
                chroma = 1 + c.bin(base + 2 + i1 + o, 0);
            }
            let mut pm = c.bin(base + 3 + i1 + o, 0) << 1;
            pm |= c.bin(base + 3 + 2 * i1 + o, 0);
            mb.kind = Kind::I16;
            mb.i16pred = pm as u8;
            mb.i16cbp_c = chroma as u8;
            mb.i16cbp_l = luma as u8;
            return Ok(());
        }
        let k = mb.kind;
        if !k.is_intra() {
            bail!("KeyError: I macroblock type of a {} record", k.as_str());
        }
        c.bin(ctx0, if matches!(k, Kind::I4 | Kind::I8) { 0 } else { 1 });
        if matches!(k, Kind::I4 | Kind::I8) {
            return Ok(());
        }
        c.term((k == Kind::Pcm) as u32);
        if k == Kind::Pcm {
            return Ok(());
        }
        c.bin(base + 1 + o, mb.i16cbp_l as u32);
        c.bin(base + 2 + o, (mb.i16cbp_c > 0) as u32);
        if mb.i16cbp_c > 0 {
            c.bin(base + 2 + i1 + o, mb.i16cbp_c as u32 - 1);
        }
        c.bin(base + 3 + i1 + o, (mb.i16pred >> 1) as u32);
        c.bin(base + 3 + 2 * i1 + o, (mb.i16pred & 1) as u32);
        Ok(())
    }

    // ------------------------------------------------------------------ macroblock
    fn macroblock(&mut self, mb: &mut Mb, nb: Nb) -> Result<()> {
        self.mb_type(mb, nb)?;
        let k = mb.kind;
        mb.intra = k.is_intra();
        mb.pcm = k == Kind::Pcm;
        mb.i16 = k == Kind::I16;
        mb.direct16 = k == Kind::B && mb.btype.ok_or_else(|| missing("btype"))? == 0;
        // mb.setdefault('t8', 0): a record always has t8
        let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
        // reset neighbour state for this MB
        for y in y0..y0 + 4 {
            for x in x0..x0 + 4 {
                let i = y * self.w4 + x;
                for l in 0..2 {
                    self.ref_[l][i] = -1;
                    self.mvd[l][i] = [0; 2];
                }
                self.dir[i] = 0;
                self.cbf_l[i] = 0;
                self.t8[i] = 0;
                self.imode[i] = -1;
            }
        }
        self.set_cbf_c(mb, 0);
        mb.dc_l = false;
        mb.dc_c = [false, false];
        if k == Kind::Pcm {
            self.c.pcm(&mut mb.samples)?;
            mb.cbp = 0x2f;
            mb.t8 = false;
            mb.cmode = 0;
            mb.qpd = 0;
            for y in y0..y0 + 4 {
                for x in x0..x0 + 4 {
                    self.cbf_l[y * self.w4 + x] = 1;
                }
            }
            self.set_cbf_c(mb, 1);
            mb.dc_l = true;
            mb.dc_c = [true, true];
            self.prev_qpd = 0;
            return Ok(());
        }
        if mb.intra {
            self.intra_pred(mb, nb)?;
        } else {
            match mb.part.ok_or_else(|| missing("part"))? {
                Part::P8x8 => self.sub_mb_pred(mb)?,
                Part::Direct => {
                    for y in y0..y0 + 4 {
                        for x in x0..x0 + 4 {
                            self.dir[y * self.w4 + x] = 1;
                        }
                    }
                }
                _ => self.mb_pred(mb)?,
            }
        }
        // coded_block_pattern
        if k != Kind::I16 {
            self.cbp(mb, nb);
            if (mb.cbp & 15) != 0 && !mb.intra && self.transform_8x8 && self.t8_allowed(mb)? {
                let ctx = 399 + self.t8_inc(mb);
                mb.t8 = self.c.bin(ctx, ev!(mb.t8 as u32)) != 0;
            }
        } else {
            mb.cbp = (mb.i16cbp_c << 4) | if mb.i16cbp_l != 0 { 15 } else { 0 };
        }
        if mb.t8 {
            for y in y0..y0 + 4 {
                for x in x0..x0 + 4 {
                    self.t8[y * self.w4 + x] = 1;
                }
            }
        }
        if mb.cbp != 0 || k == Kind::I16 {
            self.qp_delta(mb);
            self.residual(mb, nb)?;
        } else {
            mb.qpd = 0;
            self.prev_qpd = 0;
        }
        Ok(())
    }

    fn t8_allowed(&self, mb: &Mb) -> Result<bool> {
        if mb.part == Some(Part::P8x8) {
            let sub = mb.sub.ok_or_else(|| missing("sub"))?;
            for &st in &sub {
                let shape = if self.t == T::B {
                    let shape = B_SUB.get(st as usize).ok_or_else(|| anyhow!("IndexError: B sub type {st}"))?.0;
                    if shape == SubShape::Direct {
                        continue; // direct_8x8_inference is 1 in these streams
                    }
                    shape
                } else {
                    *P_SUB.get(st as usize).ok_or_else(|| anyhow!("IndexError: P sub type {st}"))?
                };
                if shape != SubShape::S8x8 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn t8_inc(&self, mb: &Mb) -> usize {
        let (x, y) = (mb.x as usize * 4, mb.y as usize * 4);
        let a = if x > 0 { self.t8[y * self.w4 + x - 1] } else { 0 };
        let b = if y > 0 { self.t8[(y - 1) * self.w4 + x] } else { 0 };
        (a + b) as usize
    }

    // ------------------------------------------------------------------ intra
    fn intra_pred(&mut self, mb: &mut Mb, nb: Nb) -> Result<()> {
        if matches!(mb.kind, Kind::I4 | Kind::I8) {
            if self.transform_8x8 {
                let ctx = 399 + self.t8_inc(mb);
                let t8 = self.c.bin(ctx, ev!((mb.kind == Kind::I8) as u32)) != 0;
                mb.kind = if t8 { Kind::I8 } else { Kind::I4 };
                mb.t8 = t8;
            }
            let n = if mb.kind == Kind::I8 { 4 } else { 16 };
            let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
            if DEC {
                mb.ipred.clear();
                mb.imodes.clear();
            } else if mb.imodes.len() < n {
                bail!("IndexError: imodes has {} entries, {n} needed", mb.imodes.len());
            }
            for i in 0..n {
                let (bx, by, ncell) = if n == 4 {
                    ((i % 2) * 2, (i / 2) * 2, 2)
                } else {
                    let (bx, by) = BLK_XY[i];
                    (bx, by, 1)
                };
                let (gx, gy) = (x0 + bx, y0 + by);
                let ma = if gx > 0 { self.imode[gy * self.w4 + gx - 1] } else { -1 };
                let mb_ = if gy > 0 { self.imode[(gy - 1) * self.w4 + gx] } else { -1 };
                let pred = (if ma < 0 { 2 } else { ma }).min(if mb_ < 0 { 2 } else { mb_ });
                let mode: i8;
                if DEC {
                    let c = &mut self.c;
                    let r: i8;
                    if c.bin(68, 0) != 0 {
                        r = -1;
                        mode = pred;
                    } else {
                        let rr = c.bin(69, 0) | (c.bin(69, 0) << 1) | (c.bin(69, 0) << 2);
                        r = rr as i8;
                        mode = if r < pred { r } else { r + 1 };
                    }
                    mb.ipred.push(r);
                    mb.imodes.push(mode as u8);
                } else {
                    mode = mb.imodes[i] as i8;
                    let r = if mode == pred {
                        -1
                    } else if mode < pred {
                        mode
                    } else {
                        mode - 1
                    };
                    let c = &mut self.c;
                    c.bin(68, (r < 0) as u32);
                    if r >= 0 {
                        c.bin(69, (r & 1) as u32);
                        c.bin(69, ((r >> 1) & 1) as u32);
                        c.bin(69, ((r >> 2) & 1) as u32);
                    }
                }
                for cy in 0..ncell {
                    for cx in 0..ncell {
                        self.imode[(gy + cy) * self.w4 + gx + cx] = mode;
                    }
                }
            }
        }
        // intra_chroma_pred_mode
        let f = |n: Option<&Mb>| n.is_some_and(|m| m.intra && !m.pcm && m.cmode != 0) as usize;
        let inc = f(nb.a) + f(nb.b);
        let c = &mut self.c;
        if DEC {
            mb.cmode = if c.bin(64 + inc, 0) == 0 {
                0
            } else if c.bin(67, 0) == 0 {
                1
            } else {
                2 + c.bin(67, 0) as u8
            };
        } else {
            let m = mb.cmode;
            c.bin(64 + inc, (m > 0) as u32);
            if m > 0 {
                c.bin(67, (m > 1) as u32);
                if m > 1 {
                    c.bin(67, (m > 2) as u32);
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------ inter
    fn ref_idx(&mut self, lst: usize, bx: usize, by: usize, val: u32) -> Result<u32> {
        let mut inc = 0;
        if bx > 0 {
            let i = by * self.w4 + bx - 1;
            if self.ref_[lst][i] > 0 && !(self.t == T::B && self.dir[i] != 0) {
                inc += 1;
            }
        }
        if by > 0 {
            let i = (by - 1) * self.w4 + bx;
            if self.ref_[lst][i] > 0 && !(self.t == T::B && self.dir[i] != 0) {
                inc += 2;
            }
        }
        let mut ctx = 54 + inc;
        if DEC {
            let mut v = 0u32;
            while self.c.bin(ctx, 0) != 0 {
                v += 1;
                if self.c.err || v > 127 {
                    bail!("ref_idx out of range");
                }
                ctx = 54 + if v == 1 { 4 } else { 5 };
            }
            return Ok(v);
        }
        for i in 0..val {
            self.c.bin(ctx, 1);
            ctx = 54 + if i == 0 { 4 } else { 5 };
        }
        self.c.bin(ctx, 0);
        Ok(val)
    }

    fn mvd_comp(&mut self, lst: usize, bx: usize, by: usize, comp: usize, val: i32) -> Result<i32> {
        let mut s = 0i32;
        if bx > 0 {
            s += self.mvd[lst][by * self.w4 + bx - 1][comp] as i32;
        }
        if by > 0 {
            s += self.mvd[lst][(by - 1) * self.w4 + bx][comp] as i32;
        }
        let base = if comp == 0 { 40 } else { 47 };
        let inc = if s < 3 {
            0
        } else if s <= 32 {
            1
        } else {
            2
        };
        let c = &mut self.c;
        if DEC {
            if c.bin(base + inc, 0) == 0 {
                return Ok(0);
            }
            let mut v = 1i32;
            let mut ctx = base + 3;
            while v < 9 && c.bin(ctx, 0) != 0 {
                v += 1;
                if v < 5 {
                    ctx += 1;
                }
            }
            if v >= 9 {
                let mut k = 3;
                while c.bypass(0) != 0 {
                    v += 1 << k;
                    k += 1;
                    if c.err || k > 24 {
                        bail!("mvd out of range");
                    }
                }
                while k > 0 {
                    k -= 1;
                    v += (c.bypass(0) as i32) << k;
                }
            }
            return Ok(if c.bypass(0) != 0 { -v } else { v });
        }
        let a = val.abs();
        c.bin(base + inc, (a > 0) as u32);
        if a == 0 {
            return Ok(0);
        }
        let mut ctx = base + 3;
        let mut v = 1;
        while v < 9 {
            if a > v {
                c.bin(ctx, 1);
                v += 1;
                if v < 5 {
                    ctx += 1;
                }
            } else {
                c.bin(ctx, 0);
                break;
            }
        }
        if a >= 9 {
            let mut r = a - 9;
            let mut k = 3;
            while r >= (1 << k) {
                c.bypass(1);
                r -= 1 << k;
                k += 1;
            }
            c.bypass(0);
            while k > 0 {
                k -= 1;
                c.bypass(((r >> k) & 1) as u32);
            }
        }
        c.bypass((val < 0) as u32);
        Ok(val)
    }

    /// 16x16 / 16x8 / 8x16 partitions in P and B slices.
    fn mb_pred(&mut self, mb: &mut Mb) -> Result<()> {
        let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
        let part = mb.part.ok_or_else(|| missing("part"))?;
        let parts = mb_parts(part);
        let np = parts.len();
        let preds: &[Pred] = if self.t == T::P {
            &[Pred::L0, Pred::L0][..np]
        } else {
            let bt = mb.btype.ok_or_else(|| missing("btype"))? as usize;
            B_TYPES.get(bt).ok_or_else(|| anyhow!("IndexError: btype {bt}"))?.1
        };
        if preds.len() < np {
            bail!("IndexError: prediction directions of btype");
        }
        if DEC {
            mb.refs = Some([vec![None; np], vec![None; np]]);
            mb.mvds = Some(Mvds::Parts([vec![None; np], vec![None; np]]));
        }
        let refs = mb.refs.as_mut().ok_or_else(|| missing("refs"))?;
        for lst in 0..self.lists {
            for (p, blocks) in parts.iter().enumerate() {
                if !preds[p].uses(lst) {
                    continue;
                }
                let (bx, by) = (x0 + blocks[0].0, y0 + blocks[0].1);
                let slot = refs[lst].get_mut(p).ok_or_else(|| anyhow!("IndexError: refs[{lst}][{p}]"))?;
                let r = if self.n_ref[lst] > 1 {
                    let val = if DEC { 0 } else { slot.ok_or_else(|| anyhow!("TypeError: refs[{lst}][{p}] is None"))? as u32 };
                    self.ref_idx(lst, bx, by, val)? as i8
                } else {
                    0
                };
                *slot = Some(r);
                for &(x, y) in blocks.iter() {
                    self.ref_[lst][(y0 + y) * self.w4 + x0 + x] = r;
                }
            }
        }
        let Some(Mvds::Parts(mvds)) = mb.mvds.as_mut() else {
            return Err(missing("mvds (partition form)"));
        };
        for lst in 0..self.lists {
            for (p, blocks) in parts.iter().enumerate() {
                if !preds[p].uses(lst) {
                    continue;
                }
                let (bx, by) = (x0 + blocks[0].0, y0 + blocks[0].1);
                let slot = mvds[lst].get_mut(p).ok_or_else(|| anyhow!("IndexError: mvds[{lst}][{p}]"))?;
                let old = if DEC { [0, 0] } else { slot.ok_or_else(|| anyhow!("TypeError: mvds[{lst}][{p}] is None"))? };
                let dx = self.mvd_comp(lst, bx, by, 0, old[0])?;
                let dy = self.mvd_comp(lst, bx, by, 1, old[1])?;
                *slot = Some([dx, dy]);
                let a = [dx.unsigned_abs().min(70) as u8, dy.unsigned_abs().min(70) as u8];
                for &(x, y) in blocks.iter() {
                    self.mvd[lst][(y0 + y) * self.w4 + x0 + x] = a;
                }
            }
        }
        Ok(())
    }

    fn sub_mb_pred(&mut self, mb: &mut Mb) -> Result<()> {
        let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
        let mut subs = [0u8; 4];
        if DEC {
            for sub in subs.iter_mut() {
                let c = &mut self.c;
                let st: u32 = if self.t == T::P {
                    if c.bin(21, 0) != 0 {
                        0
                    } else if c.bin(22, 0) == 0 {
                        1
                    } else if c.bin(23, 0) != 0 {
                        2
                    } else {
                        3
                    }
                } else if c.bin(36, 0) == 0 {
                    0
                } else if c.bin(37, 0) == 0 {
                    1 + c.bin(39, 0)
                } else {
                    let mut st = 3;
                    if c.bin(38, 0) != 0 {
                        if c.bin(39, 0) != 0 {
                            st = 11 + c.bin(39, 0);
                        } else {
                            st = 7 + 2 * c.bin(39, 0);
                            st += c.bin(39, 0);
                        }
                    } else {
                        st += 2 * c.bin(39, 0);
                        st += c.bin(39, 0);
                    }
                    st
                };
                *sub = st as u8;
            }
            mb.sub = Some(subs);
        } else {
            subs = mb.sub.ok_or_else(|| missing("sub"))?;
            for &st in &subs {
                if self.t == T::P {
                    if st >= 4 {
                        bail!("IndexError: P sub type {st}");
                    }
                    let c = &mut self.c;
                    c.bin(21, (st == 0) as u32);
                    if st != 0 {
                        c.bin(22, (st != 1) as u32);
                        if st != 1 {
                            c.bin(23, (st == 2) as u32);
                        }
                    }
                } else {
                    if st >= 13 {
                        bail!("IndexError: B sub type {st}");
                    }
                    self.b_sub_enc(st as u32);
                }
            }
        }
        let is_p = self.t == T::P;
        let info = move |st: u8| -> (SubShape, Pred) {
            if is_p { (P_SUB[st as usize], Pred::L0) } else { B_SUB[st as usize] }
        };
        // direct sub-blocks are marked before refs are coded
        for (i, &st) in subs.iter().enumerate() {
            let (shape, _) = info(st);
            if shape == SubShape::Direct {
                for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    self.dir[(y0 + (i / 2) * 2 + y) * self.w4 + x0 + (i % 2) * 2 + x] = 1;
                }
            }
        }
        if DEC {
            mb.refs = Some([vec![None; 4], vec![None; 4]]);
            mb.mvds = Some(Mvds::Sub([[vec![], vec![], vec![], vec![]], [vec![], vec![], vec![], vec![]]]));
        }
        let refs = mb.refs.as_mut().ok_or_else(|| missing("refs"))?;
        for lst in 0..self.lists {
            for (i, &st) in subs.iter().enumerate() {
                let (shape, pred) = info(st);
                if shape == SubShape::Direct || !pred.uses(lst) {
                    continue;
                }
                let (bx, by) = (x0 + (i % 2) * 2, y0 + (i / 2) * 2);
                let slot = refs[lst].get_mut(i).ok_or_else(|| anyhow!("IndexError: refs[{lst}][{i}]"))?;
                let r = if self.n_ref[lst] > 1 {
                    let val = if DEC { 0 } else { slot.ok_or_else(|| anyhow!("TypeError: refs[{lst}][{i}] is None"))? as u32 };
                    self.ref_idx(lst, bx, by, val)? as i8
                } else {
                    0
                };
                *slot = Some(r);
                for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    self.ref_[lst][(by + y) * self.w4 + bx + x] = r;
                }
            }
        }
        let Some(Mvds::Sub(mvds)) = mb.mvds.as_mut() else {
            return Err(missing("mvds (sub-macroblock form)"));
        };
        for lst in 0..self.lists {
            for (i, &st) in subs.iter().enumerate() {
                let (shape, pred) = info(st);
                if shape == SubShape::Direct || !pred.uses(lst) {
                    continue;
                }
                let (ox, oy) = (x0 + (i % 2) * 2, y0 + (i / 2) * 2);
                for (j, blocks) in sub_parts(shape).iter().enumerate() {
                    let (bx, by) = (ox + blocks[0].0, oy + blocks[0].1);
                    let old = if DEC {
                        [0, 0]
                    } else {
                        *mvds[lst][i].get(j).ok_or_else(|| anyhow!("IndexError: mvds[{lst}][{i}][{j}]"))?
                    };
                    let dx = self.mvd_comp(lst, bx, by, 0, old[0])?;
                    let dy = self.mvd_comp(lst, bx, by, 1, old[1])?;
                    if DEC {
                        mvds[lst][i].push([dx, dy]);
                    }
                    let a = [dx.unsigned_abs().min(70) as u8, dy.unsigned_abs().min(70) as u8];
                    for &(x, y) in blocks.iter() {
                        self.mvd[lst][(oy + y) * self.w4 + ox + x] = a;
                    }
                }
            }
        }
        Ok(())
    }

    fn b_sub_enc(&mut self, st: u32) {
        let c = &mut self.c;
        c.bin(36, (st != 0) as u32);
        if st == 0 {
            return;
        }
        c.bin(37, (st >= 3) as u32);
        if st < 3 {
            c.bin(39, st - 1);
            return;
        }
        if st >= 11 {
            c.bin(38, 1);
            c.bin(39, 1);
            c.bin(39, st - 11);
        } else if st >= 7 {
            c.bin(38, 1);
            c.bin(39, 0);
            c.bin(39, ((st - 7) >> 1) & 1);
            c.bin(39, (st - 7) & 1);
        } else {
            c.bin(38, 0);
            c.bin(39, ((st - 3) >> 1) & 1);
            c.bin(39, (st - 3) & 1);
        }
    }

    // ------------------------------------------------------------------ cbp / qp
    fn cbp(&mut self, mb: &mut Mb, nb: Nb) {
        let val = ev!(mb.cbp as u32);
        let mut cbp = 0u32;
        // condTermFlagN = 0 when N unavailable, I_PCM, or its 8x8 is coded
        let cond_luma = |n: Option<&Mb>, bit: usize| -> usize {
            match n {
                None => 0,
                Some(n) if n.pcm => 0,
                Some(n) if !n.skip && (n.cbp >> bit) & 1 != 0 => 0,
                _ => 1,
            }
        };
        for i in 0..4usize {
            let (bx, by) = (i % 2, i / 2);
            let ca = if bx == 0 { cond_luma(nb.a, by * 2 + 1) } else { ((cbp >> (i - 1)) & 1 == 0) as usize };
            let cb = if by == 0 { cond_luma(nb.b, 2 + bx) } else { ((cbp >> (i - 2)) & 1 == 0) as usize };
            let v = self.c.bin(73 + ca + 2 * cb, ev!((val >> i) & 1));
            cbp |= v << i;
        }
        let chroma = |n: Option<&Mb>| -> u8 {
            match n {
                None => 0,
                Some(n) if n.pcm => 2,
                Some(n) if n.skip => 0,
                Some(n) => n.cbp >> 4,
            }
        };
        let (ca, cb) = (chroma(nb.a), chroma(nb.b));
        let v0 = self.c.bin(77 + (ca > 0) as usize + 2 * (cb > 0) as usize, ev!(((val >> 4) > 0) as u32));
        if v0 != 0 {
            let v1 = self.c.bin(77 + 4 + (ca == 2) as usize + 2 * (cb == 2) as usize, ev!(((val >> 4) == 2) as u32));
            cbp |= (1 + v1) << 4;
        }
        mb.cbp = cbp as u8;
    }

    fn qp_delta(&mut self, mb: &mut Mb) {
        let c = &mut self.c;
        let ctx0 = 60 + (self.prev_qpd != 0) as usize;
        if DEC {
            let v: i32;
            if c.bin(ctx0, 0) == 0 {
                v = 0;
            } else {
                let mut k = 1i32;
                let mut ctx = 62;
                while c.bin(ctx, 0) != 0 {
                    k += 1;
                    ctx = 63;
                    if c.err || k > 1000 {
                        break;
                    }
                }
                v = if k & 1 != 0 { (k + 1) / 2 } else { -(k / 2) };
            }
            mb.qpd = v;
        } else {
            let v = mb.qpd;
            let k = if v > 0 { 2 * v - 1 } else { -2 * v };
            c.bin(ctx0, (k > 0) as u32);
            if k != 0 {
                let mut ctx = 62;
                for _ in 0..k - 1 {
                    c.bin(ctx, 1);
                    ctx = 63;
                }
                c.bin(ctx, 0);
            }
        }
        self.prev_qpd = mb.qpd;
    }

    // ------------------------------------------------------------------ residual
    fn residual(&mut self, mb: &mut Mb, nb: Nb) -> Result<()> {
        if DEC {
            mb.coef = Some(Box::default());
        } else if mb.coef.is_none() {
            return Err(missing("coef"));
        }
        let (x0, y0) = (mb.x as usize * 4, mb.y as usize * 4);
        let intra = mb.intra;
        let cbp = mb.cbp;
        let unavail = intra as usize;
        if mb.i16 {
            let dc_cond = |n: Option<&Mb>| -> usize {
                match n {
                    None => unavail,
                    Some(n) if n.pcm => 1,
                    Some(n) => (n.i16 && n.dc_l) as usize,
                }
            };
            let inc = dc_cond(nb.a) + 2 * dc_cond(nb.b);
            mb.dc_l = self.block(mb, Key::Ldc, 0, inc)?;
        }
        for i8 in 0..4 {
            if (cbp >> i8) & 1 == 0 {
                continue;
            }
            if mb.t8 {
                self.block(mb, Key::L8(i8), 5, 0)?;
                let (bx, by) = ((i8 % 2) * 2, (i8 / 2) * 2);
                for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    self.cbf_l[(y0 + by + y) * self.w4 + x0 + bx + x] = 1;
                }
                continue;
            }
            for i4 in 0..4 {
                let idx = i8 * 4 + i4;
                let (bx, by) = BLK_XY[idx];
                let (gx, gy) = (x0 + bx, y0 + by);
                let ca = if gx > 0 { self.luma_cond(gx - 1, gy, mb, nb) } else { unavail };
                let cb = if gy > 0 { self.luma_cond(gx, gy - 1, mb, nb) } else { unavail };
                let cat = if mb.i16 { 1 } else { 2 };
                let f = self.block(mb, Key::L4(idx), cat, ca + 2 * cb)?;
                self.cbf_l[gy * self.w4 + gx] = f as u8;
            }
        }
        if cbp >> 4 != 0 {
            for comp in 0..2 {
                let cdc = |n: Option<&Mb>| -> usize {
                    match n {
                        None => unavail,
                        Some(n) if n.pcm => 1,
                        Some(n) if n.skip || (n.cbp >> 4) == 0 => 0,
                        Some(n) => n.dc_c[comp] as usize,
                    }
                };
                let inc = cdc(nb.a) + 2 * cdc(nb.b);
                mb.dc_c[comp] = self.block(mb, Key::Cdc(comp), 3, inc)?;
            }
        }
        if cbp >> 4 == 2 {
            let cw = self.w * 2;
            for comp in 0..2 {
                for i in 0..4 {
                    let (cx, cy) = (mb.x as usize * 2 + i % 2, mb.y as usize * 2 + i / 2);
                    let ca = if cx > 0 { self.chroma_cond(comp, cx - 1, cy, mb, nb) } else { unavail };
                    let cb = if cy > 0 { self.chroma_cond(comp, cx, cy - 1, mb, nb) } else { unavail };
                    let f = self.block(mb, Key::Cac(comp, i), 4, ca + 2 * cb)?;
                    self.cbf_c[comp][cy * cw + cx] = f as u8;
                }
            }
        }
        Ok(())
    }

    /// coded_block_flag of the luma 4x4 block at (gx, gy) as a context
    /// neighbour of the current macroblock `cur`.
    fn luma_cond(&self, gx: usize, gy: usize, cur: &Mb, nb: Nb) -> usize {
        let (mx, my) = (gx / 4, gy / 4);
        let i = gy * self.w4 + gx;
        if (mx, my) == (cur.x as usize, cur.y as usize) {
            return self.cbf_l[i] as usize;
        }
        let n = if mx != cur.x as usize { nb.a } else { nb.b }.expect("neighbour inside the picture");
        if n.skip {
            return 0;
        }
        if n.pcm {
            return 1;
        }
        let b8 = ((gy % 4) / 2) * 2 + (gx % 4) / 2;
        if (n.cbp >> b8) & 1 == 0 {
            return 0;
        }
        self.cbf_l[i] as usize
    }

    fn chroma_cond(&self, comp: usize, cx: usize, cy: usize, cur: &Mb, nb: Nb) -> usize {
        let (mx, my) = (cx / 2, cy / 2);
        let i = cy * (self.w * 2) + cx;
        if (mx, my) == (cur.x as usize, cur.y as usize) {
            return self.cbf_c[comp][i] as usize;
        }
        let n = if mx != cur.x as usize { nb.a } else { nb.b }.expect("neighbour inside the picture");
        if n.skip {
            return 0;
        }
        if n.pcm {
            return 1;
        }
        if n.cbp >> 4 != 2 {
            return 0;
        }
        self.cbf_c[comp][i] as usize
    }

    /// residual_block_cabac. Returns coded_block_flag. Coefficients are
    /// stored in scan order under `mb.coef`.
    fn block(&mut self, mb: &mut Mb, key: Key, cat: usize, cbf_inc: usize) -> Result<bool> {
        let n = MAX_COEFF[cat];
        let coefs = mb.coef.as_mut().ok_or_else(|| missing("coef"))?;
        let slot = coef_slot(coefs, key);
        let c = &mut self.c;
        if DEC {
            if cat != 5 && c.bin(CBF_BASE[cat] + cbf_inc, 0) == 0 {
                return Ok(false);
            }
            let mut sig = [0u8; 64];
            let mut nsig = 0;
            let mut last = false;
            let mut i = 0;
            while i < n - 1 {
                let (si, li) = sig_ctx(cat, i);
                if c.bin(si, 0) != 0 {
                    sig[nsig] = i as u8;
                    nsig += 1;
                    if c.bin(li, 0) != 0 {
                        last = true;
                        break;
                    }
                }
                i += 1;
            }
            if !last {
                sig[nsig] = (n - 1) as u8;
                nsig += 1;
            }
            let mut out = vec![0i32; n];
            let mut node = 0;
            for &pos in sig[..nsig].iter().rev() {
                let a: i32;
                if c.bin(ABS_BASE[cat] + ABS1_CTX[node], 0) == 0 {
                    a = 1;
                    node = TRANS_EQ1[node];
                } else {
                    let ctx = ABS_BASE[cat] + ABSGT1_CTX[node];
                    node = TRANS_GT1[node];
                    let mut aa = 2;
                    while aa < 15 && c.bin(ctx, 0) != 0 {
                        aa += 1;
                    }
                    if aa >= 15 {
                        let mut k = 0;
                        while c.bypass(0) != 0 {
                            k += 1;
                            if c.err || k > 24 {
                                bail!("coefficient level out of range");
                            }
                        }
                        let mut v = 1i32;
                        for _ in 0..k {
                            v = 2 * v + c.bypass(0) as i32;
                        }
                        aa = v + 14;
                    }
                    a = aa;
                }
                out[pos as usize] = if c.bypass(0) != 0 { -a } else { a };
            }
            *slot = Some(out);
            return Ok(true);
        }
        let coef = slot.as_ref();
        if cat != 5 {
            let cbf = c.bin(CBF_BASE[cat] + cbf_inc, coef.is_some() as u32);
            if cbf == 0 {
                return Ok(false);
            }
        }
        let coef = coef.ok_or_else(|| anyhow!("TypeError: no coefficients for an 8x8 block"))?;
        let sig: Vec<usize> = coef.iter().enumerate().filter(|(_, v)| **v != 0).map(|(i, _)| i).collect();
        let last = *sig.last().ok_or_else(|| anyhow!("IndexError: coded block without a nonzero coefficient"))?;
        for i in 0..(last + 1).min(n - 1) {
            let (si, li) = sig_ctx(cat, i);
            let s = (coef[i] != 0) as u32;
            c.bin(si, s);
            if s != 0 {
                c.bin(li, (i == last) as u32);
            }
        }
        let mut node = 0;
        for &pos in sig.iter().rev() {
            let v = coef[pos];
            let a = v.unsigned_abs();
            c.bin(ABS_BASE[cat] + ABS1_CTX[node], (a > 1) as u32);
            if a == 1 {
                node = TRANS_EQ1[node];
            } else {
                let ctx = ABS_BASE[cat] + ABSGT1_CTX[node];
                node = TRANS_GT1[node];
                let m = a - 1; // coeff_abs_level_minus1, prefix TU cMax 14
                for _ in 0..m.min(14) - 1 {
                    c.bin(ctx, 1);
                }
                if m < 14 {
                    c.bin(ctx, 0);
                } else {
                    let mut r = m - 14;
                    let mut k = 0;
                    while r >= (1 << k) {
                        c.bypass(1);
                        r -= 1 << k;
                        k += 1;
                    }
                    c.bypass(0);
                    while k > 0 {
                        k -= 1;
                        c.bypass((r >> k) & 1);
                    }
                }
            }
            c.bypass((v < 0) as u32);
        }
        Ok(true)
    }
}

#[inline]
fn sig_ctx(cat: usize, i: usize) -> (usize, usize) {
    if cat == 5 {
        return (402 + SIG_8X8[i] as usize, 417 + LAST_8X8[i] as usize);
    }
    if cat == 3 {
        let j = i.min(2);
        return (SIG_BASE[3] + j, LAST_BASE[3] + j);
    }
    (SIG_BASE[cat] + i, LAST_BASE[cat] + i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a pseudo-random sequence of context, bypass and terminate bins
    /// (with a PCM-style restart in the middle) and decode it back.
    #[test]
    fn engine_roundtrip() {
        let mut seed = 12345u64;
        let mut rnd = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as u32
        };
        // (kind, ctx, value): 0 = bin, 1 = bypass, 2 = term(0)
        let mut ops = vec![];
        for i in 0..20000 {
            let r = rnd();
            let v = if r & 0x70 == 0 { 1 } else { 0 }; // skewed, so states move and LPS happen
            match r % 7 {
                0..=3 => ops.push((0u8, (r >> 8) as usize % 1024, v ^ ((r >> 20) & (i & 1))))
                ,
                4 | 5 => ops.push((1, 0, (r >> 9) & 1)),
                _ => ops.push((2, 0, 0)),
            }
        }
        let mut e = Coder::<false>::new(30, 1, false, &[], 0, BitWriter::new()).unwrap();
        let half = ops.len() / 2;
        let samples: Option<Box<[u8; 384]>> = Some(Box::new(std::array::from_fn(|i| (i * 7) as u8)));
        for (n, &(k, ctx, v)) in ops.iter().enumerate() {
            if n == half {
                assert_eq!(e.term(1), 1);
                let mut s = samples.clone();
                e.pcm(&mut s).unwrap();
            }
            match k {
                0 => {
                    e.bin(ctx, v);
                }
                1 => {
                    e.bypass(v);
                }
                _ => {
                    e.term(0);
                }
            }
        }
        e.term(1);
        let mut bytes = e.out.finish();
        bytes.extend_from_slice(&[0, 0]);
        let mut d = Coder::<true>::new(30, 1, false, &bytes, 0, BitWriter::new()).unwrap();
        for (n, &(k, ctx, v)) in ops.iter().enumerate() {
            if n == half {
                assert_eq!(d.term(0), 1);
                let mut s = None;
                d.pcm(&mut s).unwrap();
                assert_eq!(s, samples);
            }
            let got = match k {
                0 => d.bin(ctx, 0),
                1 => d.bypass(0),
                _ => d.term(0),
            };
            let want = if k == 2 { 0 } else { v };
            assert_eq!(got, want, "op {n}");
        }
        assert_eq!(d.term(0), 1);
        assert!(!d.err);
    }

    #[test]
    fn read_past_end_is_an_error_flag() {
        let mut d = Coder::<true>::new(26, 0, true, &[0xff], 0, BitWriter::new()).unwrap();
        assert!(d.err); // the 9 bits of the initialisation do not fit in one byte
        let _ = d.bin(3, 0);
    }
}

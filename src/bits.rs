//! Bit-level helpers: RBSP escaping, bit reader and writer, in-place bit
//! patching. Port of the helpers in `h264hdr.py` (`unescape`, `escape`,
//! `set_bits`, `Bits`) and `roundtrip_test.py` (`to_bits`, `from_bits`).

use anyhow::{Result, bail};

/// Strip emulation-prevention bytes (00 00 03 -> 00 00).
///
/// Mirrors `re.sub(rb'\x00\x00\x03', b'\x00\x00', b)`: matches are
/// non-overlapping and found left to right, so `00 00 00 03` becomes
/// `00 00 00` and `00 00 03 03` becomes `00 00 03`.
pub fn unescape(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len());
    let n = b.len();
    let mut copied = 0; // input bytes before `copied` have been written
    // j scans for the 03 that ends a match; the match starts at j - 2 >= copied.
    let mut j = 2;
    while j < n {
        let c = b[j];
        if c == 3 && b[j - 1] == 0 && b[j - 2] == 0 {
            out.extend_from_slice(&b[copied..j]); // through the two zeros
            copied = j + 1;
            j = copied + 2;
        } else if c == 0 {
            j += 1;
        } else {
            // a match ending at j + 1 or j + 2 would need b[j] == 0
            j += 3;
        }
    }
    out.extend_from_slice(&b[copied..]);
    out
}

/// Insert emulation-prevention bytes into an RBSP.
///
/// Mirrors `re.sub(rb'\x00\x00(?=[\x00-\x03]|\Z)', b'\x00\x00\x03', b)`: a
/// pair of zeros followed by a byte <= 3 (or the end of the data) gets a 03
/// after it; the pair is consumed, so `00 00 00 00 00` becomes
/// `00 00 03 00 00 03 00`.
pub fn escape(b: &[u8]) -> Vec<u8> {
    let n = b.len();
    let mut out = Vec::with_capacity(n + n / 64 + 4);
    let mut i = 0;
    let mut copied = 0;
    while i + 1 < n {
        if b[i + 1] != 0 {
            i += 2; // neither i nor i + 1 can start a pair of zeros
        } else if b[i] == 0 && (i + 2 == n || b[i + 2] <= 3) {
            out.extend_from_slice(&b[copied..i + 2]);
            out.push(3);
            i += 2;
            copied = i;
        } else {
            i += 1;
        }
    }
    out.extend_from_slice(&b[copied..]);
    out
}

/// Overwrite `n` bits at bit offset `pos` (MSB first) with the low `n` bits
/// of `value`.
pub fn set_bits(buf: &mut [u8], pos: usize, n: u32, value: u64) {
    for i in 0..n as usize {
        let bit = ((value >> (n as usize - 1 - i)) & 1) as u8;
        let (byte, sh) = ((pos + i) >> 3, 7 - ((pos + i) & 7));
        buf[byte] = (buf[byte] & !(1 << sh)) | (bit << sh);
    }
}

/// One bit per byte (0 or 1), MSB first. Python: `roundtrip_test.to_bits`.
pub fn to_bits(b: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(b.len() * 8);
    for &byte in b {
        for k in 0..8 {
            out.push((byte >> (7 - k)) & 1);
        }
    }
    out
}

/// Pack bits (0/1 per byte) into bytes. Python: `roundtrip_test.from_bits`.
/// Like the Python, a final partial byte is NOT padded to the left
/// (`[1, 0]` gives `0x02`, not `0x80`); callers align to a byte boundary
/// first. Prefer [`BitWriter`] for new code.
pub fn from_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bits.len().div_ceil(8));
    for chunk in bits.chunks(8) {
        let mut v = 0u8;
        for &b in chunk {
            v = (v << 1) | b;
        }
        out.push(v);
    }
    out
}

/// MSB-first bit reader over a byte slice (Python: `Bits`). Reading past the
/// end is an error (Python raises IndexError).
#[derive(Clone)]
pub struct BitReader<'a> {
    d: &'a [u8],
    /// Current position in bits (Python: `Bits.p`).
    pub p: usize,
}

impl<'a> BitReader<'a> {
    pub fn new(d: &'a [u8]) -> Self {
        BitReader { d, p: 0 }
    }

    /// Total number of bits in the data.
    pub fn len_bits(&self) -> usize {
        self.d.len() * 8
    }

    /// u(n), n <= 32.
    #[inline]
    pub fn u(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            bail!("u({n}): more than 32 bits at once");
        }
        let end = self.p + n as usize;
        if end > self.d.len() * 8 {
            bail!("IndexError: bitstream read past the end ({} bits)", self.d.len() * 8);
        }
        let first = self.p >> 3;
        let last = (end - 1) >> 3;
        let mut acc = 0u64;
        for &byte in &self.d[first..=last] {
            acc = (acc << 8) | byte as u64;
        }
        let total = (last - first + 1) * 8;
        let shift = total - ((self.p & 7) + n as usize);
        self.p = end;
        Ok(((acc >> shift) & ((1u64 << n) - 1)) as u32)
    }

    /// A single bit as a bool.
    #[inline]
    pub fn flag(&mut self) -> Result<bool> {
        Ok(self.u(1)? != 0)
    }

    /// ue(v).
    pub fn ue(&mut self) -> Result<u32> {
        let mut z = 0u32;
        while self.u(1)? == 0 {
            z += 1;
            if z > 32 {
                bail!("ue(v) with more than 32 leading zero bits");
            }
        }
        let v = (1u64 << z) - 1 + self.u(z)? as u64;
        if v > u32::MAX as u64 {
            bail!("ue(v) value {v} does not fit 32 bits");
        }
        Ok(v as u32)
    }

    /// ue(v) + 1.
    pub fn ue1(&mut self) -> Result<u32> {
        match self.ue()?.checked_add(1) {
            Some(v) => Ok(v),
            None => bail!("ue(v) + 1 overflows 32 bits"),
        }
    }

    /// se(v).
    pub fn se(&mut self) -> Result<i32> {
        let k = self.ue()? as i64;
        let v = if k & 1 == 1 { (k + 1) / 2 } else { -(k / 2) };
        Ok(v as i32)
    }
}

/// MSB-first bit writer. Bits are appended; [`BitWriter::finish`] pads the
/// last byte with zeros (use [`BitWriter::align_ones`] first for
/// cabac_alignment_one_bits).
#[derive(Clone, Default, Debug)]
pub struct BitWriter {
    buf: Vec<u8>,
    /// Pending bits not yet in `buf` (right-aligned), fewer than 8.
    acc: u32,
    nacc: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(bytes: usize) -> Self {
        BitWriter { buf: Vec::with_capacity(bytes), acc: 0, nacc: 0 }
    }

    /// Number of bits written so far.
    #[inline]
    pub fn len_bits(&self) -> usize {
        self.buf.len() * 8 + self.nacc as usize
    }

    #[inline]
    pub fn is_aligned(&self) -> bool {
        self.nacc == 0
    }

    /// Append the low `n` bits of `v` (n <= 32), MSB first.
    #[inline]
    pub fn put(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        let v = if n == 32 { v as u64 } else { (v as u64) & ((1u64 << n) - 1) };
        let mut acc = ((self.acc as u64) << n) | v;
        let mut nacc = self.nacc + n;
        while nacc >= 8 {
            nacc -= 8;
            self.buf.push((acc >> nacc) as u8);
        }
        acc &= (1u64 << nacc) - 1;
        self.acc = acc as u32;
        self.nacc = nacc;
    }

    /// Append the low `n` bits of a 64-bit value (n <= 64).
    pub fn put64(&mut self, n: u32, v: u64) {
        if n > 32 {
            self.put(n - 32, (v >> 32) as u32);
            self.put(32, v as u32);
        } else {
            self.put(n, v as u32);
        }
    }

    #[inline]
    pub fn put_bit(&mut self, bit: bool) {
        self.put(1, bit as u32);
    }

    /// ue(v).
    pub fn ue(&mut self, v: u32) {
        let x = v as u64 + 1;
        let len = 64 - x.leading_zeros(); // bits in x
        self.put64(len - 1, 0);
        self.put64(len, x);
    }

    /// se(v).
    pub fn se(&mut self, v: i32) {
        let v = v as i64;
        let k = if v > 0 { 2 * v - 1 } else { -2 * v };
        self.ue(k as u32);
    }

    /// Pad with 1 bits to a byte boundary (cabac_alignment_one_bit).
    pub fn align_ones(&mut self) {
        if self.nacc != 0 {
            let k = 8 - self.nacc;
            self.put(k, (1 << k) - 1);
        }
    }

    /// Pad with 0 bits to a byte boundary.
    pub fn align_zeros(&mut self) {
        if self.nacc != 0 {
            self.put(8 - self.nacc, 0);
        }
    }

    /// Append whole bytes (at any current alignment).
    pub fn put_bytes(&mut self, bytes: &[u8]) {
        if self.nacc == 0 {
            self.buf.extend_from_slice(bytes);
        } else {
            for &b in bytes {
                self.put(8, b as u32);
            }
        }
    }

    /// Append bits `[start, end)` of `src` (Python: `bits[start:end]` of a
    /// `to_bits` list).
    pub fn put_bits_from(&mut self, src: &[u8], start: usize, end: usize) {
        debug_assert!(start <= end && end <= src.len() * 8);
        let mut pos = start;
        // up to the next byte boundary of the source
        while pos < end && pos & 7 != 0 {
            self.put(1, ((src[pos >> 3] >> (7 - (pos & 7))) & 1) as u32);
            pos += 1;
        }
        // whole bytes
        let whole = (end - pos) / 8;
        if whole > 0 {
            self.put_bytes(&src[pos >> 3..(pos >> 3) + whole]);
            pos += whole * 8;
        }
        while pos < end {
            self.put(1, ((src[pos >> 3] >> (7 - (pos & 7))) & 1) as u32);
            pos += 1;
        }
    }

    /// Append one bit per byte of `bits` (0/1), like extending a Python bit list.
    pub fn put_bit_list(&mut self, bits: &[u8]) {
        for &b in bits {
            self.put(1, (b & 1) as u32);
        }
    }

    /// The written bits, zero-padded to a whole number of bytes.
    pub fn finish(mut self) -> Vec<u8> {
        self.align_zeros();
        self.buf
    }

    /// The bytes completed so far (excluding a partial last byte).
    pub fn bytes_so_far(&self) -> &[u8] {
        &self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementations written as literal translations of the
    /// regexes, used to cross-check the fast versions.
    fn unescape_ref(b: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if i + 2 < b.len() && b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 3 {
                out.extend_from_slice(&[0, 0]);
                i += 3;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        out
    }

    fn escape_ref(b: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < b.len() {
            if i + 1 < b.len() && b[i] == 0 && b[i + 1] == 0 && (i + 2 == b.len() || b[i + 2] <= 3) {
                out.extend_from_slice(&[0, 0, 3]);
                i += 2;
            } else {
                out.push(b[i]);
                i += 1;
            }
        }
        out
    }

    #[test]
    fn escape_cases() {
        let cases: &[(&[u8], &[u8])] = &[
            (&[], &[]),
            (&[0, 0], &[0, 0, 3]),
            (&[0, 0, 0], &[0, 0, 3, 0]),
            (&[0, 0, 0, 0], &[0, 0, 3, 0, 0, 3]),
            (&[0, 0, 0, 0, 0], &[0, 0, 3, 0, 0, 3, 0]),
            (&[0, 0, 1], &[0, 0, 3, 1]),
            (&[0, 0, 3], &[0, 0, 3, 3]),
            (&[0, 0, 4], &[0, 0, 4]),
            (&[1, 0, 0], &[1, 0, 0, 3]),
            (&[0, 0, 2, 0, 0], &[0, 0, 3, 2, 0, 0, 3]),
            (&[0, 1, 0, 0, 1], &[0, 1, 0, 0, 3, 1]),
        ];
        for (inp, want) in cases {
            assert_eq!(escape(inp), *want, "escape {inp:?}");
            assert_eq!(escape_ref(inp), *want);
        }
    }

    #[test]
    fn unescape_cases() {
        let cases: &[(&[u8], &[u8])] = &[
            (&[], &[]),
            (&[0, 0, 3], &[0, 0]),
            (&[0, 0, 3, 0, 0, 3], &[0, 0, 0, 0]),
            (&[0, 0, 0, 3], &[0, 0, 0]),
            (&[0, 0, 3, 3], &[0, 0, 3]),
            (&[0, 0, 3, 0, 3], &[0, 0, 0, 3]),
            (&[0, 3], &[0, 3]),
            (&[3, 0, 0, 3, 0, 0, 3, 3], &[3, 0, 0, 0, 0, 3]),
        ];
        for (inp, want) in cases {
            assert_eq!(unescape(inp), *want, "unescape {inp:?}");
            assert_eq!(unescape_ref(inp), *want);
        }
    }

    #[test]
    fn escape_unescape_vs_reference_fuzz() {
        let mut s = 12345u32;
        for _ in 0..20000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            let len = (s >> 24) as usize % 40;
            let mut v = Vec::new();
            for _ in 0..len {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push([0, 0, 0, 1, 2, 3, 4][(s >> 20) as usize % 7]);
            }
            assert_eq!(escape(&v), escape_ref(&v), "{v:?}");
            assert_eq!(unescape(&v), unescape_ref(&v), "{v:?}");
            // escaping then unescaping restores the RBSP
            assert_eq!(unescape(&escape(&v)), v, "{v:?}");
        }
    }

    #[test]
    fn set_bits_cases() {
        let mut b = [0u8; 4];
        set_bits(&mut b, 3, 13, 0x1abc);
        assert_eq!(b, [0b0001_1010, 0b1011_1100, 0, 0]);
        let mut b = [0xffu8; 3];
        set_bits(&mut b, 4, 8, 0);
        assert_eq!(b, [0xf0, 0x0f, 0xff]);
        // only the low n bits of value are used
        let mut b = [0u8; 2];
        set_bits(&mut b, 0, 4, 0xfa);
        assert_eq!(b, [0xa0, 0]);
    }

    #[test]
    fn reader_basics() {
        let d = [0b1010_0110, 0b1100_0000, 0xff, 0xff, 0xff, 0xff, 0xff];
        let mut r = BitReader::new(&d);
        assert_eq!(r.u(1).unwrap(), 1);
        assert_eq!(r.u(3).unwrap(), 0b010);
        assert_eq!(r.u(0).unwrap(), 0);
        assert_eq!(r.u(6).unwrap(), 0b011011);
        assert_eq!(r.p, 10);
        let mut r = BitReader::new(&d);
        r.p = 3;
        let want = to_bits(&d)[3..35].iter().fold(0u32, |a, &b| (a << 1) | b as u32);
        assert_eq!(r.u(32).unwrap(), want);
        assert_eq!(r.p, 35);
        let mut r = BitReader::new(&[0xff]);
        assert!(r.u(9).is_err());
        assert_eq!(r.p, 0);
        assert!(BitReader::new(&[0]).ue().is_err());
    }

    #[test]
    fn ue_se_values() {
        // 1 | 010 | 011 | 00100 | 00101 | 00110 | 00111
        let mut w = BitWriter::new();
        for v in 0..7 {
            w.ue(v);
        }
        assert_eq!(w.len_bits(), 1 + 3 + 3 + 5 * 4);
        let bytes = w.finish();
        assert_eq!(bytes, vec![0b1010_0110, 0b0100_0010, 0b1001_1000, 0b1110_0000]);
        let mut r = BitReader::new(&bytes);
        for v in 0..7 {
            assert_eq!(r.ue().unwrap(), v);
        }
        // se mapping: 0, 1, -1, 2, -2
        let mut r = BitReader::new(&[0b1010_0110, 0b0100_0010, 0b1001_1000]);
        let got: Vec<i32> = (0..5).map(|_| r.se().unwrap()).collect();
        assert_eq!(got, vec![0, 1, -1, 2, -2]);
    }

    #[test]
    fn writer_reader_roundtrip() {
        let mut s = 99u32;
        let mut w = BitWriter::new();
        let mut want: Vec<(u32, i64)> = Vec::new(); // (kind, value); kind 0..=32 = u(n), 100 = ue, 101 = se
        for _ in 0..5000 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            match (s >> 28) % 3 {
                0 => {
                    let n = (s >> 8) % 33;
                    let v = if n == 32 { s.wrapping_mul(2654435761) } else { s.wrapping_mul(2654435761) & ((1u32 << n) - 1) };
                    w.put(n, v);
                    want.push((n, v as i64));
                }
                1 => {
                    let v = (s >> 3) % 100000;
                    w.ue(v);
                    want.push((100, v as i64));
                }
                _ => {
                    let v = ((s >> 3) % 200001) as i32 - 100000;
                    w.se(v);
                    want.push((101, v as i64));
                }
            }
        }
        // extremes
        w.ue(u32::MAX - 1);
        w.se(i32::MIN + 1);
        w.se(i32::MAX);
        let nbits = w.len_bits();
        let bytes = w.finish();
        assert_eq!(bytes.len(), nbits.div_ceil(8));
        let mut r = BitReader::new(&bytes);
        for (k, v) in want {
            match k {
                100 => assert_eq!(r.ue().unwrap() as i64, v),
                101 => assert_eq!(r.se().unwrap() as i64, v),
                n => assert_eq!(r.u(n).unwrap() as i64, v),
            }
        }
        assert_eq!(r.ue().unwrap(), u32::MAX - 1);
        assert_eq!(r.se().unwrap(), i32::MIN + 1);
        assert_eq!(r.se().unwrap(), i32::MAX);
        assert_eq!(r.p, nbits);
    }

    #[test]
    fn writer_alignment_and_copy() {
        let mut w = BitWriter::new();
        w.put(3, 0b101);
        assert!(!w.is_aligned());
        w.align_ones();
        assert!(w.is_aligned());
        assert_eq!(w.len_bits(), 8);
        w.put(1, 1);
        w.align_zeros();
        w.put_bytes(&[0xde, 0xad]);
        w.put(2, 3);
        w.put_bytes(&[0xff]);
        assert_eq!(w.clone().finish(), vec![0b1011_1111, 0x80, 0xde, 0xad, 0xff, 0xc0]);
        // copying a bit range equals slicing a to_bits list
        let src = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc];
        let bits = to_bits(&src);
        for (a, b) in [(0, 0), (0, 48), (3, 45), (5, 6), (8, 40), (7, 33), (13, 14), (16, 16)] {
            let mut w = BitWriter::new();
            w.put(3, 5);
            w.put_bits_from(&src, a, b);
            let mut expect = to_bits(&[0b1010_0000])[..3].to_vec();
            expect.extend_from_slice(&bits[a..b]);
            let mut w2 = BitWriter::new();
            w2.put_bit_list(&expect);
            assert_eq!(w.len_bits(), expect.len());
            assert_eq!(w.finish(), w2.finish(), "range {a}..{b}");
        }
    }

    #[test]
    fn to_from_bits() {
        let b = [0xa5u8, 0x0f, 0xff];
        assert_eq!(from_bits(&to_bits(&b)), b);
        assert_eq!(to_bits(&[0x81]), vec![1, 0, 0, 0, 0, 0, 0, 1]);
        // Python quirk: a partial last byte is right-aligned
        assert_eq!(from_bits(&[1, 0]), vec![2]);
        assert_eq!(from_bits(&[1, 1, 1, 1, 1, 1, 1, 1, 1]), vec![0xff, 1]);
    }
}

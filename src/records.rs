//! Macroblock records: what the CABAC codec parses into and encodes from.
//!
//! These mirror the Python dicts of `h264cabac.SliceCodec` one to one,
//! including which keys exist (`Option`), so that the canonical JSON form
//! (`Mb::canonical`) equals `port/canon.py`'s form of the Python record.
//! The shared tables (B_TYPES, B_SUB, P_SUB, partition shapes) live here
//! because the codec, motion derivation and re-expression all use them.

use serde_json::{Map, Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    I4,
    I8,
    I16,
    Pcm,
    P,
    B,
    Skip,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::I4 => "I4",
            Kind::I8 => "I8",
            Kind::I16 => "I16",
            Kind::Pcm => "PCM",
            Kind::P => "P",
            Kind::B => "B",
            Kind::Skip => "skip",
        }
    }

    pub fn is_intra(self) -> bool {
        matches!(self, Kind::I4 | Kind::I8 | Kind::I16 | Kind::Pcm)
    }
}

/// Macroblock partitioning. `Direct` is B_Direct_16x16 (Python 'D').
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    P16x16,
    P16x8,
    P8x16,
    P8x8,
    Direct,
}

impl Part {
    pub fn as_str(self) -> &'static str {
        match self {
            Part::P16x16 => "16x16",
            Part::P16x8 => "16x8",
            Part::P8x16 => "8x16",
            Part::P8x8 => "8x8",
            Part::Direct => "D",
        }
    }
}

/// Sub-macroblock shape. `Direct` only occurs in B slices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubShape {
    S8x8,
    S8x4,
    S4x8,
    S4x4,
    Direct,
}

/// Prediction direction of a partition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pred {
    L0,
    L1,
    Bi,
}

impl Pred {
    /// Python `uses(pred, lst)`.
    pub fn uses(self, lst: usize) -> bool {
        match self {
            Pred::Bi => true,
            Pred::L0 => lst == 0,
            Pred::L1 => lst == 1,
        }
    }
}

use Part::*;
use Pred::*;

/// B macroblock types (Table 7-14): (partitioning, predictions per partition).
/// Index 0 is B_Direct_16x16, 22 is B_8x8.
pub const B_TYPES: [(Part, &[Pred]); 23] = [
    (Direct, &[]),
    (P16x16, &[L0]),
    (P16x16, &[L1]),
    (P16x16, &[Bi]),
    (P16x8, &[L0, L0]),
    (P8x16, &[L0, L0]),
    (P16x8, &[L1, L1]),
    (P8x16, &[L1, L1]),
    (P16x8, &[L0, L1]),
    (P8x16, &[L0, L1]),
    (P16x8, &[L1, L0]),
    (P8x16, &[L1, L0]),
    (P16x8, &[L0, Bi]),
    (P8x16, &[L0, Bi]),
    (P16x8, &[L1, Bi]),
    (P8x16, &[L1, Bi]),
    (P16x8, &[Bi, L0]),
    (P8x16, &[Bi, L0]),
    (P16x8, &[Bi, L1]),
    (P8x16, &[Bi, L1]),
    (P16x8, &[Bi, Bi]),
    (P8x16, &[Bi, Bi]),
    (P8x8, &[]),
];

/// B sub-macroblock types (Table 7-18): (shape, prediction). Index 0 is direct.
pub const B_SUB: [(SubShape, Pred); 13] = [
    (SubShape::Direct, Bi), // prediction unused for direct
    (SubShape::S8x8, L0),
    (SubShape::S8x8, L1),
    (SubShape::S8x8, Bi),
    (SubShape::S8x4, L0),
    (SubShape::S4x8, L0),
    (SubShape::S8x4, L1),
    (SubShape::S4x8, L1),
    (SubShape::S8x4, Bi),
    (SubShape::S4x8, Bi),
    (SubShape::S4x4, L0),
    (SubShape::S4x4, L1),
    (SubShape::S4x4, Bi),
];

/// P sub-macroblock types (Table 7-17).
pub const P_SUB: [SubShape; 4] = [SubShape::S8x8, SubShape::S8x4, SubShape::S4x8, SubShape::S4x4];

/// P macroblock partitionings in mb_type order.
pub const P_TYPES: [Part; 4] = [P16x16, P16x8, P8x16, P8x8];

/// B_TYPES index of a (partitioning, predictions) pair (Python `B_TYPES.index`).
pub fn b_type_index(part: Part, preds: &[Pred]) -> Option<u8> {
    B_TYPES.iter().position(|(p, pr)| *p == part && *pr == preds).map(|i| i as u8)
}

/// 4x4-block coordinates (within an 8x8) of each sub-partition (Python `sub_parts`).
pub fn sub_parts(shape: SubShape) -> &'static [&'static [(usize, usize)]] {
    match shape {
        SubShape::S8x8 => &[&[(0, 0), (1, 0), (0, 1), (1, 1)]],
        SubShape::S8x4 => &[&[(0, 0), (1, 0)], &[(0, 1), (1, 1)]],
        SubShape::S4x8 => &[&[(0, 0), (0, 1)], &[(1, 0), (1, 1)]],
        SubShape::S4x4 => &[&[(0, 0)], &[(1, 0)], &[(0, 1)], &[(1, 1)]],
        SubShape::Direct => panic!("sub_parts of a direct sub-macroblock"),
    }
}

/// 4x4-block coordinates (within the MB) of each partition, in Python's
/// order (row-major per partition). Only 16x16, 16x8, 8x16.
pub fn mb_parts(part: Part) -> &'static [&'static [(usize, usize)]] {
    const P16: [(usize, usize); 16] = [
        (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3),
    ];
    const P16X8_0: [(usize, usize); 8] = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)];
    const P16X8_1: [(usize, usize); 8] = [(0, 2), (1, 2), (2, 2), (3, 2), (0, 3), (1, 3), (2, 3), (3, 3)];
    const P8X16_0: [(usize, usize); 8] = [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)];
    const P8X16_1: [(usize, usize); 8] = [(2, 0), (3, 0), (2, 1), (3, 1), (2, 2), (3, 2), (2, 3), (3, 3)];
    match part {
        P16x16 => &[&P16],
        P16x8 => &[&P16X8_0, &P16X8_1],
        P8x16 => &[&P8X16_0, &P8X16_1],
        _ => panic!("mb_parts({part:?})"),
    }
}

/// luma4x4BlkIdx -> (x, y) in 4x4 units.
pub const BLK_XY: [(usize, usize); 16] = {
    let mut t = [(0usize, 0usize); 16];
    let mut i = 0;
    while i < 16 {
        t[i] = ((i % 4) % 2 + 2 * ((i / 4) % 2), (i % 4) / 2 + 2 * (i / 8));
        i += 1;
    }
    t
};

pub type Mv = [i32; 2];

/// Motion vector differences. Python stores `mvds` as `[2][parts]` of
/// (x, y) or None for 16x16 / 16x8 / 8x16, and as `[2][4][subparts]` of
/// (x, y) for 8x8 (an unused list or sub-block is an empty list after
/// decoding; re-expression fills unused entries with (0, 0)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mvds {
    Parts([Vec<Option<Mv>>; 2]),
    Sub([[Vec<Mv>; 4]; 2]),
}

/// Residual levels in scan order, one slot per Python `coef` key:
/// ('LDC',), ('L8', i8), ('L4', idx), ('CDC', comp), ('CAC', comp, i).
/// A slot is `Some` exactly when the Python dict has that key.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct Coefs {
    pub ldc: Option<Vec<i32>>,
    pub l8: [Option<Vec<i32>>; 4],
    pub l4: [Option<Vec<i32>>; 16],
    pub cdc: [Option<Vec<i32>>; 2],
    pub cac: [[Option<Vec<i32>>; 4]; 2],
}

/// One macroblock record. Field names follow the Python keys; `Option`
/// fields are `None` where the Python dict lacks the key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mb {
    pub x: u16,
    pub y: u16,
    pub skip: bool,
    pub kind: Kind,
    pub intra: bool,
    pub pcm: bool,
    pub i16: bool,
    pub direct16: bool,
    pub t8: bool,
    pub cbp: u8,
    pub qpd: i32,
    /// coded_block_flag of the luma DC block (`dcL`).
    pub dc_l: bool,
    /// coded_block_flag of the chroma DC blocks (`dcC`).
    pub dc_c: [bool; 2],
    pub cmode: u8,
    /// I4 / I8: rem_intra_pred_mode per block, -1 when the predicted mode was used.
    pub ipred: Vec<i8>,
    /// I4 / I8: the intra prediction mode per block (16 or 4 entries).
    pub imodes: Vec<u8>,
    /// I16 only (`i16pred`, `i16cbpC`, `i16cbpL`).
    pub i16pred: u8,
    pub i16cbp_c: u8,
    pub i16cbp_l: u8,
    pub part: Option<Part>,
    /// B inter MBs: index into B_TYPES.
    pub btype: Option<u8>,
    /// 8x8 MBs: sub_mb_type per quadrant (index into P_SUB or B_SUB).
    pub sub: Option<[u8; 4]>,
    /// ref_idx per list and partition; `None` where the list is unused.
    /// Re-expressed records may carry `Some(-1)` there (Python stores the
    /// motion field's -1), which the encoder never reads.
    pub refs: Option<[Vec<Option<i8>>; 2]>,
    pub mvds: Option<Mvds>,
    pub coef: Option<Box<Coefs>>,
    /// I_PCM samples: 256 Y, then 64 U, then 64 V.
    pub samples: Option<Box<[u8; 384]>>,
}

impl Mb {
    /// A blank record at (x, y), as the decoder starts one.
    pub fn new(x: u16, y: u16) -> Mb {
        Mb {
            x,
            y,
            skip: false,
            kind: Kind::Skip,
            intra: false,
            pcm: false,
            i16: false,
            direct16: false,
            t8: false,
            cbp: 0,
            qpd: 0,
            dc_l: false,
            dc_c: [false, false],
            cmode: 0,
            ipred: Vec::new(),
            imodes: Vec::new(),
            i16pred: 0,
            i16cbp_c: 0,
            i16cbp_l: 0,
            part: None,
            btype: None,
            sub: None,
            refs: None,
            mvds: None,
            coef: None,
            samples: None,
        }
    }

    /// Canonical JSON form, equal to `port/canon.py:canon_mb` of the
    /// corresponding Python record. Key order does not matter
    /// (serde_json's map is sorted).
    pub fn canonical(&self) -> Value {
        let b = |v: bool| json!(v as i32);
        let mut m = Map::new();
        m.insert("x".into(), json!(self.x));
        m.insert("y".into(), json!(self.y));
        m.insert("skip".into(), b(self.skip));
        m.insert("kind".into(), json!(self.kind.as_str()));
        m.insert("intra".into(), b(self.intra));
        m.insert("pcm".into(), b(self.pcm));
        m.insert("i16".into(), b(self.i16));
        m.insert("direct16".into(), b(self.direct16));
        m.insert("t8".into(), b(self.t8));
        m.insert("cbp".into(), json!(self.cbp));
        m.insert("qpd".into(), json!(self.qpd));
        m.insert("dcL".into(), b(self.dc_l));
        m.insert("dcC".into(), json!([self.dc_c[0] as i32, self.dc_c[1] as i32]));
        m.insert("cmode".into(), json!(self.cmode));
        if matches!(self.kind, Kind::I4 | Kind::I8) {
            m.insert("ipred".into(), json!(self.ipred));
            m.insert("imodes".into(), json!(self.imodes));
        }
        if self.kind == Kind::I16 {
            m.insert("i16pred".into(), json!(self.i16pred));
            m.insert("i16cbpC".into(), json!(self.i16cbp_c));
            m.insert("i16cbpL".into(), json!(self.i16cbp_l));
        }
        if let Some(p) = self.part {
            m.insert("part".into(), json!(p.as_str()));
        }
        if let Some(t) = self.btype {
            m.insert("btype".into(), json!(t));
        }
        if let Some(s) = self.sub {
            m.insert("sub".into(), json!(s));
        }
        if let Some(r) = &self.refs {
            m.insert("refs".into(), json!([r[0], r[1]]));
        }
        if let Some(mv) = &self.mvds {
            let v = match mv {
                Mvds::Parts(l) => json!([l[0], l[1]]),
                Mvds::Sub(l) => json!([l[0], l[1]]),
            };
            m.insert("mvds".into(), v);
        }
        if let Some(c) = &self.coef {
            let mut cm = Map::new();
            if let Some(v) = &c.ldc {
                cm.insert("LDC".into(), json!(v));
            }
            for (i, v) in c.l8.iter().enumerate() {
                if let Some(v) = v {
                    cm.insert(format!("L8:{i}"), json!(v));
                }
            }
            for (i, v) in c.l4.iter().enumerate() {
                if let Some(v) = v {
                    cm.insert(format!("L4:{i}"), json!(v));
                }
            }
            for (i, v) in c.cdc.iter().enumerate() {
                if let Some(v) = v {
                    cm.insert(format!("CDC:{i}"), json!(v));
                }
            }
            for (comp, blocks) in c.cac.iter().enumerate() {
                for (i, v) in blocks.iter().enumerate() {
                    if let Some(v) = v {
                        cm.insert(format!("CAC:{comp}:{i}"), json!(v));
                    }
                }
            }
            m.insert("coef".into(), Value::Object(cm));
        }
        if let Some(s) = &self.samples {
            m.insert("samples".into(), json!(s.to_vec()));
        }
        Value::Object(m)
    }
}

/// Canonical JSON text of a value: sorted keys, no whitespace. Equal to
/// Python's `json.dumps(v, sort_keys=True, separators=(',', ':'))` for the
/// integers, strings, lists and nulls used here.
pub fn canonical_text(v: &Value) -> String {
    serde_json::to_string(v).expect("JSON values always serialise")
}

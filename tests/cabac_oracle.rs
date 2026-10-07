//! CABAC codec against the cached Python oracle (`work/port/oracle/records_<clip>.py.jsonl`,
//! made by `python -m port.oracle records <clip>.h264`). Skips when the clips or the cached
//! output are missing.

use std::path::Path;

use frame_dedup::picture::{decode_picture, hex, sha256, try_encode_picture};
use frame_dedup::records::{Mb, canonical_text};
use serde_json::Value;

const ROOT: &str = r"C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port";

#[test]
fn records_match_python_and_canadian_clips_round_trip() {
    for clip in ["sample", "foliage", "smoke"] {
        let es_path = format!(r"{ROOT}\clips\{clip}.h264");
        let cache = format!(r"{ROOT}\oracle\records_{clip}.py.jsonl");
        if !Path::new(&es_path).exists() || !Path::new(&cache).exists() {
            eprintln!("skipping {clip}: no clip or oracle output");
            continue;
        }
        let es = std::fs::read(&es_path).unwrap();
        let (spss, ppss, pics) = frame_dedup::stream::load(Path::new(&es_path)).unwrap();
        let want: Vec<Value> =
            std::fs::read_to_string(&cache).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(want.len(), pics.len());
        for (pic, w) in pics.iter().zip(&want) {
            let mut d = decode_picture(&es, &spss, &ppss, pic).unwrap();
            let list = Value::Array(d.mbs.iter().map(Mb::canonical).collect());
            assert_eq!(w["digest"], hex(&sha256(canonical_text(&list).as_bytes())), "{clip} picture {}", pic.dec);
            // an unchanged picture encodes back to its own NAL
            let sl = &pic.slices[0];
            let mut mbs = std::mem::take(&mut d.mbs);
            let nal = try_encode_picture(&d, &spss, &ppss, &mut mbs).unwrap();
            assert_eq!(nal, &es[sl.offset..sl.offset + sl.size], "{clip} picture {} re-encode", pic.dec);
        }
    }
}

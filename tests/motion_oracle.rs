//! Motion derivation, re-expression (target mode) and the re-encoded NALs against the cached Python
//! oracle (`work/port/oracle/motion_t_<clip>.py.jsonl`, made by
//! `python -m port.oracle motion <clip>.h264 --target --workers 4`). Skips when the clips or the cached
//! output are missing.

use std::path::Path;

use frame_dedup::reexpress::motion_lines;
use serde_json::Value;

const ROOT: &str = r"C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port";

#[test]
fn motion_and_target_mode_match_python() {
    for clip in ["sample", "foliage", "smoke", "mmco"] {
        let es_path = format!(r"{ROOT}\clips\{clip}.h264");
        let cache = format!(r"{ROOT}\oracle\motion_t_{clip}.py.jsonl");
        if !Path::new(&es_path).exists() || !Path::new(&cache).exists() {
            eprintln!("skipping {clip}: no clip or oracle output");
            continue;
        }
        let want: Vec<Value> =
            std::fs::read_to_string(&cache).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let mut got: Vec<Value> = Vec::new();
        motion_lines(&es_path, 0, None, 4, true, &mut |l| got.push(serde_json::from_str(&l).unwrap())).unwrap();
        assert_eq!(got.len(), want.len(), "{clip}: number of lines");
        for (g, w) in got.iter().zip(&want) {
            assert_eq!(g, w, "{clip}: picture {}", w["dec"]);
        }
    }
}

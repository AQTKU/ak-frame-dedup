//! `patch_pictures` against calls recorded from Python's repair tasks
//! (`work/port/oracle/patch_*_corpus*.json`, made by `python -m port.oracle patch-record`; the corpora
//! refer to the stream files of their scenario). Skips corpora that are missing.

use std::path::Path;

use frame_dedup::patch::replay_corpus;

const ORACLE: &str = r"C:\Users\aquat\Documents\GitHub\ak-frame-dedup\work\port\oracle";

#[test]
fn patch_pictures_replays_python_calls() {
    let mut total = 0;
    for name in ["patch_clip_corpus", "patch_sample_corpus", "patch_sample_corpus2", "patch_smoke_corpus", "patch_synth_sample_corpus", "patch_full_corpus"] {
        let path = format!(r"{ORACLE}\{name}.json");
        if !Path::new(&path).exists() {
            eprintln!("skipping {name}: no corpus");
            continue;
        }
        let corpus: serde_json::Value = serde_json::from_reader(std::fs::File::open(&path).unwrap()).unwrap();
        let explicit = corpus["explicit"].as_bool() == Some(true);
        let exists = |k: &str| corpus[k].as_str().is_some_and(|p| Path::new(p).exists());
        if !exists("new_es") || (explicit && !exists("orig_es")) {
            eprintln!("skipping {name}: stream files missing");
            continue;
        }
        let mut lines = Vec::new();
        let sum = replay_corpus(&path, &[], &mut |l| lines.push(l)).unwrap();
        assert_eq!(sum.problems, 0, "{name}:\n{}", lines.join("\n"));
        total += sum.calls;
    }
    eprintln!("{total} recorded patch_pictures calls replayed identically");
}

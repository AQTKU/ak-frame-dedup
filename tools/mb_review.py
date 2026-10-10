"""Run frame-dedup with --verify --mb-report and build an HTML page of the
frames that differ, so the differing macroblocks can be checked by eye.

    python tools/mb_review.py <input.mkv> [--out-dir DIR] [--threshold 3] [--max-frames 150]
                              [--exe PATH] [--workers N] [--mb-json mb.json --output out.mkv]

Without --mb-json it runs the tool (below-normal priority) as
    frame-dedup <input> -o <output> --verify --mb-report <out-dir>/mb.json --mb-threshold N
With --mb-json and --output it reuses an earlier run instead.

Frames are ranked by their worst visible macroblock (max |difference| over
Y, U and V) and the worst --max-frames are drawn. Both files are decoded once
each by ffmpeg, which hands over only the selected frames, so seeking is exact.
For each frame the page shows the full frame (original / output, switchable,
with the listed macroblocks outlined and a difference overlay) and lossless
4x crops of its worst macroblocks. Writes <out-dir>/index.html.

All images are PNG, so what the page shows is exactly what was decoded.
"""
import argparse
import html
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ZOOM = 4           # crop enlargement
CROP_MBS = 3       # crop size in macroblocks (the hit in the middle)
CROPS_PER_FRAME = 6
BOXES_MAX = 300    # outline blocks by default up to this many per frame
FFMPEG_THREADS = 4
# heat colours for |difference|: (lower bound, RGBA)
HEAT = [(1, (40, 90, 255, 150)), (3, (255, 230, 0, 210)), (6, (255, 140, 0, 235)),
        (10, (255, 30, 30, 255)), (20, (255, 0, 255, 255))]
STROKE = [(0, "#ffe600"), (6, "#ff8c00"), (10, "#ff1e1e"), (20, "#ff00ff")]


def below_normal():
    return {"creationflags": 0x00004000} if os.name == "nt" else {"preexec_fn": lambda: os.nice(10)}


def run_tool(a, mb_json, output):
    cmd = [a.exe, str(a.input), "-o", str(output), "--verify", "--mb-report", str(mb_json),
           "--mb-threshold", str(a.threshold)]
    if a.workers:
        cmd += ["--workers", str(a.workers)]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, **below_normal())
    # 4 = verify found differences where there should be none: still worth drawing
    if r.returncode not in (0, 4):
        sys.exit(f"frame-dedup exited with {r.returncode}")


def select_expr(ns):
    """`select` expression true on the frames in `ns`, as a balanced tree:
    ffmpeg's parser fails on a long flat chain of `+`."""
    if len(ns) == 1:
        return f"eq(n\\,{ns[0]})"
    m = len(ns) // 2
    return f"({select_expr(ns[:m])}+{select_expr(ns[m:])})"


def frames(path, wanted, w, h):
    """Decode `path` once and yield (n, Y, U, V) for the display indices in
    `wanted` (ascending), as uint8 planes of the visible picture."""
    wanted = sorted(set(wanted))
    if not wanted:
        return
    cw, ch = (w + 1) // 2, (h + 1) // 2
    size = w * h + 2 * cw * ch
    sel = select_expr(wanted)
    cmd = ["ffmpeg", "-v", "error", "-threads", str(FFMPEG_THREADS), "-i", str(path), "-map", "0:v:0",
           "-vf", f"select='{sel}'", "-fps_mode", "passthrough", "-frames:v", str(len(wanted)),
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, **below_normal())
    try:
        for n in wanted:
            buf = p.stdout.read(size)
            if len(buf) < size:
                raise RuntimeError(f"{path}: ffmpeg stopped before frame {n}")
            a = np.frombuffer(buf, np.uint8)
            yield (n, a[:w * h].reshape(h, w), a[w * h:w * h + cw * ch].reshape(ch, cw),
                   a[w * h + cw * ch:].reshape(ch, cw))
    finally:
        p.stdout.close()
        p.wait()


def rgb(y, u, v):
    """BT.709 limited range to RGB (display only)."""
    h, w = y.shape
    up = lambda c: np.repeat(np.repeat(c, 2, 0), 2, 1)[:h, :w].astype(np.float32) - 128
    yy = (y.astype(np.float32) - 16) * 1.164
    u, v = up(u), up(v)
    out = np.stack([yy + 1.793 * v, yy - 0.213 * u - 0.533 * v, yy + 2.112 * u], -1)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def absdiff(a, b):
    """Per-pixel max |difference| over Y, U and V at luma resolution."""
    h, w = a[0].shape
    d = lambda i: np.abs(a[i].astype(np.int16) - b[i].astype(np.int16))
    up = lambda c: np.repeat(np.repeat(c, 2, 0), 2, 1)[:h, :w]
    return np.maximum(d(0), np.maximum(up(d(1)), up(d(2)))).astype(np.uint8)


def heat(d):
    out = np.zeros(d.shape + (4,), np.uint8)
    for lo, c in HEAT:
        out[d >= lo] = c
    return out


def stroke(v):
    return [c for lo, c in STROKE if v >= lo][-1]


def save_full(arr, path):
    Image.fromarray(arr).save(path, compress_level=3)
    return path.name


def zoomed(arr, path):
    Image.fromarray(arr).resize((arr.shape[1] * ZOOM, arr.shape[0] * ZOOM), Image.NEAREST).save(path)
    return path.name


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--threshold", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--exe", default=str(Path(__file__).resolve().parent.parent / "target/release/frame-dedup.exe"))
    ap.add_argument("--workers", type=int)
    ap.add_argument("--mb-json", type=Path, help="reuse this --mb-report instead of running the tool")
    ap.add_argument("--output", type=Path, help="the deduped MKV (written, or reused with --mb-json)")
    a = ap.parse_args()

    out_dir = a.out_dir or Path("review") / f"{a.input.stem}-mb{a.threshold}"
    img = out_dir / "img"
    img.mkdir(parents=True, exist_ok=True)
    output = a.output or out_dir / f"{a.input.stem}-dedup.mkv"
    if a.mb_json:
        if not a.output:
            sys.exit("--mb-json needs --output (the MKV that report was made from)")
        mb_json = a.mb_json
    else:
        mb_json = out_dir / "mb.json"
        run_tool(a, mb_json, output)

    rep = json.load(open(mb_json, encoding="utf-8"))
    thr = max(a.threshold, rep["threshold"])
    if a.threshold < rep["threshold"]:
        print(f"note: the report lists blocks from {rep['threshold']} up only")
    cl, _, ct, _ = rep["crop"]
    w, h = rep["visible_size_px"]

    # frames with a visible macroblock at or above the threshold, worst first
    listed = []
    for f in rep["frames"]:
        mbs = [m for m in f["mbs"] if m[4] and m[2] >= thr]
        if mbs:
            listed.append((max(m[2] for m in mbs), len(mbs), f, mbs))
    listed.sort(key=lambda t: (-t[0], -t[1], t[2]["out_frame"]))
    chosen = listed[:a.max_frames]
    print(f"{len(rep['frames'])} differing frames in the report, {len(listed)} with a visible block >= {thr}; "
          f"drawing {len(chosen)}", flush=True)

    t0 = time.time()
    orig = {}
    for n, *planes in frames(a.input, [c[2]["in_frame"] for c in chosen], w, h):
        orig[n] = planes
    print(f"decoded original frames in {time.time() - t0:.0f}s", flush=True)

    by_out = {c[2]["out_frame"]: c for c in chosen}
    uses = {}
    for c in chosen:
        uses[c[2]["in_frame"]] = uses.get(c[2]["in_frame"], 0) + 1
    cards = {}
    t0 = time.time()
    for n, *outp in frames(output, list(by_out), w, h):
        worst, count, f, mbs = by_out[n]
        o = orig[f["in_frame"]]
        d = absdiff(o, outp)
        # self-check: what we decoded should agree with what the tool measured
        mine = max(int(d[max(my * 16 - ct, 0):my * 16 - ct + 16, max(mx * 16 - cl, 0):mx * 16 - cl + 16].max())
                   for mx, my, *_ in mbs)
        warn = "" if mine == worst else f"decoded worst {mine}, report says {worst}: frames may be misaligned"
        if warn:
            print(f"warning: out {n}: {warn}", flush=True)
        stem = f"o{n:06d}"
        orig_rgb, out_rgb = rgb(*o), rgb(*outp)
        files = {"orig": save_full(orig_rgb, img / f"{stem}_orig.png"),
                 "out": save_full(out_rgb, img / f"{stem}_out.png")}
        Image.fromarray(heat(d)).save(img / f"{stem}_heat.png")
        files["heat"] = f"{stem}_heat.png"
        crops = []
        cs = CROP_MBS * 16
        for mx, my, mmax, mluma, _ in mbs[:CROPS_PER_FRAME]:
            x0 = min(max(mx * 16 - cl - (cs - 16) // 2, 0), w - cs)
            y0 = min(max(my * 16 - ct - (cs - 16) // 2, 0), h - cs)
            sl = np.s_[y0:y0 + cs, x0:x0 + cs]
            dh = heat(d[sl])  # on black: unchanged pixels are (0, 0, 0)
            tag = f"{stem}_{mx}_{my}"
            crops.append({"mb": (mx, my), "max": mmax, "luma": mluma, "box": (mx * 16 - cl - x0, my * 16 - ct - y0),
                          "orig": zoomed(orig_rgb[sl], img / f"{tag}_orig.png"),
                          "out": zoomed(out_rgb[sl], img / f"{tag}_out.png"),
                          "diff": zoomed(dh[..., :3], img / f"{tag}_diff.png")})
        cards[n] = (worst, count, f, mbs, files, crops, warn)
        uses[f["in_frame"]] -= 1
        if not uses[f["in_frame"]]:
            del orig[f["in_frame"]]
    print(f"decoded output frames and wrote images in {time.time() - t0:.0f}s", flush=True)

    write_page(out_dir, rep, a, thr, listed, [cards[c[2]["out_frame"]] for c in chosen], output, cl, ct, w, h)
    print(f"wrote {out_dir / 'index.html'}")


def write_page(out_dir, rep, a, thr, listed, cards, output, cl, ct, w, h):
    e = html.escape
    hist = {}
    for worst, *_ in listed:
        hist[worst] = hist.get(worst, 0) + 1
    hmax = max(hist.values(), default=1)
    hist_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td><td><span class=bar style='width:{100 * v / hmax:.1f}%;"
        f"background:{stroke(k)}'></span></td></tr>" for k, v in sorted(hist.items()))
    body = []
    for worst, count, f, mbs, files, crops, warn in cards:
        n = f["out_frame"]
        rects = "".join(
            f"<rect x='{mx * 16 - cl}' y='{my * 16 - ct}' width='16' height='16' stroke='{stroke(mm)}'>"
            f"<title>MB ({mx}, {my}): max {mm}, luma {ml}</title></rect>" for mx, my, mm, ml, _ in mbs)
        crop_html = "".join(
            f"<div class=crop><div class=cap>MB ({c['mb'][0]}, {c['mb'][1]}) · max {c['max']} · luma {c['luma']}</div>"
            f"<div class=trio>" + "".join(
                f"<figure><div class=cw><img loading=lazy width={CROP_MBS * 16 * ZOOM} height={CROP_MBS * 16 * ZOOM} src='img/{c[k]}'>"
                f"<i style='left:{c['box'][0] * ZOOM}px;top:{c['box'][1] * ZOOM}px'></i></div>"
                f"<figcaption>{lbl}</figcaption></figure>"
                for k, lbl in (("orig", "original"), ("out", "output"), ("diff", "difference"))) +
            "</div></div>" for c in crops)
        flags = [x for x, on in (("IDR", f["idr"]), ("rebuilt leading B", f["rebuilt_leading_b"]),
                                 ("after dropped ref twin", f["hard_drop_neighbourhood"])) if on]
        body.append(f"""
<section class=card id=f{n} data-worst={worst} data-out={n}>
 <h2>Output frame {n} <small>{f['out_time_s']:.3f}s · input frame {f['in_frame']} ({f['in_time_s']:.3f}s) ·
  {e(f['type'])}{' · ' + e(', '.join(flags)) if flags else ''}</small></h2>
 <p class=stats>worst <b style='color:{stroke(worst)}'>{worst}</b> · {count} visible block{'s' if count != 1 else ''} ≥ {thr}
  · {f['mbs_diff']} blocks differ in all · period {f['in_period']}, POC {f['in_poc']}
  {f"<span class=warn>{e(warn)}</span>" if warn else ''}</p>
 <div class=viewer data-orig='img/{files['orig']}' data-out='img/{files['out']}'>
  <div class=tools><button class=on data-v=orig>Original</button><button data-v=out>Output</button>
   <button data-v=flick>Flicker</button>
   <label><input type=checkbox class=heat> difference</label><label><input type=checkbox class=boxes{' checked' if count <= BOXES_MAX else ''}> blocks</label></div>
  <div class=stage><img class=base loading=lazy width={w} height={h} src='img/{files['orig']}'>
   <img class=heatimg loading=lazy src='img/{files['heat']}' hidden>
   <svg viewBox='0 0 {w} {h}' preserveAspectRatio=none class=boxsvg{'' if count <= BOXES_MAX else ' style=display:none'}>{rects}</svg></div>
 </div>
 <div class=crops>{crop_html}</div>
</section>""")
    legend = "".join(f"<span><i style='background:rgba{c}'></i>≥ {lo}</span>" for lo, c in HEAT)
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Macroblock review</title>
<style>
:root{{--bg:#16181c;--fg:#dde;--mute:#8a8f99;--card:#1e2127;--line:#333842}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:16px}}
h1{{font-size:20px;margin:4px 0}} h2{{font-size:16px;margin:0 0 4px}} small{{color:var(--mute);font-weight:normal}}
.meta{{color:var(--mute);overflow-wrap:anywhere}} .meta code{{color:var(--fg)}}
.top{{display:flex;flex-wrap:wrap;gap:24px;align-items:flex-start}}
table{{border-collapse:collapse}} td{{padding:1px 8px 1px 0;font-variant-numeric:tabular-nums}}
td:last-child{{width:240px}} .bar{{display:block;height:10px;border-radius:2px}}
.legend span{{margin-right:12px;white-space:nowrap}} .legend i{{display:inline-block;width:12px;height:12px;vertical-align:-2px;margin-right:4px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}}
.stats{{margin:0 0 8px}} .warn{{color:#ff6b6b;margin-left:8px}}
.tools{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}}
button{{background:#2a2e36;color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:3px 10px;cursor:pointer}}
button.on{{background:#3d6fd1;border-color:#3d6fd1}} label{{margin-left:6px;user-select:none}}
.stage{{position:relative;line-height:0}} .stage img{{width:100%;height:auto}}
.heatimg,.boxsvg{{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}}
.boxsvg rect{{fill:none;stroke-width:2;vector-effect:non-scaling-stroke}}
.crops{{display:flex;flex-wrap:wrap;gap:14px;margin-top:10px}} .cap{{color:var(--mute);margin-bottom:2px}}
.trio{{display:flex;gap:4px}} figure{{margin:0;text-align:center;color:var(--mute);font-size:12px}}
.cw{{position:relative;line-height:0}} .cw img{{image-rendering:pixelated;width:{CROP_MBS * 16 * ZOOM}px;max-width:28vw;height:auto}}
.cw i{{position:absolute;width:{16 * ZOOM}px;height:{16 * ZOOM}px;outline:1px dashed #fff8;pointer-events:none}}
@media (max-width:700px){{.cw i{{display:none}}}}
.sort{{margin:12px 0}}
</style>
<main>
<h1>Macroblock review: {e(Path(rep['input']).name)}</h1>
<p class=meta>Output <code>{e(str(output))}</code> · frame-dedup {e(rep['version'])} · {e(rep['input_fps']['exact'])} → {e(rep['output_fps']['exact'])} fps ·
 {w}×{h} visible · blocks listed from max |difference| (Y/U/V) ≥ {thr}<br>
 {len(rep['frames'])} differing frames in the report; {len(listed)} have a visible block ≥ {thr};
 the worst {len(cards)} are drawn below.</p>
<div class=top>
 <div><b>Frames by worst block</b><table>{hist_rows}</table></div>
 <div><b>Difference colours</b><div class=legend>{legend}</div>
  <p class=meta>Keys on a frame: <b>o</b> original, <b>p</b> output, <b>f</b> flicker, <b>d</b> difference, <b>b</b> blocks.</p></div>
</div>
<div class=sort>Order: <button class=on data-sort=worst>worst first</button> <button data-sort=out>by time</button></div>
<div id=cards>{''.join(body)}</div>
</main>
<script>
let hovered=null;
document.querySelectorAll('.viewer').forEach(v=>{{
  const base=v.querySelector('.base'),heat=v.querySelector('.heatimg'),svg=v.querySelector('.boxsvg');
  const btn=v.querySelectorAll('button');let timer=null;
  const show=k=>{{clearInterval(timer);btn.forEach(b=>b.classList.toggle('on',b.dataset.v===k));
    if(k==='flick'){{let s=0;timer=setInterval(()=>{{s^=1;base.src=s?v.dataset.out:v.dataset.orig}},500)}}
    else base.src=v.dataset[k]}};
  btn.forEach(b=>b.onclick=()=>show(b.dataset.v));
  const hc=v.querySelector('.heat'),bc=v.querySelector('.boxes');
  hc.onchange=()=>heat.hidden=!hc.checked; bc.onchange=()=>svg.style.display=bc.checked?'':'none';
  v.closest('.card').addEventListener('mouseenter',()=>hovered=v);
  v._keys={{o:()=>show('orig'),p:()=>show('out'),f:()=>show('flick'),
    d:()=>{{hc.checked=!hc.checked;hc.onchange()}},b:()=>{{bc.checked=!bc.checked;bc.onchange()}}}};
}});
document.addEventListener('keydown',ev=>{{if(hovered&&!ev.ctrlKey&&!ev.metaKey&&hovered._keys[ev.key])hovered._keys[ev.key]()}});
document.querySelectorAll('[data-sort]').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('[data-sort]').forEach(x=>x.classList.toggle('on',x===b));
  const c=document.getElementById('cards'),k=b.dataset.sort;
  [...c.children].sort((x,y)=>k==='out'?x.dataset.out-y.dataset.out:(y.dataset.worst-x.dataset.worst)||(x.dataset.out-y.dataset.out))
    .forEach(e=>c.appendChild(e));
}});
</script>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()

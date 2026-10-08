"""Before/after crops of the worst macroblocks for review by eye.

    python tools/review_crops.py <original.mkv> <out_dir> <mb.json> <label=output.mkv>... [--n 12]

Picks the worst visible macroblocks (luma) from <mb.json> (an --mb-report of
the first output; at most one per frame), and writes for each a 96x96 crop
around it, 4x enlarged: the original, every output, and each output's
difference from the original (x8, grey = 0). Writes <out_dir>/index.html.
Output frames are matched by the report's out_frame (all outputs must have
the same drops).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 1920, 1080
CROP, ZOOM, GAIN = 96, 4, 8


def frame(path, n):
    """Luma of display frame n (0-based) as uint8 HxW."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(H, W)


def save(a, p):
    Image.fromarray(a).resize((a.shape[1] * ZOOM, a.shape[0] * ZOOM), Image.NEAREST).save(p)


def main():
    args = sys.argv[1:]
    n = 12
    if "--n" in args:
        i = args.index("--n")
        n = int(args[i + 1])
        del args[i:i + 2]
    orig, out_dir, mbj = Path(args[0]), Path(args[1]), json.load(open(args[2]))
    outs = [a.split("=", 1) for a in args[3:]]
    out_dir.mkdir(parents=True, exist_ok=True)
    cands = []
    for f in mbj["frames"]:
        best = max((m for m in f["mbs"] if m[4]), key=lambda m: m[3], default=None)
        if best:
            cands.append((best[3], f["in_frame"], f["out_frame"], f["type"], best[0], best[1]))
    cands.sort(reverse=True)
    rows = []
    for k, (lum, fin, fout, typ, mx, my) in enumerate(cands[:n]):
        cx = min(max(mx * 16 + 8 - CROP // 2, 0), W - CROP)
        cy = min(max(my * 16 + 8 - CROP // 2, 0), H - CROP)
        o = frame(orig, fin)[cy:cy + CROP, cx:cx + CROP]
        save(o, out_dir / f"{k}_orig.png")
        cells = [f"<td><img src='{k}_orig.png'><br>original</td>"]
        for label, path in outs:
            a = frame(path, fout)[cy:cy + CROP, cx:cx + CROP]
            d = a.astype(int) - o.astype(int)
            save(a, out_dir / f"{k}_{label}.png")
            save(np.clip(128 + GAIN * d, 0, 255).astype(np.uint8), out_dir / f"{k}_{label}_diff.png")
            cells.append(f"<td><img src='{k}_{label}.png'><br>{label} (max |d| {abs(d).max()})</td>")
            cells.append(f"<td><img src='{k}_{label}_diff.png'><br>{label} diff x{GAIN}</td>")
        rows.append(f"<h3>#{k}: output frame {fout} (input {fin}, {typ}), MB ({mx}, {my}), "
                    f"luma off by {lum} in the first output</h3><table><tr>{''.join(cells)}</tr></table>")
    (out_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Worst macroblocks</title>"
        "<style>body{font:14px sans-serif;background:#222;color:#ddd}td{text-align:center;padding:4px}"
        "img{image-rendering:pixelated}</style>" + "".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Python-side dumpers for checking the Rust port against frame_dedup v1.2.0.

Usage: python -m port.oracle <command> [args...]

Each command prints canonical JSON (port/canon.py), usually one object per
line, matching the Rust `frame-dedup dev <command>` of the same name.
"""
import json
import sys

COMMANDS = {}


def command(fn):
    COMMANDS[fn.__name__.replace('_', '-')] = fn
    return fn


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f'usage: python -m port.oracle <{"|".join(sorted(COMMANDS))}> [args...]')
    COMMANDS[argv[0]](*argv[1:])


@command
def framemd5(path, *rest):
    """framemd5 <file.h264> [start end] [--take N] [--size WxH]: MD5 of every decoded frame."""
    import hashlib
    import os
    from frame_dedup.decode import iter_frames
    pos, take, size, it = [], None, (1920, 1088), iter(rest)
    for a in it:
        if a == '--take':
            take = int(next(it))
        elif a == '--size':
            size = tuple(int(x) for x in next(it).split('x'))
        else:
            pos.append(a)
    rng = (int(pos[0]), int(pos[1])) if len(pos) >= 2 else (0, os.path.getsize(path))
    fs = size[0] * size[1] * 3 // 2
    for n, fr in enumerate(iter_frames(path, [rng], fs)):
        if take is not None and n >= take:
            break
        print(n, hashlib.md5(fr).hexdigest())


@command
def mux_cmd(in_mkv, es, out_mkv, num, den):
    """mux-cmd <in.mkv> <es> <out.mkv> <num> <den>: the mkvmerge command line (not run)."""
    import subprocess
    from fractions import Fraction
    from frame_dedup import mux as m
    real_run, seen = subprocess.run, []

    def fake(cmd, *a, **k):
        if cmd[:2] == ['mkvmerge', '-J']:
            return real_run(cmd, *a, **k)
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '', '')
    m.subprocess.run = fake
    m.mux(in_mkv, es, out_mkv, Fraction(int(num), int(den)), log=lambda s: None)
    print(json.dumps(seen[0], separators=(',', ':')))


@command
def mux(in_mkv, es, out_mkv, num, den):
    """mux <in.mkv> <es> <out.mkv> <num> <den>: really mux; prints the log line and the messages."""
    from fractions import Fraction
    from frame_dedup import mux as m
    msgs = m.mux(in_mkv, es, out_mkv, Fraction(int(num), int(den)), log=print)
    print(json.dumps(msgs, separators=(',', ':')))


@command
def duration_check(in_mkv, out_mkv, n_out, num, den):
    """duration-check <in.mkv> <out.mkv> <n_out> <num> <den>."""
    from fractions import Fraction
    from frame_dedup.mux import duration_check as dc
    from .canon import text
    print(text(dc(in_mkv, out_mkv, int(n_out), Fraction(int(num), int(den)))))


@command
def maxdiff(path, mb_w, mb_h, cl, cr, ct, cb, i, j):
    """maxdiff <file.h264> <mb_w> <mb_h> <cl> <cr> <ct> <cb> <i> <j>: mb_maxdiff / visible_maxdiff maps."""
    import os
    from frame_dedup.decode import iter_frames
    from frame_dedup.repair import visible_maxdiff
    from frame_dedup.verify import mb_maxdiff
    from .canon import text
    W, H, i, j = int(mb_w), int(mb_h), int(i), int(j)
    frames = []
    for k, f in enumerate(iter_frames(path, [(0, os.path.getsize(path))], W * H * 384)):
        frames.append(f)
        if k == max(i, j):
            break
    m, ly = mb_maxdiff(frames[i], frames[j], W, H)
    vis = visible_maxdiff(frames[i], frames[j], {'mb_w': W, 'mb_h': H, 'crop': tuple(int(x) for x in (cl, cr, ct, cb))})
    print(text({'mb': m.ravel().tolist(), 'luma': ly.ravel().tolist(), 'visible': vis.ravel().tolist()}))


if __name__ == '__main__':
    main(sys.argv[1:])

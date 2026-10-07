"""Full-resolution decoding of Annex-B byte ranges with ffmpeg."""
import subprocess
import threading


def frame_size(sps):
    return sps['mb_w'] * 16 * sps['mb_h'] * 16 * 3 // 2


def _feed(proc, path, ranges):
    try:
        with open(path, 'rb') as f:
            for a, b in ranges:
                f.seek(a)
                left = b - a
                while left > 0:
                    chunk = f.read(min(left, 1 << 22))
                    if not chunk:
                        break
                    proc.stdin.write(chunk)
                    left -= len(chunk)
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass


def iter_frames(path, ranges, fs, threads=2):
    """Decode the concatenated byte ranges of an Annex-B file (they must start
    at an IDR access unit) and yield uncropped yuv420p frames in output order.
    Raises RuntimeError on decoder errors."""
    cmd = ['ffmpeg', '-v', 'error', '-threads', str(threads), '-apply_cropping', 'none', '-f', 'h264',
           '-i', '-', '-fps_mode', 'passthrough', '-pix_fmt', 'yuv420p', '-f', 'rawvideo', '-']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         bufsize=fs * 2)
    t = threading.Thread(target=_feed, args=(p, path, ranges), daemon=True)
    t.start()
    err = []
    te = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    te.start()
    try:
        while True:
            buf = p.stdout.read(fs)
            if len(buf) < fs:
                break
            yield buf
    finally:
        p.stdout.close()
        p.wait()
        t.join()
        te.join()
    msg = (err[0] if err else b'').decode(errors='replace').strip()
    if p.returncode or msg:
        raise RuntimeError(f'decoder error on {path}: {msg[:500]}')


def decode_pocs(path, pics, a, b, want, fs, threads=2):
    """Decode pictures [a, b) (decode order; a must be an IDR, all in one IDR
    period) and return {poc: frame} for the POCs in `want`."""
    pocs = sorted(pics[d]['poc'] for d in range(a, b))
    out, i = {}, 0
    for i, fr in enumerate(iter_frames(path, [(pics[a]['au_start'], pics[b - 1]['au_end'])], fs, threads)):
        if pocs[i] in want:
            out[pocs[i]] = fr
    missing = set(want) - set(out)
    if missing:
        raise RuntimeError(f'segment decode {a}..{b} did not produce POCs {sorted(missing)[:5]}')
    return out

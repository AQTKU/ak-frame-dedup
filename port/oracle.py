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


@command
def headers(path):
    """SPSs and PPSs on the first line, then one line per picture with its first
    slice header (stream.load: dpb_walk, display order, AU ranges)."""
    from frame_dedup.stream import load
    from .canon import text
    spss, ppss, pics = load(path)
    ppss = {k: dict(v) for k, v in ppss.items()}
    for p in ppss.values():
        p.setdefault('scaling_matrix', 0)  # the Rust struct cannot tell absent from 0
    print(text({'spss': {str(k): v for k, v in spss.items()}, 'ppss': {str(k): v for k, v in ppss.items()}}))
    for p in pics:
        print(text({'dec': p['dec'], 'poc': p['poc'], 'frame_num': p['frame_num'], 'lists': p['lists'],
                    'dpb_after': p['dpb_after'], 'period': p['period'], 'disp': p['disp'],
                    'au_start': p['au_start'], 'au_end': p['au_end'], 'idr': p['idr'],
                    'ref_idc': p['ref_idc'], 'type': p['type'], 'sl': p['slices'][0]}))


@command
def escape(path, n):
    """For the first n slice NALs: unescape / escape results as [length, crc32]."""
    import zlib
    from frame_dedup.h264hdr import escape as esc, iter_nals, unescape as unesc
    from .canon import text
    count = 0
    for s, e, mm in iter_nals(path):
        if mm[s] & 0x1f not in (1, 5):
            continue
        if count >= int(n):
            break
        count += 1
        body = bytes(mm[s + 1:e])
        rbsp = unesc(body)
        enc = esc(rbsp)
        raw_esc = esc(body)
        raw_unesc = unesc(enc)
        print(text({'offset': s, 'unesc': [len(rbsp), zlib.crc32(rbsp)], 'esc': [len(enc), zlib.crc32(enc)],
                    'same': enc == body, 'esc_raw': [len(raw_esc), zlib.crc32(raw_esc)],
                    'unesc_raw': [len(raw_unesc), zlib.crc32(raw_unesc)]}))


@command
def patch_sps(path):
    """Every SPS NAL, patched to 1001/48000 and to 1/48 (hex)."""
    from frame_dedup.h264hdr import iter_nals, patch_sps_timing
    from .canon import text
    for s, e, mm in iter_nals(path):
        if mm[s] & 0x1f != 7:
            continue
        nal = bytes(mm[s:e])
        print(text({'offset': s, 'nal': nal.hex(), 'p1001': patch_sps_timing(nal, 1001, 48000).hex(),
                    'p1_48': patch_sps_timing(nal, 1, 48).hex()}))


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


def _picture_args(rest):
    """[--from N] [--count N] [--full] after the file name."""
    opts, it = {'from': 0, 'count': None, 'full': False}, iter(rest)
    pos = []
    for a in it:
        if a == '--from':
            opts['from'] = int(next(it))
        elif a == '--count':
            opts['count'] = int(next(it))
        elif a == '--full':
            opts['full'] = True
        elif a == '--threads':
            next(it)  # accepted for symmetry with the Rust command
        else:
            pos.append(a)
    return opts, pos


def _open_pictures(path, opts):
    import mmap
    from frame_dedup.stream import load
    spss, ppss, pics = load(path)
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    a = min(opts['from'], len(pics))
    b = len(pics) if opts['count'] is None else min(a + opts['count'], len(pics))
    return mm, spss, ppss, pics, range(a, b)


@command
def records(path, *rest):
    """records <file.h264> [--from N] [--count N] [--full]: per picture, the digest of the
    canonical macroblock records (CABAC parse)."""
    import sys
    import time
    from frame_dedup.roundtrip_test import decode_picture
    from .canon import canon_mbs, digest, text
    opts, _ = _picture_args(rest)
    mm, spss, ppss, pics, rng = _open_pictures(path, opts)
    t0 = time.time()
    for i in rng:
        p = pics[i]
        d = decode_picture(mm, spss, ppss, p)
        c = canon_mbs(d['mbs'])
        line = {'dec': p['dec'], 'type': p['type'], 'n_mbs': len(c), 'digest': digest(c)}
        if opts['full']:
            line['mbs'] = c
        print(text(line), flush=True)
    print(f'records: {len(rng)} pictures in {time.time() - t0:.2f}s', file=sys.stderr)


@command
def roundtrip(path, *rest):
    """roundtrip <file.h264> [--from N] [--count N]: roundtrip_test.check for each picture."""
    import sys
    import time
    from frame_dedup.roundtrip_test import decode_picture, encode_picture
    from .canon import text
    opts, _ = _picture_args(rest)
    mm, spss, ppss, pics, rng = _open_pictures(path, opts)
    ok_n, bad, dec_s, enc_s = 0, [], 0.0, 0.0
    for i in rng:
        p = pics[i]
        t0 = time.time()
        d = decode_picture(mm, spss, ppss, p)
        t1 = time.time()
        sl = d['sl']
        tail = d['bits'][d['end']:]
        stop_ok = d['bits'][d['end'] - 1] == 1 and not any(tail)
        late_stop = (not stop_ok) and d['end'] < len(d['bits']) and d['bits'][d['end']] == 1 and not any(d['bits'][d['end'] + 1:])
        nal2, enc_bits = encode_picture(d)
        t2 = time.time()
        same_bits = enc_bits == d['bits'][sl['data_bit']:d['end']]
        same_nal = nal2 == d['nal']
        dec_s += t1 - t0
        enc_s += t2 - t1
        if stop_ok and same_bits and same_nal:
            ok_n += 1
        else:
            bad.append(p['dec'])
        print(text({'dec': p['dec'], 'type': p['type'], 'nal_len': len(d['nal']), 'stop_bit_ok': stop_ok,
                    'tail_bits': len(tail), 'bits_equal': same_bits, 'nal_equal': same_nal,
                    'late_stop_bit': late_stop}), flush=True)
    print(text({'summary': True, 'pictures': len(rng), 'ok': ok_n, 'failed': len(bad), 'failed_dec': bad}))
    n = max(len(rng), 1)
    print(f'roundtrip: mean decode {dec_s / n * 1e3:.1f} ms, mean encode {enc_s / n * 1e3:.1f} ms', file=sys.stderr)


class SplitMix64:
    M = (1 << 64) - 1

    def __init__(self, seed):
        self.s = seed & self.M

    def next(self):
        self.s = (self.s + 0x9E3779B97F4A7C15) & self.M
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & self.M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & self.M
        return z ^ (z >> 31)


def apply_edit(mbs, t, w, seed):
    """The deterministic edit spec of `encode-edited` (same as picture.rs `apply_edit`)."""
    from frame_dedup.patch import to_explicit
    rng = SplitMix64(seed)
    period = 6 + seed % 29
    counts = [0, 0, 0, 0]
    for i, mb in enumerate(mbs):
        r = rng.next()
        x, y = i % w, i // w
        if r % period == 0:
            mbs[i] = {'skip': 0, 'kind': 'PCM', 'samples': [(x * 31 + y * 17 + j * 7 + seed) & 255 for j in range(384)],
                      'intra': 1, 'pcm': 1, 'i16': 0, 'direct16': 0, 'cbp': 0x2f, 't8': 0, 'cmode': 0, 'qpd': 0}
            counts[0] += 1
        elif t == 'P' and mb['skip'] and (r >> 20) % 6 == 0:
            mbs[i] = to_explicit(mb, 'P', None)
            counts[1] += 1
        elif t == 'B' and mb['skip'] and (r >> 20) % 6 == 0:
            mb.update(skip=0, kind='B', btype=0, part='D', direct16=1, cbp=0, t8=0, qpd=0, coef={})
            counts[1] += 1
        elif not mb['skip'] and mb['cbp'] == 0 and mb['kind'] != 'I16' and (r >> 32) % 4 == 0:
            mb['cbp'] |= 0x10
            mb.setdefault('coef', {})
            mb['qpd'] = ((r >> 40) % 5) - 2
            counts[2] += 1
        elif not mb['skip'] and (mb['cbp'] != 0 or mb['kind'] == 'I16') and (r >> 48) % 9 == 0:
            mb['qpd'] = ((r >> 52) % 7) - 3
            counts[3] += 1
    return counts


@command
def encode_edited(path, *pairs):
    """encode-edited <file.h264> <dec> <seed> [<dec> <seed> ...]: decode picture dec, apply the
    deterministic edit, encode; one line per (dec, seed)."""
    import hashlib
    from frame_dedup.roundtrip_test import decode_picture, encode_picture
    from .canon import text
    mm, spss, ppss, pics, _ = _open_pictures(path, {'from': 0, 'count': None})
    for k in range(0, len(pairs), 2):
        dec, seed = int(pairs[k]), int(pairs[k + 1])
        p = pics[dec]
        d = decode_picture(mm, spss, ppss, p)
        mbs = d['mbs']
        counts = apply_edit(mbs, d['sl']['slice_type'], d['sps']['mb_w'], seed)
        nal, _ = encode_picture(d, mbs)
        print(text({'dec': dec, 'seed': seed, 'type': p['type'], 'len': len(nal),
                    'sha256': hashlib.sha256(nal).hexdigest(), 'pcm': counts[0], 'explicit': counts[1],
                    'empty_dc': counts[2], 'qpd': counts[3]}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])

"""Annex-B stream handling: extraction, parsing, display order and rewriting."""
import mmap
import re
import subprocess

from .h264hdr import escape, parse_stream, patch_sps_timing, set_bits, sps_frame_rate, unescape
from .explicit import rewrite_slice
from .h264mv import dpb_walk
from .system import popen_kwargs


START_CODE = bytes((0, 0, 0, 1))


class Unsupported(Exception):
    """The input is outside what this tool can process."""


def extract_annexb(mkv, out_path):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', mkv, '-map', '0:v:0', '-c', 'copy',
                        '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', out_path], capture_output=True, text=True,
                       **popen_kwargs())
    if r.returncode:
        raise Unsupported(f'could not extract an H.264 video stream: {r.stderr.strip()[:400]}')


def au_offsets(pics, size):
    """Byte range of every access unit; parameter sets and other non-VCL NALs
    in front of a picture belong to it."""
    prev = 0
    for p in pics:
        p['au_start'] = prev
        prev = p['slices'][-1]['offset'] + p['slices'][-1]['size']
        p['au_end'] = prev
    if pics:
        pics[-1]['au_end'] = size


def load(es_path):
    """Parse an Annex-B stream. Returns (spss, ppss, pics) with decode index,
    POC, ref lists, display index and AU byte ranges attached."""
    spss, ppss, pics = parse_stream(es_path)
    if not pics:
        raise Unsupported('no pictures in the video stream')
    check_scope(spss, ppss, pics)
    try:
        dpb_walk(spss, pics)
    except ValueError as e:
        raise Unsupported(str(e))
    period = -1
    for p in pics:
        if p['idr']:
            period += 1
        p['period'] = period
        p['type'] = p['slices'][0]['slice_type']
    order = sorted(range(len(pics)), key=lambda d: (pics[d]['period'], pics[d]['poc']))
    for i, d in enumerate(order):
        pics[d]['disp'] = i
    with open(es_path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
    au_offsets(pics, size)
    return spss, ppss, pics


def check_scope(spss, ppss, pics):
    if not pics[0]['idr']:
        raise Unsupported('stream does not start with an IDR picture')
    if len(spss) != 1:
        raise Unsupported(f'{len(spss)} distinct SPS ids; only one is supported')
    s = spss[0]
    if s['profile_idc'] not in (77, 100) or s['chroma_format_idc'] != 1 or not s['frame_mbs_only']:
        raise Unsupported(f"needs Main/High 4:2:0 progressive H.264 (profile {s['profile_idc']}, "
                          f"chroma {s['chroma_format_idc']}, frame_mbs_only {s['frame_mbs_only']})")
    if s['poc_type'] != 0:
        raise Unsupported('only POC type 0 is supported')
    for p in ppss.values():
        if not p['cabac']:
            raise Unsupported('only CABAC streams are supported')
    multi = sum(1 for p in pics if len(p['slices']) > 1)
    if multi:
        raise Unsupported(f'{multi} pictures have more than one slice; only one slice per picture is supported')
    for p in pics:
        sl = p['slices'][0]
        if sl.get('long_term_ref'):
            raise Unsupported('long-term references are not supported')


def input_rate(spss):
    fr = sps_frame_rate(spss[0])
    if fr is None:
        raise Unsupported('the SPS carries no VUI timing_info; cannot read the input frame rate')
    return fr  # (num, den)


def _nals(buf):
    """(start_code_pos, payload_start, payload_end) for NALs in a byte range."""
    starts = [m.end() for m in re.finditer(rb'\x00\x00\x01', buf)]
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] - 3 if i + 1 < len(starts) else len(buf)
        while e > s and buf[e - 1] == 0:
            e -= 1
        sc = s - 3
        if sc > 0 and buf[sc - 1] == 0:
            sc -= 1
        out.append((sc, s, e))
    return out


def _write_head(o, head, timing, sps_cache):
    """Write the non-VCL NALs in front of a picture, with every SPS's VUI
    timing patched to `timing`. Returns the number of SPSs patched."""
    if timing is None or not head:
        o.write(head)
        return 0
    pos = n = 0
    for sc, s, e in _nals(head):
        if head[s] & 0x1f == 7:
            nal = bytes(head[s:e])
            if nal not in sps_cache:
                sps_cache[nal] = patch_sps_timing(nal, *timing)
            o.write(head[pos:s])
            o.write(sps_cache[nal])
            pos = e
            n += 1
    o.write(head[pos:])
    return n


def write_explicit(es_path, pics, spss, ppss, out_path, entries, hdrs, timing=None):
    """Write the stream planned by explicit.plan_stream: pictures not in
    `entries` are dropped (their non-VCL NALs ride along with the next
    picture), headers that changed are rewritten, and a demoted IDR (anchor
    rebuild "I2") is written in K's slot behind K's and its own non-VCL NALs
    and a recovery point SEI. Returns (rewritten slices, SPSs patched)."""
    from .rebuild import RECOVERY_SEI
    at = {}
    i2_at = {}
    for e, h in zip(entries, hdrs):
        if e.get('demote'):
            i2_at[e['as']['dec']] = (e['pic'], h)
        else:
            at[e['pic']['dec']] = h
    moved = {D['dec'] for D, _ in i2_at.values()}
    n_rw = n_sps = 0
    sps_cache = {}

    def first_sc(mm, sl):
        first = sl['offset'] - 3
        if mm[first - 1] == 0:
            first -= 1
        return first

    with open(es_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out_path, 'wb', buffering=1 << 22) as o:
            for p in pics:
                sl = p['slices'][0]
                first = first_sc(mm, sl)
                end = sl['offset'] + sl['size']
                if p['dec'] in moved:          # written in K's slot
                    o.write(mm[end:p['au_end']])
                    continue
                n_sps += _write_head(o, mm[p['au_start']:first], timing, sps_cache)
                if p['dec'] in i2_at:
                    D, h = i2_at[p['dec']]
                    dsl = D['slices'][0]
                    n_sps += _write_head(o, mm[D['au_start']:first_sc(mm, dsl)], timing, sps_cache)
                    o.write(START_CODE + RECOVERY_SEI)
                    o.write(START_CODE + rewrite_slice(mm[dsl['offset']:dsl['offset'] + dsl['size']], dsl,
                                                                spss, ppss, h))
                    n_rw += 1
                    o.write(mm[end:p['au_end']])
                    continue
                if p['dec'] not in at:         # dropped
                    o.write(mm[end:p['au_end']])
                    continue
                h = at[p['dec']]
                if h['same']:
                    o.write(mm[first:end])
                else:
                    o.write(START_CODE + rewrite_slice(mm[sl['offset']:end], sl, spss, ppss, h))
                    n_rw += 1
                o.write(mm[end:p['au_end']])
        mm.close()
    return n_rw, n_sps


def write_stream(es_path, pics, spss, out_path, drop=(), repl=None, timing=None, hdrs=None, ppss=None):
    """Write a copy of the stream with pictures in `drop` (decode indices)
    removed, frame_num renumbered after dropped reference pictures (up to the
    next IDR), NALs in `repl` ({dec: nal bytes}) substituted, and every SPS's
    VUI timing set to `timing` = (num_units_in_tick, time_scale).
    With `hdrs` ({dec: header plan from explicit.plan}, explicit mode) the
    slice headers named there are rewritten instead of renumbered.
    Returns the number of rewritten slices and SPSs patched."""
    drop = set(drop)
    repl = repl or {}
    log2_fn = spss[0]['log2_max_frame_num']
    max_fn = 1 << log2_fn
    n_renum = n_sps = 0
    sps_cache = {}
    with open(es_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out_path, 'wb', buffering=1 << 22) as o:
            shift = 0
            for p in pics:
                if p['idr']:
                    shift = 0
                sl = p['slices'][0]
                first = sl['offset'] - 3
                if mm[first - 1] == 0:
                    first -= 1
                # non-VCL NALs in front of the picture (SPS/PPS/AUD/SEI) ride along
                n_sps += _write_head(o, mm[p['au_start']:first], timing, sps_cache)
                end = sl['offset'] + sl['size']
                if p['dec'] in drop:
                    if p['ref_idc']:
                        shift += 1
                    o.write(mm[end:p['au_end']])  # trailing zero bytes, if any
                    continue
                if p['dec'] in repl:
                    o.write(b'\x00\x00\x00\x01')
                    nal = repl[p['dec']]
                else:
                    o.write(mm[first:sl['offset']])
                    nal = mm[sl['offset']:end]
                if hdrs is not None:
                    if p['dec'] in hdrs:
                        nal = rewrite_slice(nal, sl, spss, ppss, hdrs[p['dec']])
                        n_renum += 1
                elif shift:
                    rbsp = bytearray(unescape(bytes(nal[1:])))
                    set_bits(rbsp, sl['frame_num_bit'], log2_fn, (sl['frame_num'] - shift) % max_fn)
                    nal = bytes(nal[:1]) + escape(bytes(rbsp))
                    n_renum += 1
                o.write(nal)
                o.write(mm[end:p['au_end']])
        mm.close()
    return n_renum, n_sps


def remap_after_drop(pics, drop):
    """For the stream with `drop` removed: list of original decode indices,
    in new decode order."""
    drop = set(drop)
    return [p['dec'] for p in pics if p['dec'] not in drop]


def new_lists(spss, pics, drop):
    """Header-only reference lists of the deduplicated stream, expressed as
    original decode indices. Returns {orig dec: lists} for kept pictures."""
    shadow = []
    max_fn = 1 << spss[0]['log2_max_frame_num']
    shift = 0
    dropset = set(drop)
    for p in pics:
        if p['idr']:
            shift = 0
        if p['dec'] in dropset:
            if p['ref_idc']:
                shift += 1
            continue
        q = {'slices': [dict(p['slices'][0])], 'idr': p['idr'], 'ref_idc': p['ref_idc'], 'orig': p['dec']}
        q['slices'][0]['frame_num'] = (p['slices'][0]['frame_num'] - shift) % max_fn
        shadow.append(q)
    dpb_walk(spss, shadow)
    by_dec = {p['dec']: p for p in pics}
    out = {}
    for q in shadow:
        out[q['orig']] = [[shadow[i]['orig'] for i in q['lists'][l]] for l in range(2)]
        assert q['poc'] == by_dec[q['orig']]['poc']
    return out


def free_drop_checker(spss, pics):
    """Returns free(dec): True when removing picture `dec` (a reference
    picture) changes no other picture's reference lists in its IDR period,
    i.e. nothing reads it and the sliding window stays equivalent. Such a
    drop is bit-exact for every other frame, like a non-reference drop."""
    periods = {}
    for p in pics:
        periods.setdefault(p['period'], []).append(p)

    def free(dec):
        p = pics[dec]
        if p['idr']:
            return False
        per = periods[p['period']]
        nl = new_lists(spss, per, {dec})
        return all(nl[q['dec']] == q['lists'] for q in per if q['dec'] != dec)
    return free

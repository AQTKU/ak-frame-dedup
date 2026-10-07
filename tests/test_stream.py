"""Bitstream-level tests on a short real extract and an end-to-end fail-out.

The CABAC / SPS tests need sample.mkv (or another test file from the brief)
in the repository root; they are skipped otherwise. The fail-out test needs
ffmpeg with libx264.
"""
import hashlib
import os
import subprocess
import sys

import pytest

from frame_dedup.decode import frame_size, iter_frames
from frame_dedup.h264hdr import iter_nals, parse_sps, patch_sps_timing, sps_frame_rate, unescape
from frame_dedup.roundtrip_test import check
from frame_dedup.stream import load, write_stream

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = [os.path.join(ROOT, f) for f in ('sample.mkv', 'foliage.mkv', 'smoke.mkv')]


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    src = next((s for s in SOURCES if os.path.exists(s)), None)
    if src is None:
        pytest.skip('no test MKV in the repository root')
    out = str(tmp_path_factory.mktemp('clip') / 'clip.h264')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-t', '4', '-map', '0:v:0', '-c', 'copy',
                    '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', out], check=True)
    return out


def test_cabac_roundtrip(clip):
    import mmap
    spss, ppss, pics = load(clip)
    with open(clip, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        picked, seen = [], set()
        for p in pics:
            if p['type'] not in seen and p['slices'][0]['size'] > 3000:
                seen.add(p['type'])
                picked.append(p)
        assert seen == {'I', 'P', 'B'}
        assert all(check(mm, spss, ppss, p) for p in picked)
        mm.close()


def test_sps_timing_patch(clip):
    for s, e, mm in iter_nals(clip):
        if mm[s] & 0x1f == 7:
            nal = bytes(mm[s:e])
            break
    for nuit, ts, fps in ((1001, 48000, (48000, 2002)), (1, 50, (50, 2)), (1, 48, (48, 2))):
        new = patch_sps_timing(nal, nuit, ts)
        a, b = parse_sps(unescape(nal[1:])), parse_sps(unescape(new[1:]))
        assert sps_frame_rate(b) == fps
        for k in a:
            if k != 'vui':
                assert a[k] == b[k]
        for k in a['vui']:
            if k != 'timing':
                assert a['vui'][k] == b['vui'][k]


def _hashes(path, pics):
    fs = frame_size(load(path)[0][0])
    keys = sorted((p['period'], p['poc']) for p in pics)
    return dict(zip(keys, (hashlib.md5(f).hexdigest() for f in iter_frames(path, [(0, os.path.getsize(path))], fs))))


def test_drop_nonref_is_bit_exact(clip, tmp_path):
    spss, ppss, pics = load(clip)
    drop = {p['dec'] for p in pics if p['type'] == 'B' and not p['ref_idc']}
    drop = set(sorted(drop)[::4])
    out = str(tmp_path / 'drop.h264')
    write_stream(clip, pics, spss, out, drop=drop, timing=(1001, 48000))
    s2, _, p2 = load(out)
    assert len(p2) == len(pics) - len(drop)
    assert sps_frame_rate(s2[0]) == (48000, 2002)
    a, b = _hashes(clip, pics), _hashes(out, p2)
    assert all(a[k] == v for k, v in b.items())


def _has_x264():
    r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
    return 'libx264' in r.stdout


@pytest.mark.skipif(not _has_x264(), reason='ffmpeg without libx264')
def test_cli_fails_out_on_native_content(tmp_path):
    """Native 29.97 fps motion (no repeated frames): exit code 2, no output."""
    mkv = str(tmp_path / 'native.mkv')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc2=size=640x360:rate=30000/1001:duration=40', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=40', '-c:v', 'libx264', '-profile:v', 'high', '-preset', 'fast',
                    '-x264-params', 'bframes=3:b-pyramid=none:slices=1:threads=1:direct=temporal',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', mkv], check=True)
    out = str(tmp_path / 'out.mkv')
    rep = str(tmp_path / 'r.json')
    r = subprocess.run([sys.executable, '-m', 'frame_dedup', mkv, '-o', out, '--report', rep,
                        '--work', str(tmp_path / 'work'), '--workers', '2'],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'no duplicate-frame cadence' in r.stdout
    assert not os.path.exists(out)
    assert not os.path.exists(out + '.partial.mkv')

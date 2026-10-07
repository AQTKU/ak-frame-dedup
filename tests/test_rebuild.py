"""Anchor rebuild on a real pair: smoke.mkv display 720, a P picture K
followed by an IDR D, where K is the forward anchor of the two B pictures
before it. Needs smoke.mkv in the repository root (skipped otherwise)."""
import hashlib
import os
import subprocess

import pytest

from frame_dedup.decode import frame_size, iter_frames
from frame_dedup.rebuild import PCM_THRESHOLD, RECOVERY_SEI, candidate, plan_headers, rebuild
from frame_dedup.stream import load
from frame_dedup.verify import verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMOKE = os.path.join(ROOT, 'smoke.mkv')


@pytest.fixture(scope='module')
def clip(tmp_path_factory):
    if not os.path.exists(SMOKE):
        pytest.skip('smoke.mkv not in the repository root')
    d = tmp_path_factory.mktemp('rebuild')
    head = str(d / 'head.h264')
    subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-i', SMOKE, '-t', '40', '-map', '0:v:0', '-c', 'copy',
                    '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', head], check=True)
    _, _, pics = load(head)
    D = next(p for p in pics if p['disp'] == 720)
    assert D['idr']
    per = [p for p in pics if p['period'] in (D['period'] - 1, D['period'])]
    out = str(d / 'clip.h264')
    with open(head, 'rb') as f:
        f.seek(per[0]['au_start'])
        data = f.read(per[-1]['au_end'] - per[0]['au_start'])
    with open(out, 'wb') as f:
        f.write(data)
    return str(d), out


def test_rebuild_pair_720(clip):
    d, es = clip
    spss, ppss, pics = load(es)
    periods = {}
    for p in pics:
        periods.setdefault(p['period'], []).append(p)
    D = periods[1][0]
    K = next(p for p in pics if p['disp'] == D['disp'] - 1)
    assert D['idr'] and K['type'] == 'P' and K['ref_idc']
    r, why = candidate(spss, pics, periods, K['dec'], D['dec'])
    assert r is not None, why
    assert r['B'] and all(pics[b]['type'] == 'B' and not pics[b]['ref_idc'] for b in r['B'])

    # headers: I2 takes K's frame_num and POC; D's period continues from it
    entries = plan_headers(pics, spss, [r])
    assert len(entries) == len(pics) - 1
    i2 = next(e for e in entries if e[1])
    assert i2[0] is D and i2[2] == K['frame_num'] and i2[3] == K['poc']
    after = [e for e in entries if e[0]['period'] == 1 and not e[1]]
    assert all(e[3] == e[0]['poc'] + K['poc'] - D['poc'] for e in after)

    out = os.path.join(d, 'rebuilt.h264')
    orig_of_out, summ = rebuild(es, es, out, spss, ppss, pics, pics, list(range(len(pics))), [r], 1, d,
                                log=lambda *a: None)
    assert summ['pairs'] == 1 and summ['worst_in_leading_b'] < PCM_THRESHOLD
    ospss, _, opics = load(out)
    assert len(opics) == len(pics) - 1
    assert sum(p['idr'] for p in opics) == 1  # one merged IDR period
    data = open(out, 'rb').read()
    assert data.count(b'\x00\x00\x00\x01' + RECOVERY_SEI) == 1

    # everything but the leading B pictures decodes bit-exact
    res = verify(es, out, spss, pics, opics, 1, log=lambda *a: None, orig_of_out=orig_of_out)
    assert not res['errors'] and res['frames_compared'] == len(opics)
    lead = {pics[b]['disp'] for b in r['B']}
    assert {x['display'] for x in res['differing_frames']} <= lead
    assert res['worst_difference'] < PCM_THRESHOLD

    # I2 is a random-access point: decoding from it reproduces every frame
    # from I2 on (the leading B pictures, which precede it, are skipped)
    i2pic = next(q for q in opics if q['type'] == 'I' and not q['idr'] and q['poc'] == K['poc'])
    fs = frame_size(ospss[0])
    full = [hashlib.md5(f).digest() for f in iter_frames(out, [(0, len(data))], fs)]
    part = [hashlib.md5(f).digest() for f in iter_frames(out, [(i2pic['au_start'], len(data))], fs)]
    n_from = sum(1 for q in opics if q['poc'] >= i2pic['poc'])
    assert part == full[-n_from:]

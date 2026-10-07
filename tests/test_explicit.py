"""Explicit reference control (explicit.py) on a B-pyramid stream with MMCO
and reference list modification. Needs mmco-or-ref-list-modification.mkv in
the repository root (skipped otherwise)."""
import hashlib
import os
import subprocess

import pytest

from frame_dedup import explicit as X
from frame_dedup.h264mv import modify_list
from frame_dedup.stream import load, write_explicit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'mmco-or-ref-list-modification.mkv')


def test_modify_list_reorders_and_duplicates():
    # x264 weightp: [P(fn 1), P(fn 1) again, B(fn 2), I(fn 0)] at frame_num 3
    refs = [{'dec': 10, 'frame_num': 0}, {'dec': 11, 'frame_num': 1}, {'dec': 12, 'frame_num': 2}]
    init = [12, 11, 10]
    cmds = [(0, 1), (0, 15), (1, 0), (0, 1)]
    assert modify_list(init, cmds, 4, 3, refs, 16) == [11, 11, 12, 10]
    # the planner's own commands reproduce any list
    want = [10, 12, 12, 11]
    assert modify_list(init, X._rplm_cmds(want, 3, refs, 16), 4, 3, refs, 16) == want


@pytest.fixture(scope='module')
def head(tmp_path_factory):
    if not os.path.exists(SRC):
        pytest.skip('mmco-or-ref-list-modification.mkv not in the repository root')
    d = tmp_path_factory.mktemp('explicit')
    es = str(d / 'head.h264')
    subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-t', '20', '-i', SRC, '-map', '0:v:0', '-c', 'copy',
                    '-bsf:v', 'h264_mp4toannexb', '-f', 'h264', es], check=True)
    spss, ppss, pics = load(es)
    # keep whole IDR periods only
    last = max(p['period'] for p in pics)
    pics = [p for p in pics if p['period'] < last]
    return str(d), es, spss, ppss, pics


def _md5s(path):
    r = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-fps_mode', 'passthrough', '-f', 'framemd5', '-'],
                       capture_output=True, text=True)
    assert not r.stderr.strip(), r.stderr[:300]
    return [l.split(',')[-1].strip() for l in r.stdout.splitlines() if not l.startswith('#')]


def test_identity_plan_reproduces_every_header(head):
    _, es, spss, ppss, pics = head
    assert X.stream_is_explicit(pics)
    entries, hdrs = X.plan_stream(spss, pics, set(), {}, [])
    assert all(h['same'] for h in hdrs)
    # rewriting a header with its own values gives the same bytes
    with open(es, 'rb') as f:
        data = f.read()
    for p, h in list(zip(pics, hdrs))[:200]:
        sl = p['slices'][0]
        nal = data[sl['offset']:sl['offset'] + sl['size']]
        assert X.rewrite_slice(nal, sl, spss, ppss, h) == nal, p['dec']


def test_drop_reference_twin(head):
    """Drop every reference B whose display predecessor is a P (the most
    common hard pair), substituted by that P: the stream decodes, parses to
    the planned lists, and every picture that does not read a dropped picture
    (directly or through a reference that does) is bit-exact."""
    d, es, spss, ppss, pics = head
    disp = sorted(pics, key=lambda p: (p['period'], p['poc']))
    sigma = {}
    for a, b in zip(disp, disp[1:]):
        if a['type'] == 'P' and b['type'] == 'B' and b['ref_idc'] and a['period'] == b['period'] and b['dec'] > a['dec']:
            sigma[b['dec']] = a['dec']
    assert len(sigma) > 20
    entries, hdrs = X.plan_stream(spss, pics, set(sigma), sigma, [])
    out = os.path.join(d, 'drop.h264')
    write_explicit(es, pics, spss, ppss, out, entries, hdrs)
    _, _, npics = load(out)
    src = [e['pic']['dec'] for e in entries]
    for q, h in zip(npics, hdrs):
        assert [[src[i] for i in l] for l in q['lists']] == h['lists']
    tainted = set(sigma)
    for p in pics:
        if any(x in tainted for l in p['lists'] for x in l) and p['ref_idc']:
            tainted.add(p['dec'])
    readers = {p['dec'] for p in pics if any(x in tainted for l in p['lists'] for x in l)}
    a, b = _md5s(es), _md5s(out)
    kept = [p for p in disp if p['dec'] not in sigma]
    assert len(b) == len(kept)
    by_disp = {p['dec']: i for i, p in enumerate(disp)}
    for i, p in enumerate(kept):
        if p['dec'] not in readers:
            assert a[by_disp[p['dec']]] == b[i], p['dec']


def test_anchor_rebuild_plan(head):
    """Demote an IDR into the slot of the reference picture before it: the
    merged stream decodes, I2 decodes exactly like the IDR, and everything
    from I2 on in display order is bit-exact (only the leading pictures read
    a different anchor)."""
    d, es, spss, ppss, pics = head
    periods = {}
    for p in pics:
        periods.setdefault(p['period'], []).append(p)
    D = next(p for p in pics if p['idr'] and p['period'] == 2)
    K = max((p for p in periods[1] if p['ref_idc']), key=lambda p: p['poc'])
    rb, why = X.candidate(spss, pics, periods, K['dec'], D['dec'])
    assert rb is not None, why
    entries, hdrs = X.plan_stream(spss, pics, set(), {}, [(K['dec'], D['dec'])])
    out = os.path.join(d, 'rebuild.h264')
    write_explicit(es, pics, spss, ppss, out, entries, hdrs)
    _, _, npics = load(out)
    assert len(npics) == len(pics) - 1 and not npics[K['dec']]['idr']
    a, b = _md5s(es), _md5s(out)
    disp = sorted(pics, key=lambda p: (p['period'], p['poc']))
    kept = [p for p in disp if p['dec'] != K['dec']]
    lead = set(rb['B'])
    i2 = [i for i, p in enumerate(kept) if p['dec'] == D['dec']][0]
    for i, p in enumerate(kept):
        if p['dec'] in lead:
            continue
        assert a[disp.index(p)] == b[i], (p['dec'], i >= i2)

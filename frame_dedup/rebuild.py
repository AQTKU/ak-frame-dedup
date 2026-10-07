"""Anchor rebuild: drop the earlier twin K of a pair whose later twin D is an
IDR, by demoting D to a non-IDR I picture ("I2") that takes K's place.

Display / decode order around such a pair (K a P picture, D an IDR):

    display:  P0  B1 B2 B3  K | D  b b b  P ...
    decode:   P0  ... K  B1 B2 B3  D  P  b b b ...

K and D encode the same source frame. I2 is D's slice data under a non-IDR
header carrying K's frame_num and K's POC, written in K's decode slot, so
B1-B3 see an anchor at exactly the same list positions and POC distances.
D's former IDR period is merged into K's: its POCs are shifted by the
constant POC(K) - POC(D) and frame_num continues from K's. Inter-picture POC
differences in that period are unchanged, so its direct modes and implicit
weights are too, and the reference indices it uses still point at the same
pictures (`check_lists`, header-only: older pictures that now linger in the
DPB only ever extend a list beyond the entries the original could use).

B1-B3 lose K's motion field (I2 is all-intra), so their temporal-direct
blocks whose motion would change are rewritten as explicit prediction with
the original motion (`reexpress_compact`). Their reads of K now read I2, a
second encode of the same frame, so macroblocks whose decoded result moves
by PCM_THRESHOLD or more become I_PCM copies of the original, worst first and
at most PCM_MAX_FRACTION of the picture. B1-B3 are
non-reference pictures, so nothing propagates.

I2 stays a random-access point through a recovery point SEI
(recovery_frame_cnt 0, exact_match 1): an open GOP whose leading pictures
are B1-B3.
"""
import copy
import mmap
import multiprocessing as mp
import os
import time

from .decode import decode_pocs, frame_size, iter_frames
from .h264cabac import B_TYPES
from .h264hdr import escape, parse_slice, set_bits, unescape
from .h264mv import Motion, derive_motion, dpb_walk
from .patch import fix_qp, pcm_record, qp_chain, same_motion
from .roundtrip_test import decode_picture, encode_picture, from_bits, to_bits
from .system import REPAIR_PEAK, lower_priority, peak_rss, pool_size
from .verify import mb_maxdiff
from .verify_motion import reexpress

# sei_message: payloadType 6 (recovery point), payloadSize 1,
# recovery_frame_cnt ue 0 ('1'), exact_match 1, broken_link 0,
# changing_slice_group_idc 00, payload alignment '100'; rbsp_trailing_bits
RECOVERY_SEI = b'\x06\x06\x01\xc4\x80'

# A B1-B3 macroblock whose decoded result differs from the original by this
# much or more (max over Y, U, V) is replaced by I_PCM, up to a budget. K and D themselves
# differ by up to 23-90 levels; at this threshold the encoder's local
# "refresh" spots are caught for a handful of MBs per picture.
PCM_THRESHOLD = 16
# At most this fraction of a picture's MBs is patched, worst first. In fast
# motion the IDR twin can be coded very differently from K (MAD 2-5, peaks
# of 60-120), and holding every MB under the threshold would mean PCM-ing
# most of the picture; there the motion masks the remaining differences.
PCM_MAX_FRACTION = 0.02
MAX_ROUNDS = 4


# ---------------------------------------------------------------- headers

def _int_bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def demote_idr(nal, sl, spss, ppss, frame_num, poc_lsb):
    """Return an IDR slice NAL rewritten as a non-IDR I slice (nal_unit_type 1,
    same nal_ref_idc) with the given frame_num and pic_order_cnt_lsb:
    idr_pic_id removed, dec_ref_pic_marking replaced by
    adaptive_ref_pic_marking_mode_flag = 0. The CABAC slice data starts
    byte-aligned and depends only on slice QP and slice type, so its bytes
    are appended unchanged."""
    assert nal[0] & 0x1f == 5 and sl['nal_type'] == 5
    sps = spss[ppss[sl['pps_id']]['sps_id']]
    rbsp = unescape(bytes(nal[1:]))
    lfn, lpoc = sps['log2_max_frame_num'], sps['log2_max_poc_lsb']
    hb = to_bits(rbsp[:sl['data_bit'] // 8])
    assert all(hb[sl['hdr_bits']:]), 'cabac_alignment_one_bits expected'
    assert sl['idr_pic_id_bit'] == sl['frame_num_bit'] + lfn
    out = hb[:sl['frame_num_bit']] + _int_bits(frame_num, lfn)   # up to frame_num; idr_pic_id dropped
    out += _int_bits(poc_lsb, lpoc)
    out += hb[sl['poc_lsb_bit'] + lpoc:sl['drpm_bit']]
    out.append(0)  # adaptive_ref_pic_marking_mode_flag
    out += hb[sl['drpm_end_bit']:sl['hdr_bits']]
    hdr_bits = len(out)
    while len(out) % 8:
        out.append(1)  # cabac_alignment_one_bit
    new_rbsp = from_bits(out) + rbsp[sl['data_bit'] // 8:]
    ref_idc = (nal[0] >> 5) & 3
    new = bytes([(nal[0] & 0xe0) | 1]) + escape(new_rbsp)
    chk = parse_slice(new_rbsp[:64], 1, ref_idc, spss, ppss)
    for k in ('first_mb', 'slice_type', 'pps_id', 'qp', 'deblock_idc', 'alpha_off', 'beta_off',
              'delta_poc_bottom', 'cabac_init_idc'):
        assert chk.get(k) == sl.get(k), (k, chk.get(k), sl.get(k))
    assert chk['frame_num'] == frame_num and chk['poc_lsb'] == poc_lsb and not chk['mmco']
    assert chk['hdr_bits'] == hdr_bits and chk['data_bit'] == len(out)
    return new


def plan_headers(pics, spss, rebuilds):
    """The rewritten stream in decode order: [(picture, is_i2, frame_num,
    poc)]. D is written in K's slot as I2 with K's POC; every POC of D's
    former period is shifted by POC(K) - POC(D); frame_num is re-derived as
    PrevRefFrameNum + 1 (0 at a real IDR), which is what the input carries,
    since it has no frame_num gaps."""
    sps = spss[0]
    max_fn = 1 << sps['log2_max_frame_num']
    byK = {r['K']: r for r in rebuilds}
    byD = {r['D']: r for r in rebuilds}
    by_dec = {p['dec']: p for p in pics}
    out, i2poc = [], {}
    prev_fn = off = 0
    for p in pics:
        if p['dec'] in byD:
            off = i2poc[p['dec']] - p['poc']
            continue
        if p['dec'] in byK:
            D = by_dec[byK[p['dec']]['D']]
            fn = (prev_fn + 1) % max_fn
            poc = p['poc'] + off
            out.append((D, True, fn, poc))
            i2poc[D['dec']] = poc
            prev_fn = fn
            continue
        if p['idr']:
            off = fn = 0
        else:
            fn = (prev_fn + 1) % max_fn
        out.append((p, False, fn, p['poc'] + off))
        if p['ref_idc']:
            prev_fn = fn
    return out


def check_lists(spss, pics, rebuilds):
    """Simulate the decoder's reference lists for the rewritten stream from
    headers alone. Every list entry the original could use must still point
    at the same picture (B1-B3 read I2 where they read K). Returns (failures,
    shadow pictures in new decode order)."""
    max_lsb = 1 << spss[0]['log2_max_poc_lsb']
    entries = plan_headers(pics, spss, rebuilds)
    shadow = []
    for p, i2, fn, poc in entries:
        sl = dict(p['slices'][0], frame_num=fn, poc_lsb=poc % max_lsb)
        shadow.append({'slices': [sl], 'idr': p['idr'] and not i2, 'ref_idc': p['ref_idc'], 'src': p['dec'],
                       'want_poc': poc})
    dpb_walk(spss, shadow)
    sub = {}
    for r in rebuilds:
        for b in r['B']:
            sub[b] = (r['K'], r['D'])
    by_dec = {p['dec']: p for p in pics}
    fails = []
    for (p, i2, fn, poc), q in zip(entries, shadow):
        if q['poc'] != poc:
            fails.append({'dec': p['dec'], 'what': 'poc', 'want': poc, 'got': q['poc']})
        for l in range(2):
            new = [shadow[i]['src'] for i in q['lists'][l]]
            old = [] if i2 else by_dec[p['dec']]['lists'][l]
            if p['dec'] in sub:
                K, D = sub[p['dec']]
                old = [D if x == K else x for x in old]
            if new[:len(old)] != old:
                fails.append({'dec': p['dec'], 'what': f'list{l}', 'want': old, 'got': new})
    return fails, shadow


def candidate(spss, pics, periods, K, D):
    """Header-only eligibility of a rebuild: D an IDR, K a non-IDR reference
    picture ending the previous period, only non-reference pictures between
    them in decode order, nothing outside those reading K, and the list check
    passing. Returns (rebuild dict or None, reason)."""
    pK, pD = pics[K], pics[D]
    if not pD['idr'] or pK['idr'] or not pK['ref_idc']:
        return None, 'twin types'
    if pK['period'] != pD['period'] - 1:
        return None, 'twins not in adjacent IDR periods'
    between = range(K + 1, D)
    if any(pics[d]['ref_idc'] for d in between):
        return None, 'reference picture between the twins'
    readers = [q['dec'] for q in periods[pK['period']] if K in q['lists'][0] or K in q['lists'][1]]
    if any(d not in between for d in readers):
        return None, 'K read outside the leading pictures'
    r = {'K': K, 'D': D, 'B': readers}
    fails, _ = check_lists(spss, periods[pK['period']] + periods[pD['period']], [r])
    if fails:
        return None, f'reference lists change: {fails[0]}'
    return r, None


# ---------------------------------------------------------------- writing

def _first_sc(mm, sl):
    first = sl['offset'] - 3
    if mm[first - 1] == 0:
        first -= 1
    return first


def _patch_fn_poc(nal, sl, sps, fn, lsb):
    rbsp = bytearray(unescape(bytes(nal[1:])))
    set_bits(rbsp, sl['frame_num_bit'], sps['log2_max_frame_num'], fn)
    set_bits(rbsp, sl['poc_lsb_bit'], sps['log2_max_poc_lsb'], lsb)
    return bytes(nal[:1]) + escape(bytes(rbsp))


def write_rebuilt(src_es, pics, spss, ppss, out_path, rebuilds, repl=None):
    """Write `pics` (a contiguous run of a parsed stream starting at an IDR)
    with every rebuild applied: K removed; D demoted and written in K's slot
    behind K's and D's non-VCL NALs (D's SPS / PPS travel with it) and a
    recovery point SEI; frame_num and POC rewritten per `plan_headers`.
    `repl`: {dec: NAL} for re-encoded pictures. Returns the plan entries."""
    sps = spss[0]
    max_lsb = 1 << sps['log2_max_poc_lsb']
    repl = repl or {}
    byK = {r['K']: r for r in rebuilds}
    entries = plan_headers(pics, spss, rebuilds)
    at = {p['dec']: (fn, poc) for p, i2, fn, poc in entries if not i2}
    i2_of = {p['dec']: (fn, poc) for p, i2, fn, poc in entries if i2}
    i2_at = {r['K']: i2_of[r['D']] for r in rebuilds}
    by_dec = {p['dec']: p for p in pics}
    with open(src_es, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out_path, 'wb', buffering=1 << 22) as o:
            for p in pics:
                sl = p['slices'][0]
                first = _first_sc(mm, sl)
                end = sl['offset'] + sl['size']
                if p['dec'] in byK:
                    D = by_dec[byK[p['dec']]['D']]
                    dsl = D['slices'][0]
                    fn, poc = i2_at[p['dec']]
                    o.write(mm[p['au_start']:first])
                    o.write(mm[D['au_start']:_first_sc(mm, dsl)])
                    o.write(b'\x00\x00\x00\x01' + RECOVERY_SEI)
                    o.write(b'\x00\x00\x00\x01' + demote_idr(mm[dsl['offset']:dsl['offset'] + dsl['size']], dsl,
                                                             spss, ppss, fn, poc % max_lsb))
                    o.write(mm[end:p['au_end']])
                    continue
                if p['dec'] not in at:  # a demoted D: already written in K's slot
                    o.write(mm[end:p['au_end']])
                    continue
                fn, poc = at[p['dec']]
                o.write(mm[p['au_start']:first])
                if p['dec'] in repl:
                    o.write(b'\x00\x00\x00\x01')
                    nal = repl[p['dec']]
                else:
                    o.write(mm[first:sl['offset']])
                    nal = mm[sl['offset']:end]
                if fn != sl['frame_num'] or poc % max_lsb != sl['poc_lsb']:
                    nal = _patch_fn_poc(nal, sl, sps, fn, poc % max_lsb)
                o.write(nal)
                o.write(mm[end:p['au_end']])
        mm.close()
    return entries


# ---------------------------------------------------------------- B1-B3

def reexpress_compact(mb, motion):
    """An explicit B macroblock record carrying `motion` for a B_Skip /
    B_Direct_16x16 MB (or a B_8x8 MB with direct sub-blocks), using the
    coarsest partitioning that fits: direct motion is uniform per 8x8
    quadrant (direct_8x8_inference), so 16x16, 16x8 or 8x16 often suffice
    and cost far fewer mvds than 8x8. mvds are left for derive_motion's
    target mode to solve."""
    if mb.get('part') == '8x8' and not mb['skip']:
        return reexpress(mb, 'B', motion)
    x0, y0 = mb['x'] * 4, mb['y'] * 4

    def quad(i):
        gx, gy = x0 + (i % 2) * 2, y0 + (i // 2) * 2
        return tuple((motion.refidx[l][gy][gx], motion.mv[l][gy][gx] if motion.refidx[l][gy][gx] >= 0 else None)
                     for l in range(2))
    q = [quad(i) for i in range(4)]
    if q[0] == q[1] == q[2] == q[3]:
        part, groups = '16x16', [q[0]]
    elif q[0] == q[1] and q[2] == q[3]:
        part, groups = '16x8', [q[0], q[2]]
    elif q[0] == q[2] and q[1] == q[3]:
        part, groups = '8x16', [q[0], q[1]]
    else:
        return reexpress(mb, 'B', motion)
    preds = tuple('Bi' if g[0][0] >= 0 and g[1][0] >= 0 else ('L0' if g[0][0] >= 0 else 'L1') for g in groups)
    n = copy.deepcopy(mb)
    n.update(skip=0, kind='B', part=part, btype=B_TYPES.index((part, preds)), direct16=0)
    n['refs'] = [[g[l][0] if g[l][0] >= 0 else None for g in groups] for l in range(2)]
    n['mvds'] = [[(0, 0) if g[l][0] >= 0 else None for g in groups] for l in range(2)]
    n.pop('sub', None)
    if mb['skip']:
        n.update(cbp=0, t8=0, qpd=0, coef={})
    return n


def _has_direct(mb):
    return not mb['intra'] and (mb['skip'] or mb.get('part') == 'D' or
                                (mb.get('part') == '8x8' and 0 in mb['sub']))


def _intra_motion(W, H):
    m = Motion(W, H)
    for row in m.intra:
        for x in range(len(row)):
            row[x] = 1
    return m


def _build_b(pic, dd, base, m0, pcm, yuv, pics, intra, W, H):
    """Records for a B1-B3 picture: `base` (direct blocks made explicit)
    with the MBs in `pcm` replaced by I_PCM copies of `yuv`, skips that can
    no longer inherit their QP made explicit, mvds re-solved and the QP chain
    kept. Returns (records, number of skips made explicit)."""
    new = copy.deepcopy(base)
    for i, mb in enumerate(new):
        if (mb['x'], mb['y']) in pcm:
            r = pcm_record(yuv, W, mb['x'], mb['y'])
            r.update(x=mb['x'], y=mb['y'])
            new[i] = r
    qp = pic['slices'][0]['qp']
    target_qp = qp_chain(dd['mbs'], qp)
    pred, n_skip = qp, 0
    for i, mb in enumerate(new):
        if mb['kind'] == 'PCM':
            continue
        if mb['skip'] and target_qp[i] != pred:
            new[i] = reexpress_compact(mb, m0)
            n_skip += 1
        pred = target_qp[i]
    m2 = derive_motion(dict(pic), new, W, H, pics, intra, target=m0)
    bad = [(mb['x'], mb['y']) for mb in new if mb['kind'] != 'PCM' and not mb['intra']
           and not same_motion(m0, m2, mb['x'], mb['y'])]
    assert not bad, f"dec {pic['dec']}: motion could not be preserved at {bad[:5]}"
    forced = fix_qp(new, target_qp, qp)
    return new, n_skip, forced


_G = {}


def _init(orig_es, src_es, spss, ppss, work):
    lower_priority()
    _G.update(orig_es=orig_es, src_es=src_es, spss=spss, ppss=ppss, work=work)


def run_task(task):
    """Re-encode B1-B3 of one rebuild. Returns their NALs and statistics."""
    t0 = time.time()
    spss, ppss = _G['spss'], _G['ppss']
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']
    fs = frame_size(sps)
    r = task['rebuild']
    src = task['src']                         # {dec: slim picture}, K's period through D
    K, D, Bs = r['K'], r['D'], r['B']
    f = open(_G['src_es'], 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    intra = _intra_motion(W, H)
    dK = decode_picture(mm, spss, ppss, src[K])
    mK = derive_motion(dict(src[K]), dK['mbs'], W, H, src)
    recs = {}
    stats = {b: {'src_dec': b, 'display_in': task['disp_in'][b]} for b in Bs}
    for b in Bs:
        pb = src[b]
        assert pb['lists'][1][0] == K
        dd = decode_picture(mm, spss, ppss, pb)
        m0 = derive_motion(dict(pb), dd['mbs'], W, H, src, mK)        # what B decoded with K
        m1 = derive_motion(dict(pb), dd['mbs'], W, H, src, intra)     # what its records give with I2
        base, n_dir = [], 0
        for mb in dd['mbs']:
            if _has_direct(mb) and not same_motion(m0, m1, mb['x'], mb['y']):
                base.append(reexpress_compact(mb, m0))
                n_dir += 1
            else:
                base.append(mb)
        recs[b] = (dd, base, m0)
        stats[b].update(direct_explicit=n_dir, bytes_before=len(dd['nal']))
    mm.close()
    f.close()

    # original pixels of B1-B3
    orig = task['orig']
    frames = decode_pocs(_G['orig_es'], orig, task['orig_first'], task['orig_last'] + 1,
                         set(task['orig_poc'].values()), fs)
    target = {b: frames[task['orig_poc'][b]] for b in Bs}

    seg = [src[d] for d in sorted(src)]
    tmp = os.path.join(_G['work'], f'rebuild_{K}.h264')
    pcm = {b: set() for b in Bs}
    budget = int(PCM_MAX_FRACTION * W * H)
    try:
        for rnd in range(MAX_ROUNDS):
            repl = {}
            for b in Bs:
                dd, base, m0 = recs[b]
                new, n_skip, forced = _build_b(src[b], dd, base, m0, pcm[b], target[b], src, intra, W, H)
                repl[b], _ = encode_picture(dd, new)
                stats[b].update(skip_explicit=n_skip, qp_carrier=forced, pcm=len(pcm[b]),
                                bytes_after=len(repl[b]))
            entries = write_rebuilt(_G['src_es'], seg, spss, ppss, tmp, [r], repl)
            order = sorted((poc, p['dec']) for p, i2, fn, poc in entries)
            got = {}
            for (poc, d), fr in zip(order, iter_frames(tmp, [(0, os.path.getsize(tmp))], fs)):
                if d in pcm:
                    got[d] = fr
            assert set(got) == set(Bs), 'segment decode lost a leading picture'
            grew = False
            for b in Bs:
                m, ly = mb_maxdiff(got[b], target[b], W, H)
                stats[b].update(worst=int(m.max()), mbs_ge3=int((m >= 3).sum()), mbs_diff=int((m > 0).sum()),
                                worst_luma=int(ly.max()), mbs_ge3_luma=int((ly >= 3).sum()), rounds=rnd + 1)
                more = sorted(((int(m[y, x]), int(x), int(y)) for y, x in zip(*(m >= PCM_THRESHOLD).nonzero())
                               if (int(x), int(y)) not in pcm[b]), reverse=True)
                room = budget - len(pcm[b])
                stats[b]['pcm_capped'] = len(more) > room
                if more and room > 0:
                    pcm[b] |= {(x, y) for _, x, y in more[:room]}
                    grew = True
            if not grew:
                break
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return {'K': K, 'repl': repl, 'stats': [stats[b] for b in Bs],
            'seconds': round(time.time() - t0, 1), 'peak_rss': peak_rss()}


# ---------------------------------------------------------------- driver

def _slim(p):
    keep = ('slices', 'idr', 'ref_idc', 'dec', 'poc', 'lists', 'type', 'au_start', 'au_end', 'disp', 'period')
    return {k: p[k] for k in keep if k in p}


def rebuild(orig_es, src_es, out_path, spss, ppss, orig_pics, src_pics, orig_of_src, rebuilds, workers, work,
            log=print):
    """Apply `rebuilds` ({'K', 'D', 'B'} in src decode indices) to the stream
    src_es (parsed as src_pics; orig_of_src maps its decode indices to the
    original stream's) and write out_path. Returns (original decode index of
    every output picture in output decode order, summary)."""
    t0 = time.time()
    fails, shadow = check_lists(spss, src_pics, rebuilds)
    if fails:
        raise RuntimeError(f'anchor rebuild would change reference lists: {fails[:3]}')
    periods = {}
    for p in src_pics:
        periods.setdefault(p['period'], []).append(p)
    tasks = []
    for r in rebuilds:
        K, D = src_pics[r['K']], src_pics[r['D']]
        src = {p['dec']: _slim(p) for p in periods[K['period']]}
        src[D['dec']] = _slim(D)
        ob = [orig_of_src[b] for b in r['B']]
        ofirst = max(d for d in range(min(ob) + 1) if orig_pics[d]['idr'])
        orig = {d: _slim(orig_pics[d]) for d in range(ofirst, max(ob) + 1)}
        tasks.append({'rebuild': r, 'src': src, 'orig': orig, 'orig_first': ofirst, 'orig_last': max(ob),
                      'orig_poc': {b: orig_pics[orig_of_src[b]]['poc'] for b in r['B']},
                      'disp_in': {b: orig_pics[orig_of_src[b]]['disp'] for b in r['B']}})
    nproc = pool_size(workers, REPAIR_PEAK, len(tasks)) if tasks else 0
    log(f'anchor rebuild: {len(tasks)} pairs, {sum(len(r["B"]) for r in rebuilds)} leading B pictures, '
        f'{nproc} workers')
    results = []
    if tasks:
        with mp.get_context('spawn').Pool(nproc, initializer=_init, maxtasksperchild=8,
                                          initargs=(orig_es, src_es, spss, ppss, work)) as pool:
            for i, res in enumerate(pool.imap_unordered(run_task, tasks)):
                results.append(res)
                if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                    log(f'  rebuild: {i + 1}/{len(tasks)} pairs')
    repl = {}
    for res in results:
        repl.update(res['repl'])
    entries = write_rebuilt(src_es, src_pics, spss, ppss, out_path, rebuilds, repl)
    out_src = [p['dec'] for p, i2, fn, poc in entries]

    # the written stream must parse to exactly the planned POCs and lists
    from .stream import load
    _, _, out_pics = load(out_path)
    assert len(out_pics) == len(shadow)
    for q, s in zip(out_pics, shadow):
        assert q['poc'] == s['poc'] and q['lists'] == s['lists'] and q['idr'] == s['idr'], q['dec']

    byK = {res['K']: res for res in results}
    pairs, worst = [], 0
    for r in rebuilds:
        res = byK[r['K']]
        st = res['stats']
        worst = max([worst] + [s['worst'] for s in st])
        pairs.append({'display': orig_pics[orig_of_src[r['K']]]['disp'],
                      'kept_display': orig_pics[orig_of_src[r['D']]]['disp'],
                      'types': src_pics[r['K']]['type'] + 'Ii',
                      'leading_b': len(st), 'pcm_mbs': sum(s['pcm'] for s in st),
                      'direct_explicit': sum(s['direct_explicit'] for s in st),
                      'bytes_added': sum(s['bytes_after'] - s['bytes_before'] for s in st),
                      'worst': max((s['worst'] for s in st), default=0),
                      'mbs_ge3': sum(s['mbs_ge3'] for s in st),
                      'worst_luma': max((s['worst_luma'] for s in st), default=0),
                      'pcm_capped': sum(bool(s.get('pcm_capped')) for s in st),
                      'pictures': st, 'seconds': res['seconds']})
    summary = {'pairs': len(pairs), 'leading_b_pictures': sum(p['leading_b'] for p in pairs),
               'pcm_threshold': PCM_THRESHOLD, 'pcm_max_per_picture': int(PCM_MAX_FRACTION * spss[0]['mb_w'] * spss[0]['mb_h']),
               'pictures_pcm_capped': sum(p['pcm_capped'] for p in pairs), 'pcm_mbs': sum(p['pcm_mbs'] for p in pairs),
               'direct_explicit': sum(p['direct_explicit'] for p in pairs),
               'bytes_added': sum(p['bytes_added'] for p in pairs), 'worst_in_leading_b': worst,
               'workers': nproc,
               'worker_peak_rss_mb': round(max((x['peak_rss'] for x in results), default=0) / 2**20),
               'seconds': round(time.time() - t0, 1), 'pairs_detail': pairs}
    return [orig_of_src[d] for d in out_src], summary

"""Explicit reference control for streams that use MMCO / list modification.

Encoders such as x264 with B-pyramid write reference B pictures that unmark
older pictures with MMCO 1, and reorder (and, for weighted prediction,
duplicate) entries of every P list with ref_pic_list_modification. Dropping
a reference picture from such a stream breaks both: the picNum differences
in later headers count the dropped picture, and its own MMCOs never run.

Instead of letting the decoder's default rules shift lists after a drop, the
headers of the kept pictures are regenerated so that the decoder sees, for
every kept picture,

  * exactly the original reference lists, entry by entry, with a dropped
    picture D replaced by its substitute sigma(D) (its kept twin K), and
  * exactly the original set of short-term references after marking, under
    the same substitution,

using ref_pic_list_modification for the lists and MMCO 1 for the marking.
Reference indices therefore keep their meaning, so explicit weighted
prediction tables, spatial direct reference indices and every block that
does not read D stay valid. frame_num is renumbered (no gaps), POCs are kept
(or shifted, for an anchor rebuild). Headers are re-serialised bit by bit;
the CABAC slice data is byte-aligned and is appended unchanged.

`plan` is checked by re-simulating the rewritten headers with dpb_walk.
"""
from .h264hdr import escape, parse_slice, unescape
from .h264mv import dpb_walk, init_lists, modify_list
from .roundtrip_test import from_bits, to_bits


class PlanError(Exception):
    """The requested substitution cannot be expressed in the headers."""


def stream_is_explicit(pics):
    """True when the stream uses MMCO or ref list modification anywhere."""
    for p in pics:
        sl = p['slices'][0]
        if sl['mmco'] or sl.get('adaptive') or sl['rplm'][0] or sl['rplm'][1]:
            return True
    return False


# ---------------------------------------------------------------- bits

def _ue(v):
    v1 = v + 1
    n = v1.bit_length()
    return [0] * (n - 1) + [(v1 >> (n - 1 - i)) & 1 for i in range(n)]


def _int_bits(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def _rplm_bits(t, rplm):
    out = []
    for lst in (0, 1):
        if (lst == 0 and t not in ('I', 'SI')) or (lst == 1 and t == 'B'):
            cmds = rplm[lst]
            if not cmds:
                out.append(0)
                continue
            out.append(1)
            for idc, v in cmds:
                out += _ue(idc) + _ue(v)
            out += _ue(3)
    return out


def _drpm_bits(mmco):
    """dec_ref_pic_marking of a non-IDR reference picture: None = sliding
    window, a list = adaptive marking with these (1, diff_minus1) ops."""
    if mmco is None:
        return [0]
    out = [1]
    for op, *args in mmco:
        out += _ue(op)
        for a in args:
            out += _ue(a)
    return out + _ue(0)


def rewrite_slice(nal, sl, spss, ppss, h):
    """Return the slice NAL with a rewritten header. h: 'frame_num',
    'poc_lsb', 'rplm' ([cmds, cmds]), 'mmco' (None or ops; ignored for a
    non-reference picture) and 'demote' (an IDR becomes a non-IDR slice:
    nal_unit_type 1, idr_pic_id removed, explicit marking). The CABAC slice
    data is appended byte for byte after cabac_alignment_one_bits."""
    pps = ppss[sl['pps_id']]
    sps = spss[pps['sps_id']]
    lfn, lpoc = sps['log2_max_frame_num'], sps['log2_max_poc_lsb']
    rbsp = unescape(bytes(nal[1:]))
    hb = to_bits(rbsp[:sl['data_bit'] // 8])
    assert all(hb[sl['hdr_bits']:]), 'cabac_alignment_one_bits expected'
    idr = sl['nal_type'] == 5
    demote = h.get('demote', False)
    assert not demote or idr
    ref_idc = (nal[0] >> 5) & 3
    out = hb[:sl['frame_num_bit']] + _int_bits(h['frame_num'], lfn)
    if idr and not demote:
        out += hb[sl['frame_num_bit'] + lfn:sl['poc_lsb_bit']]   # idr_pic_id
    out += _int_bits(h['poc_lsb'], lpoc)
    out += hb[sl['poc_lsb_bit'] + lpoc:sl['rplm_bit']]
    out += _rplm_bits(sl['slice_type'], h['rplm'])
    out += hb[sl['rplm_end_bit']:sl['drpm_bit']]                 # pred_weight_table
    if ref_idc:
        if idr and not demote:
            out += hb[sl['drpm_bit']:sl['drpm_end_bit']]
        else:
            out += _drpm_bits(h['mmco'])
    out += hb[sl['drpm_end_bit']:sl['hdr_bits']]
    hdr_bits = len(out)
    while len(out) % 8:
        out.append(1)  # cabac_alignment_one_bit
    new_rbsp = from_bits(out) + rbsp[sl['data_bit'] // 8:]
    nal_type = 1 if demote else sl['nal_type']
    new = bytes([(nal[0] & 0xe0) | nal_type]) + escape(new_rbsp)
    chk = parse_slice(new_rbsp[:len(out) // 8 + 8], nal_type, ref_idc, spss, ppss)
    for k in ('first_mb', 'slice_type', 'pps_id', 'qp', 'deblock_idc', 'alpha_off', 'beta_off',
              'cabac_init_idc', 'num_ref_l0', 'num_ref_l1', 'direct_spatial', 'weighted'):
        assert chk.get(k) == sl.get(k), (k, chk.get(k), sl.get(k))
    assert chk['frame_num'] == h['frame_num'] and chk['poc_lsb'] == h['poc_lsb']
    assert [list(c) for c in chk['rplm']] == [list(c or []) for c in h['rplm']]
    if ref_idc and not (idr and not demote):
        assert chk['mmco'] == list(h['mmco'] or []) and bool(chk.get('adaptive')) == (h['mmco'] is not None)
    assert chk['hdr_bits'] == hdr_bits and chk['data_bit'] == len(out)
    return new


# ---------------------------------------------------------------- planning

def _rplm_cmds(want, cur_fn, refs, max_fn):
    """ref_pic_list_modification commands naming the pictures `want` (decode
    ids) in order."""
    num_of = {}
    for r in refs:
        fn = r['frame_num']
        num_of[r['dec']] = fn - max_fn if fn > cur_fn else fn
    cmds, pred = [], cur_fn
    for d in want:
        num = num_of[d]
        nowrap = num + max_fn if num < 0 else num
        diff = nowrap - pred
        if diff < 0:
            cmds.append((0, -diff - 1))
        elif diff > 0:
            cmds.append((1, diff - 1))
        else:
            cmds.append((0, max_fn - 1))   # the same picture again: a full wrap
        pred = nowrap
    return cmds


def _plan_list(init, want, orig_cmds, n_active, cur_fn, refs, max_fn):
    """Commands for list `lst` giving exactly `want`: the original commands
    if they still do, no commands if the initial list already is `want`,
    else the shortest explicit prefix."""
    if orig_cmds:
        try:
            if modify_list(init, orig_cmds, n_active, cur_fn, refs, max_fn) == want:
                return list(orig_cmds)
        except ValueError:
            pass
    if init == want:
        return []
    for k in range(1, n_active + 1):
        cmds = _rplm_cmds(want[:k], cur_fn, refs, max_fn)
        if modify_list(init, cmds, n_active, cur_fn, refs, max_fn) == want:
            return cmds
    raise AssertionError('a full explicit list always reproduces itself')


def plan(spss, entries, sigma, orig_pics):
    """Plan the headers of a rewritten stream.

    entries: the pictures of the new stream in decode order, each a dict:
      'pic'      original picture (after dpb_walk on the original stream),
      'idr'      whether it is an IDR in the new stream,
      'poc'      its POC in the new stream,
      'demote'   (optional) an original IDR written as a non-IDR I picture,
      'as'       (optional) original picture whose reference role it takes
                 over: its dpb_after is the target (anchor rebuild: I2
                 takes K's).
    sigma: {original dec: original dec of its substitute} for every dropped
    reference picture (and for K of an anchor rebuild); None for a dropped
    reference picture that nothing reads (it just leaves the DPB).
    orig_pics: {original dec: picture} for every picture named in lists.

    Returns a list (aligned with entries) of header dicts for rewrite_slice,
    each with 'same' = True when the original header is reproduced exactly,
    plus 'lists' (the target lists, original ids). Raises PlanError."""
    sps = spss[0]
    max_fn = 1 << sps['log2_max_frame_num']
    max_lsb = 1 << sps['log2_max_poc_lsb']
    max_refs = max(sps['max_num_ref_frames'], 1)

    def sg(d):
        while d is not None and d in sigma:
            d = sigma[d]
        return d

    refs = []       # {'dec': original id, 'frame_num': new, 'poc': new}
    prev_ref_fn = 0
    out = []
    for e in entries:
        p = e['pic']
        sl = p['slices'][0]
        t = sl['slice_type']
        cur = p['dec']
        if e['idr']:
            refs, fn = [], 0
        else:
            fn = (prev_ref_fn + 1) % max_fn
        have = {r['dec'] for r in refs}
        if e.get('demote'):
            want = [[], []]
        else:
            want = [[sg(x) for x in p['lists'][l]] for l in range(2)]
            if None in want[0] or None in want[1]:
                raise PlanError(f'picture {cur}: reads a dropped picture that has no substitute')
        for l in range(2):
            missing = [x for x in want[l] if x not in have]
            if missing:
                raise PlanError(f'picture {cur}: list {l} needs {missing}, not in the DPB')
        n_act = (sl['num_ref_l0'], sl['num_ref_l1'])
        init = init_lists(t, refs, fn, e['poc'], n_act[0], n_act[1], max_fn)
        rplm = [_plan_list(init[l], want[l], sl['rplm'][l], n_act[l], fn, refs, max_fn) if want[l] else []
                for l in range(2)]
        h = {'frame_num': fn, 'poc_lsb': e['poc'] % max_lsb, 'rplm': rplm, 'mmco': None,
             'demote': bool(e.get('demote')), 'lists': want}
        if p['ref_idc']:
            src = e.get('as', p)
            target = {sg(x) for x in src['dpb_after']} - {None}
            if cur not in target:
                raise PlanError(f'picture {cur}: not in its own target DPB')
            if not target <= have | {cur}:
                raise PlanError(f'picture {cur}: target DPB needs {sorted(target - have - {cur})}')
            if len(target) > max_refs:
                raise PlanError(f'picture {cur}: target DPB too large')

            def num(r):
                return r['frame_num'] - max_fn if r['frame_num'] > fn else r['frame_num']
            if not e['idr']:
                slide = list(refs)
                if len(slide) >= max_refs:
                    slide.remove(min(slide, key=num))
                if {r['dec'] for r in slide} | {cur} == target and not sl.get('adaptive'):
                    h['mmco'] = None
                else:
                    # unmark what the target lacks, in the original's op order
                    # where the picNums still match (untouched headers stay
                    # byte-identical)
                    order = {p['frame_num'] - (args[0] + 1): i for i, (op, *args) in enumerate(sl['mmco'])}
                    gone = sorted((r for r in refs if r['dec'] not in target),
                                  key=lambda r: (order.get(num(r), len(order)), -num(r)))
                    h['mmco'] = [(1, fn - num(r) - 1) for r in gone]
                refs = [r for r in refs if r['dec'] in target]
            refs.append({'dec': cur, 'frame_num': fn, 'poc': e['poc']})
            prev_ref_fn = fn
        h['same'] = bool(not h['demote'] and fn == sl['frame_num'] and h['poc_lsb'] == sl['poc_lsb']
                     and [list(c) for c in rplm] == [list(c) for c in sl['rplm']]
                     and (not p['ref_idc'] or e['idr'] or
                          (h['mmco'] is None and not sl.get('adaptive')) or
                          (h['mmco'] is not None and sl.get('adaptive') and h['mmco'] == sl['mmco'])))
        out.append(h)
    check(spss, entries, out)
    return out


def check(spss, entries, hdrs):
    """Re-simulate the planned headers with dpb_walk: every list, POC and
    DPB must be what the plan asked for."""
    shadow = []
    for e, h in zip(entries, hdrs):
        p = e['pic']
        sl = dict(p['slices'][0], frame_num=h['frame_num'], poc_lsb=h['poc_lsb'], rplm=h['rplm'],
                  mmco=list(h['mmco'] or []), adaptive=int(h['mmco'] is not None))
        if h['demote']:
            sl['nal_type'] = 1
        shadow.append({'slices': [sl], 'idr': e['idr'], 'ref_idc': p['ref_idc'], 'src': p['dec']})
    try:
        dpb_walk(spss, shadow)
    except ValueError as ex:
        raise PlanError(f'rewritten headers do not simulate: {ex}')
    for e, h, q in zip(entries, hdrs, shadow):
        if q['poc'] != e['poc']:
            raise PlanError(f"picture {e['pic']['dec']}: POC {q['poc']}, wanted {e['poc']}")
        got = [[shadow[i]['src'] for i in q['lists'][l]] for l in range(2)]
        if got != h['lists']:
            raise PlanError(f"picture {e['pic']['dec']}: lists {got}, wanted {h['lists']}")
    return shadow


# ---------------------------------------------------------------- stream level

def free_checker(pics):
    """free(dec): a non-IDR reference picture that no picture's list names.
    Dropping it (sigma None) leaves every other picture bit-exact."""
    listed = {x for p in pics for l in p['lists'] for x in l}
    return lambda d: not pics[d]['idr'] and pics[d]['ref_idc'] and d not in listed


def build_entries(pics, drop, rebuilds):
    """The new stream in decode order as plan() entries, for `drop` (decode
    ids, including every rebuild's K) and anchor rebuilds [(K, D)]: D is
    demoted and written in K's slot with K's POC ("I2"), every POC of D's
    former IDR period is shifted by POC(K) - POC(D), and the periods merge.
    Returns entries; entry['pic']['dec'] is the original of each output
    picture."""
    byK = dict(rebuilds)
    byD = {D: K for K, D in rebuilds}
    drop = set(drop)
    out, off, i2poc = [], 0, {}
    for p in pics:
        d = p['dec']
        if d in byD:
            off = i2poc[d] - p['poc']
            continue
        if p['idr']:
            off = 0
        if d in byK:
            D = pics[byK[d]]
            poc = p['poc'] + off
            i2poc[D['dec']] = poc
            out.append({'pic': D, 'idr': False, 'poc': poc, 'demote': True, 'as': p})
            continue
        if d in drop:
            continue
        out.append({'pic': p, 'idr': p['idr'], 'poc': p['poc'] + off})
    return out


def plan_stream(spss, pics, drop, sigma, rebuilds):
    """Header plan for the whole stream. sigma: {dropped reference: substitute
    or None}; rebuilds: [(K, D)] (sigma[K] = D is added). Returns (entries,
    headers)."""
    sig = dict(sigma)
    for K, D in rebuilds:
        sig[K] = D
    entries = build_entries(pics, set(drop) | {K for K, _ in rebuilds}, rebuilds)
    hdrs = []
    # plan per new IDR period (independent, and errors name the period)
    start = 0
    for i in range(1, len(entries) + 1):
        if i == len(entries) or entries[i]['idr']:
            hdrs += plan(spss, entries[start:i], sig, None)
            start = i
    return entries, hdrs


def candidate(spss, pics, periods, K, D):
    """Explicit-mode anchor rebuild eligibility: D an IDR, K a non-IDR
    reference picture in the period before it, and the merged periods' headers
    plannable. Every picture between K and D in decode order is a leading
    picture (K's period ends there and D flushed the DPB, so nothing else can
    read K). Returns (rebuild dict or None, reason)."""
    pK, pD = pics[K], pics[D]
    if not pD['idr'] or pK['idr'] or not pK['ref_idc']:
        return None, 'twin types'
    if pK['period'] != pD['period'] - 1:
        return None, 'twins not in adjacent IDR periods'
    seg = periods[pK['period']] + periods[pD['period']]
    try:
        plan_stream(spss, _Renumbered(seg), set(), {}, [(K, D)])
    except PlanError as e:
        return None, f'headers: {e}'
    return {'K': K, 'D': D, 'B': list(range(K + 1, D))}, None


class _Renumbered(list):
    """A run of pictures indexable by their original decode ids."""

    def __init__(self, seg):
        super().__init__(seg)
        self.base = seg[0]['dec']

    def __getitem__(self, d):
        if isinstance(d, slice):
            return list.__getitem__(self, d)
        return list.__getitem__(self, d - self.base)

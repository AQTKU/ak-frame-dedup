"""Patch macroblocks in an H.264 stream with I_PCM copies of target pixels.

Everything else in a patched picture is held fixed:
  * P_Skip MBs whose inferred motion changes because a neighbour became
    intra are rewritten as explicit P_L0_16x16 with the original motion,
  * explicit partitions get their mvd re-solved against the new neighbours,
  * the QP chain is preserved (PCM passes QP_pred through, so a following MB
    that does not code mb_qp_delta is given an empty coded chroma DC so it
    can carry one),
  * B pictures whose co-located picture got PCM MBs have the affected direct
    blocks rewritten as explicit bi-predicted 8x8 blocks with the original
    temporal-direct motion.

Usage: python patch.py <in.h264> <patches.json> <yuv dir> <out.h264>
patches.json: {"<decode idx>": [[mbx, mby], ...], ...}
yuv dir holds <decode idx>.yuv (yuv420p 8-bit frames with the target pixels).
"""
import copy
import json
import mmap
import os
import sys

from .h264hdr import parse_stream
from .h264mv import CompactMotion, derive_motion, dpb_walk
from .roundtrip_test import decode_picture, encode_picture
from .verify_motion import has_direct, reexpress, reexpress_compact


def pcm_record(yuv, W, mbx, mby):
    """yuv: an uncropped (coded-size) yuv420p frame."""
    w, h = W * 16, len(yuv) * 2 // 3 // (W * 16)
    Y = yuv[:w * h]
    U = yuv[w * h:w * h + w * h // 4]
    V = yuv[w * h + w * h // 4:]
    s = []
    for y in range(16):
        o = (mby * 16 + y) * w + mbx * 16
        s.extend(Y[o:o + 16])
    for P in (U, V):
        for y in range(8):
            o = (mby * 8 + y) * (w // 2) + mbx * 8
            s.extend(P[o:o + 8])
    return {'skip': 0, 'kind': 'PCM', 'samples': list(s), 'intra': 1, 'pcm': 1, 'i16': 0,
            'direct16': 0, 'cbp': 0x2f, 't8': 0, 'cmode': 0, 'qpd': 0}


def copy_mb(mb):
    """A copy of an MB record that is safe to modify the way the patcher
    and derive_motion's target mode do (top-level fields, refs, mvds);
    coefficients and samples are shared. Much cheaper than deepcopy."""
    n = dict(mb)
    if mb.get('mvds') is not None:
        n['mvds'] = [None if l is None else [list(x) if isinstance(x, list) else x for x in l] for l in mb['mvds']]
    if mb.get('refs') is not None:
        n['refs'] = [None if l is None else list(l) for l in mb['refs']]
    return n


def same_motion(a, b, mbx, mby):
    for y in range(mby * 4, mby * 4 + 4):
        for x in range(mbx * 4, mbx * 4 + 4):
            for l in range(2):
                if a.refidx[l][y][x] != b.refidx[l][y][x]:
                    return False
                if a.refidx[l][y][x] >= 0 and a.mv[l][y][x] != b.mv[l][y][x]:
                    return False
    return True


def qp_chain(mbs, slice_qp):
    out, pred = [], slice_qp
    for mb in mbs:
        if mb['skip'] or mb['kind'] == 'PCM' or (mb['cbp'] == 0 and mb['kind'] != 'I16'):
            q = pred
        else:
            q = (pred + mb['qpd'] + 52) % 52
        out.append(q)
        pred = q
    return out


def fix_qp(new, target, slice_qp):
    """Re-solve mb_qp_delta so every non-PCM MB keeps its original QP."""
    forced, pred = 0, slice_qp
    for mb, tq in zip(new, target):
        if mb['kind'] == 'PCM':
            continue
        codes = not mb['skip'] and (mb['cbp'] != 0 or mb['kind'] == 'I16')
        if not codes and tq != pred:
            # make it carry a qp delta: empty coded chroma DC (no coefficients)
            assert not mb['skip'], 'skip MB needing QP change must be converted first'
            mb['cbp'] |= 0x10
            mb.setdefault('coef', {})
            codes = True
            forced += 1
        if codes:
            mb['qpd'] = ((tq - pred + 26) % 52) - 26
            pred = tq
    return forced


def to_explicit(mb, t, motion, compact=False):
    """Rewrite a skip/direct MB with explicit partitions carrying `motion`
    (B: explicit 8x8 blocks, or with `compact` the coarsest partitioning
    that fits)."""
    if t == 'P':
        n = copy.deepcopy(mb)
        n.update(skip=0, kind='P', part='16x16', refs=[[0], [None]], mvds=[[(0, 0)], [None]],
                 direct16=0)
        if mb['skip']:
            n.update(cbp=0, t8=0, qpd=0, coef={})
        return n
    return reexpress_compact(mb, motion) if compact else reexpress(mb, 'B', motion)


def main(es, patch_json, yuv_dir, out_path):
    patches = {int(k): [tuple(v) for v in vs] for k, vs in json.load(open(patch_json)).items()}

    def yuv_of(d):
        return open(os.path.join(yuv_dir, f'{d}.yuv'), 'rb').read()
    patch_stream(es, patches, yuv_of, out_path)


def patch_stream(es, patches, yuv_of, out_path, log=print):
    """patches: {decode idx: [(mbx, mby), ...]}; yuv_of(d) -> uncropped yuv420p
    bytes holding the target pixels of picture d."""
    spss, ppss, pics = parse_stream(es)
    dpb_walk(spss, pics)
    f = open(es, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    repl = patch_pictures(mm, spss, ppss, pics, patches, yuv_of, log)
    write_patched(mm, pics, repl, out_path)
    log(f'wrote {out_path}: {len(repl)} pictures rewritten')
    return repl


def _has(pics, k):
    return k in pics if isinstance(pics, dict) else 0 <= k < len(pics)


def patch_pictures(mm, spss, ppss, pics, patches, yuv_of, log=print, parsed=None, stats=None, force=(),
                   cache=None):
    """Compute replacement slice NALs for the patched pictures. yuv_of(d)
    gives the target pixels: an uncropped yuv420p frame, or a dict
    {(mbx, mby): pcm_record} holding just the patched MBs.
    pics: pictures indexable by decode index (a list, or a dict holding at
    least the patched pictures, their reference pictures and following B
    pictures), after dpb_walk. parsed: optional cache {dec: (decoded picture,
    motion to keep)}. force: pictures to check even without PCM targets
    (e.g. B pictures whose co-located picture changed); a picture whose
    records need no change is not rewritten. cache: a dict kept by the caller
    across calls with growing `patches`; a picture whose own patch set and
    co-located picture's state are unchanged is not re-encoded. Returns
    {dec: new NAL bytes}."""
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']

    # pictures to rewrite: patched ones, plus B pictures that use a patched
    # reference picture as co-located picture (they follow it in decode order)
    todo = set(patches) | set(force)
    keys = sorted(pics) if isinstance(pics, dict) else range(len(pics))
    for d in patches:
        for k in keys:
            if k > d and pics[k]['slices'][0]['slice_type'] == 'B' and pics[k]['lists'][1][0] == d:
                todo.add(k)

    new_motion, repl, sigs = {}, {}, {}
    parsed = {} if parsed is None else parsed

    def load(d):
        if d not in parsed:
            p = pics[d]
            t = p['slices'][0]['slice_type']
            dd = decode_picture(mm, spss, ppss, p)
            dd['bits'] = dd['bits'][:p['slices'][0]['data_bit']]  # re-encoding needs only the header
            col = p['lists'][1][0] if t == 'B' else None
            cm = load(col)[1] if t == 'B' else None
            parsed[d] = (dd, derive_motion(p, dd['mbs'], W, H, pics, cm))
        return parsed[d]

    for d in sorted(todo):
        p = pics[d]
        t = p['slices'][0]['slice_type']
        col = p['lists'][1][0] if t == 'B' else None
        targets = set(patches.get(d, []))
        col_pcm = set(patches.get(col, [])) if t == 'B' else set()
        # nothing this picture depends on changed since the cached result:
        # reuse it without parsing anything
        sig = (tuple(sorted(targets)), sigs.get(col))
        sigs[d] = sig
        if cache is not None and d in cache and cache[d]['sig'] == sig:
            c = cache[d]
            if c['motion'] is not None:
                new_motion[d] = c['motion']
            if c['nal'] is not None:
                repl[d] = c['nal']
                if stats is not None:
                    stats[d] = c['stats']
            continue
        if not targets and not (col_pcm and col in repl) and d not in force:
            if cache is not None:
                cache[d] = {'sig': sig, 'motion': None, 'nal': None}
            continue
        dd, m0 = load(d)
        if t == 'B':
            load(col)
            new_motion.setdefault(col, parsed[col][1])
        new = [copy_mb(mb) for mb in dd['mbs']]
        yuv = yuv_of(d) if targets else None
        for i, mb in enumerate(new):
            if (mb['x'], mb['y']) in targets:
                if isinstance(yuv, dict):   # {(mbx, mby): PCM record} of just the patched MBs
                    r = dict(yuv[(mb['x'], mb['y'])])
                else:
                    r = pcm_record(yuv, W, mb['x'], mb['y'])
                r.update(x=mb['x'], y=mb['y'])
                new[i] = r
        n_direct = 0
        compact = bool(p['slices'][0].get('direct_spatial'))
        if t == 'B':
            # direct blocks whose co-located MB became intra (PCM)
            for i, mb in enumerate(new):
                if mb['kind'] == 'PCM' or mb['intra']:
                    continue
                if (mb['x'], mb['y']) not in col_pcm:
                    continue
                if mb['skip'] or mb.get('part') == 'D' or (mb.get('part') == '8x8' and 0 in mb['sub']):
                    new[i] = to_explicit(mb, 'B', m0, compact)
                    n_direct += 1
        # skip / direct MBs whose inferred motion moved (a neighbour became
        # intra, or the co-located motion changed): make explicit, repeat
        # until stable
        n_skip = 0
        while True:
            # target mode: MBs already made explicit carry the motion to keep
            m1 = derive_motion(p, new, W, H, pics, new_motion.get(col), target=m0)
            changed = [i for i, mb in enumerate(new) if mb['kind'] != 'PCM' and not mb['intra']
                       and not same_motion(m0, m1, mb['x'], mb['y'])]
            skips = [i for i in changed if new[i]['skip'] or (t == 'B' and has_direct(new[i]))]
            if not skips:
                break
            for i in skips:
                if new[i]['skip']:
                    n_skip += 1
                else:
                    n_direct += 1
                new[i] = to_explicit(new[i], t, m0, compact)
        # a skip MB inherits QP_pred; if a PCM in front of it changed QP_pred,
        # it has to become a coded MB so it can carry mb_qp_delta
        target_qp = qp_chain(dd['mbs'], p['slices'][0]['qp'])
        pred = p['slices'][0]['qp']
        n_qp = 0
        for i, mb in enumerate(new):
            if mb['kind'] == 'PCM':
                continue
            if mb['skip'] and target_qp[i] != pred:
                new[i] = to_explicit(mb, t, m0, compact)
                n_qp += 1
            pred = target_qp[i]
        # solve mvds so explicit partitions carry the original motion
        m2 = derive_motion(p, new, W, H, pics, new_motion.get(col), target=m0) if n_qp else m1
        bad = [(mb['x'], mb['y']) for mb in new if mb['kind'] != 'PCM' and not mb['intra']
               and not same_motion(m0, m2, mb['x'], mb['y'])]
        assert not bad, f'dec {d} {t}: motion could not be preserved at {bad[:5]}: ' + str(
            [{k: v for k, v in mb.items() if k not in ('coef', 'samples')} for mb in new
             if (mb['x'], mb['y']) == bad[0]])
        forced = fix_qp(new, target_qp, p['slices'][0]['qp'])
        # the records now carry exactly this motion; only reference pictures
        # can be co-located pictures, and a compact copy is enough for that
        cm2 = CompactMotion(m2) if p['ref_idc'] else None
        if cm2 is not None:
            new_motion[d] = cm2
        if not targets and not n_direct and not n_skip and not forced and new == dd['mbs']:
            if cache is not None:
                cache[d] = {'sig': sig, 'motion': cm2, 'nal': None}
            continue
        nal, _ = encode_picture(dd, new)
        repl[d] = nal
        log(f"dec {d} {t}: PCM {len(targets)} MB, direct->explicit {n_direct}, skip->explicit {n_skip}, "
              f"qp-carrier {forced}; {len(dd['nal'])}B -> {len(nal)}B")
        st = dict(type=t, pcm=len(targets), direct_explicit=n_direct, skip_explicit=n_skip,
                  qp_carrier=forced, bytes_before=len(dd['nal']), bytes_after=len(nal))
        if stats is not None:
            stats[d] = st
        if cache is not None:
            cache[d] = {'sig': sig, 'motion': cm2, 'nal': nal, 'stats': st}
    return repl


def write_patched(mm, pics, repl, out_path):
    with open(out_path, 'wb') as o:
        pos = 0
        for p in pics:
            sl = p['slices'][0]
            s0 = sl['offset'] - 3
            if mm[s0 - 1] == 0:
                s0 -= 1
            o.write(mm[pos:s0])
            if p['dec'] in repl:
                o.write(b'\x00\x00\x00\x01' + repl[p['dec']])
            else:
                o.write(mm[s0:sl['offset'] + sl['size']])
            pos = sl['offset'] + sl['size']
        o.write(mm[pos:])


if __name__ == '__main__':
    main(*sys.argv[1:5])

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
from .h264mv import derive_motion, dpb_walk
from .roundtrip_test import decode_picture, encode_picture
from .verify_motion import reexpress


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


def to_explicit(mb, t, motion):
    """Rewrite a skip/direct MB with explicit partitions carrying `motion`."""
    if t == 'P':
        n = copy.deepcopy(mb)
        n.update(skip=0, kind='P', part='16x16', refs=[[0], [None]], mvds=[[(0, 0)], [None]],
                 direct16=0)
        if mb['skip']:
            n.update(cbp=0, t8=0, qpd=0, coef={})
        return n
    return reexpress(mb, 'B', motion)


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


def patch_pictures(mm, spss, ppss, pics, patches, yuv_of, log=print, parsed=None, stats=None):
    """Compute replacement slice NALs for the patched pictures.
    pics: pictures indexable by decode index (a list, or a dict holding at
    least the patched pictures, their reference pictures and following B
    pictures), after dpb_walk. parsed: optional cache {dec: (decoded picture,
    motion)}. Returns {dec: new NAL bytes}."""
    sps = spss[0]
    W, H = sps['mb_w'], sps['mb_h']

    # pictures to rewrite: patched ones, plus B pictures that use a patched
    # P picture as co-located picture (they follow it in decode order)
    todo = set(patches)
    for d in patches:
        if pics[d]['slices'][0]['slice_type'] == 'P':
            k = d + 1
            while _has(pics, k) and pics[k]['slices'][0]['slice_type'] == 'B':
                if pics[k]['lists'][1][0] == d:
                    todo.add(k)
                k += 1

    new_motion, repl = {}, {}
    parsed = {} if parsed is None else parsed

    def load(d):
        if d not in parsed:
            p = pics[d]
            t = p['slices'][0]['slice_type']
            dd = decode_picture(mm, spss, ppss, p)
            col = p['lists'][1][0] if t == 'B' else None
            cm = load(col)[1] if t == 'B' else None
            parsed[d] = (dd, derive_motion(p, dd['mbs'], W, H, pics, cm))
        return parsed[d]

    for d in sorted(todo):
        p = pics[d]
        t = p['slices'][0]['slice_type']
        dd, m0 = load(d)
        col = p['lists'][1][0] if t == 'B' else None
        if t == 'B':
            load(col)
            new_motion.setdefault(col, parsed[col][1])
        targets = set(patches.get(d, []))
        col_pcm = set(patches.get(col, [])) if t == 'B' else set()
        if not targets and not (col_pcm and col in repl):
            continue
        new = copy.deepcopy(dd['mbs'])
        yuv = yuv_of(d) if targets else None
        for i, mb in enumerate(new):
            if (mb['x'], mb['y']) in targets:
                r = pcm_record(yuv, W, mb['x'], mb['y'])
                r.update(x=mb['x'], y=mb['y'])
                new[i] = r
        n_direct = 0
        if t == 'B':
            # direct blocks whose co-located MB became intra (PCM)
            for i, mb in enumerate(new):
                if mb['kind'] == 'PCM' or mb['intra']:
                    continue
                if (mb['x'], mb['y']) not in col_pcm:
                    continue
                if mb['skip'] or mb.get('part') == 'D' or (mb.get('part') == '8x8' and 0 in mb['sub']):
                    new[i] = to_explicit(mb, 'B', m0)
                    n_direct += 1
        # P_Skip whose inferred motion moved: make explicit, repeat until stable
        n_skip = 0
        while True:
            m1 = derive_motion(p, new, W, H, pics, new_motion.get(col))
            changed = [i for i, mb in enumerate(new) if mb['kind'] != 'PCM' and not mb['intra']
                       and not same_motion(m0, m1, mb['x'], mb['y'])]
            skips = [i for i in changed if new[i]['skip']]
            if not skips:
                break
            for i in skips:
                new[i] = to_explicit(new[i], t, m0)
                n_skip += 1
        # a skip MB inherits QP_pred; if a PCM in front of it changed QP_pred,
        # it has to become a coded MB so it can carry mb_qp_delta
        target_qp = qp_chain(dd['mbs'], p['slices'][0]['qp'])
        pred = p['slices'][0]['qp']
        for i, mb in enumerate(new):
            if mb['kind'] == 'PCM':
                continue
            if mb['skip'] and target_qp[i] != pred:
                new[i] = to_explicit(mb, t, m0)
            pred = target_qp[i]
        # solve mvds so explicit partitions carry the original motion
        m2 = derive_motion(p, new, W, H, pics, new_motion.get(col), target=m0)
        bad = [(mb['x'], mb['y']) for mb in new if mb['kind'] != 'PCM' and not mb['intra']
               and not same_motion(m0, m2, mb['x'], mb['y'])]
        assert not bad, f'dec {d} {t}: motion could not be preserved at {bad[:5]}: ' + str(
            [{k: v for k, v in mb.items() if k not in ('coef', 'samples')} for mb in new
             if (mb['x'], mb['y']) == bad[0]])
        forced = fix_qp(new, target_qp, p['slices'][0]['qp'])
        new_motion[d] = derive_motion(p, new, W, H, pics, new_motion.get(col))
        nal, _ = encode_picture(dd, new)
        repl[d] = nal
        log(f"dec {d} {t}: PCM {len(targets)} MB, direct->explicit {n_direct}, skip->explicit {n_skip}, "
              f"qp-carrier {forced}; {len(dd['nal'])}B -> {len(nal)}B")
        if stats is not None:
            stats[d] = dict(type=t, pcm=len(targets), direct_explicit=n_direct, skip_explicit=n_skip,
                            qp_carrier=forced, bytes_before=len(dd['nal']), bytes_after=len(nal))
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

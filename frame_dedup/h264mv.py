"""Reference structure and motion reconstruction on top of h264cabac.

* `dpb_walk` simulates the decoded picture buffer from slice headers only and
  attaches POC and initial reference lists (as decode-order picture ids) to
  every picture. Sliding-window marking only; no MMCO, no long-term refs, no
  ref list modification (asserted).
* `derive_motion` turns a parsed picture (h264cabac records) into per-4x4
  reference picture ids and motion vectors, including P_Skip and temporal
  direct (B_Skip / B_Direct_16x16 / B_Direct_8x8, direct_8x8_inference).
"""
from .h264cabac import B_SUB, B_TYPES, P_SUB, mb_parts, sub_parts, uses


def dpb_walk(spss, pics):
    sps = spss[0]
    max_fn = 1 << sps['log2_max_frame_num']
    max_lsb = 1 << sps['log2_max_poc_lsb']
    refs = []  # short-term reference pictures (dicts)
    prev_msb = prev_lsb = 0
    for d, p in enumerate(pics):
        sl = p['slices'][0]
        assert not sl['rplm'][0] and not sl['rplm'][1] and not sl['mmco']
        p['dec'] = d
        if p['idr']:
            refs = []
            prev_msb = prev_lsb = 0
        lsb = sl['poc_lsb']
        if lsb < prev_lsb and prev_lsb - lsb >= max_lsb // 2:
            msb = prev_msb + max_lsb
        elif lsb > prev_lsb and lsb - prev_lsb > max_lsb // 2:
            msb = prev_msb - max_lsb
        else:
            msb = prev_msb
        top = msb + lsb
        p['poc'] = min(top, top + sl.get('delta_poc_bottom', 0))
        p['frame_num'] = sl['frame_num']
        # initial reference lists
        t = sl['slice_type']
        p['lists'] = [[], []]
        if t == 'P':
            def picnum(r):
                fn = r['frame_num']
                return fn - max_fn if fn > sl['frame_num'] else fn
            p['lists'][0] = [r['dec'] for r in sorted(refs, key=picnum, reverse=True)][:sl['num_ref_l0']]
        elif t == 'B':
            before = sorted((r for r in refs if r['poc'] < p['poc']), key=lambda r: -r['poc'])
            after = sorted((r for r in refs if r['poc'] > p['poc']), key=lambda r: r['poc'])
            l0 = [r['dec'] for r in before + after]
            l1 = [r['dec'] for r in after + before]
            if len(l1) > 1 and l1 == l0:
                l1[0], l1[1] = l1[1], l1[0]
            p['lists'] = [l0[:sl['num_ref_l0']], l1[:sl['num_ref_l1']]]
        # reference marking (sliding window)
        if p['ref_idc']:
            if not p['idr'] and len(refs) >= max(sps['max_num_ref_frames'], 1):
                def wrap(r):
                    fn = r['frame_num']
                    return fn - max_fn if fn > sl['frame_num'] else fn
                refs.remove(min(refs, key=wrap))
            refs.append(p)
            prev_msb, prev_lsb = msb, lsb
    return pics


def _cdiv(a, b):
    """C integer division (truncate toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _median(a, b, c):
    return max(min(a, b), min(max(a, b), c))


class Motion:
    """Per-4x4 motion field of one picture: ref picture ids and mvs."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        W4, H4 = W * 4, H * 4
        self.refidx = [[[-1] * W4 for _ in range(H4)] for _ in range(2)]
        self.mv = [[[(0, 0)] * W4 for _ in range(H4)] for _ in range(2)]
        self.refpic = [[[None] * W4 for _ in range(H4)] for _ in range(2)]
        self.intra = [[0] * W4 for _ in range(H4)]
        self.direct = [[0] * W4 for _ in range(H4)]


def derive_motion(pic, mbs, W, H, pics, col_motion=None, target=None):
    """Reconstruct motion for a parsed picture. col_motion: Motion of
    RefPicList1[0] (needed for B pictures with direct blocks).

    With `target` (a Motion), explicit partitions take their mvs from the
    target field and their mvd is solved (written into the records), so the
    records encode exactly that motion under the current neighbourhood."""
    t = pic['slices'][0]['slice_type']
    lists = pic['lists']
    m = Motion(W, H)
    pic['motion'] = m
    if t == 'I':
        for row in m.intra:
            for x in range(len(row)):
                row[x] = 1
        return m
    done = set()  # 4x4 blocks of the current MB already assigned

    def avail(gx, gy, cmx, cmy):
        if gx < 0 or gy < 0 or gx >= W * 4 or gy >= H * 4:
            return False
        mx, my = gx // 4, gy // 4
        if (mx, my) == (cmx, cmy):
            return (gx, gy) in done
        return my < cmy or (my == cmy and mx < cmx)

    def nbr(gx, gy, lst, cmx, cmy):
        if not avail(gx, gy, cmx, cmy):
            return None
        return m.refidx[lst][gy][gx], m.mv[lst][gy][gx]

    def neighbours(gx, gy, w, lst, cmx, cmy):
        A = nbr(gx - 1, gy, lst, cmx, cmy)
        B = nbr(gx, gy - 1, lst, cmx, cmy)
        C = nbr(gx + w, gy - 1, lst, cmx, cmy)
        if C is None:
            C = nbr(gx - 1, gy - 1, lst, cmx, cmy)
        if B is None and C is None and A is not None:
            B = C = A
        z = (-1, (0, 0))
        return A or z, B or z, C or z

    def mvp(gx, gy, w, lst, ref, cmx, cmy, shape=None, part=0):
        A, B, C = neighbours(gx, gy, w, lst, cmx, cmy)
        if shape == '16x8':
            if part == 0 and B[0] == ref:
                return B[1]
            if part == 1 and A[0] == ref:
                return A[1]
        elif shape == '8x16':
            if part == 0 and A[0] == ref:
                return A[1]
            if part == 1 and C[0] == ref:
                return C[1]
        match = [n for n in (A, B, C) if n[0] == ref]
        if len(match) == 1:
            return match[0][1]
        return (_median(A[1][0], B[1][0], C[1][0]), _median(A[1][1], B[1][1], C[1][1]))

    def setblk(blocks, lst, ref, mv, direct=0):
        for (gx, gy) in blocks:
            m.refidx[lst][gy][gx] = ref
            m.mv[lst][gy][gx] = mv
            m.refpic[lst][gy][gx] = lists[lst][ref] if ref >= 0 else None
            m.direct[gy][gx] = direct

    def temporal_direct(i8, cmx, cmy):
        """Temporal direct for one 8x8 quadrant (direct_8x8_inference)."""
        cur = pic
        col = pics[lists[1][0]]
        cm = col_motion
        ox, oy = cmx * 4 + (i8 % 2) * 2, cmy * 4 + (i8 // 2) * 2
        # corner 4x4 of the co-located MB for this quadrant
        cx = cmx * 4 + (3 if i8 % 2 else 0)
        cy = cmy * 4 + (3 if i8 // 2 else 0)
        blocks = [(ox + x, oy + y) for y in range(2) for x in range(2)]
        if cm.intra[cy][cx]:
            mvcol, refcol_pic = (0, 0), None
        elif cm.refidx[0][cy][cx] >= 0:
            mvcol, refcol_pic = cm.mv[0][cy][cx], cm.refpic[0][cy][cx]
        else:
            mvcol, refcol_pic = cm.mv[1][cy][cx], cm.refpic[1][cy][cx]
        ref0 = 0 if refcol_pic is None else lists[0].index(refcol_pic)
        poc0 = pics[lists[0][ref0]]['poc']
        poc1 = col['poc']
        tb = max(-128, min(127, cur['poc'] - poc0))
        td = max(-128, min(127, poc1 - poc0))
        if td == 0:
            mv0, mv1 = mvcol, (0, 0)
        else:
            tx = _cdiv(16384 + abs(td) // 2, td)
            dsf = max(-1024, min(1023, (tb * tx + 32) >> 6))
            mv0 = ((dsf * mvcol[0] + 128) >> 8, (dsf * mvcol[1] + 128) >> 8)
            mv1 = (mv0[0] - mvcol[0], mv0[1] - mvcol[1])
        setblk(blocks, 0, ref0, mv0, 1)
        setblk(blocks, 1, 0, mv1, 1)
        done.update(blocks)

    for mb in mbs:
        cmx, cmy = mb['x'], mb['y']
        x0, y0 = cmx * 4, cmy * 4
        done.clear()
        allb = [(x0 + x, y0 + y) for y in range(4) for x in range(4)]
        if mb['intra']:
            for (gx, gy) in allb:
                m.intra[gy][gx] = 1
            for lst in range(2):
                setblk(allb, lst, -1, (0, 0))
            continue
        if mb['skip'] and t == 'P':
            A = nbr(x0 - 1, y0, 0, cmx, cmy)
            B = nbr(x0, y0 - 1, 0, cmx, cmy)
            if A is None or B is None or (A[0] == 0 and A[1] == (0, 0)) or (B[0] == 0 and B[1] == (0, 0)):
                mv = (0, 0)
            else:
                mv = mvp(x0, y0, 4, 0, 0, cmx, cmy)
            setblk(allb, 0, 0, mv)
            setblk(allb, 1, -1, (0, 0))
            continue
        if mb['skip'] or mb.get('part') == 'D':
            for i8 in range(4):
                temporal_direct(i8, cmx, cmy)
            continue
        if mb['part'] != '8x8':
            parts = mb_parts(mb['part'])
            preds = ('L0',) * len(parts) if t == 'P' else B_TYPES[mb['btype']][1]
            for p, blocks in enumerate(parts):
                gb = [(x0 + x, y0 + y) for (x, y) in blocks]
                w = 4 if mb['part'] in ('16x16', '16x8') else 2
                for lst in range(2):
                    if t == 'P' and lst == 1 or not uses(preds[p], lst):
                        setblk(gb, lst, -1, (0, 0))
                        continue
                    ref = mb['refs'][lst][p]
                    pm = mvp(gb[0][0], gb[0][1], w, lst, ref, cmx, cmy, mb['part'], p)
                    if target is not None:
                        tv = target.mv[lst][gb[0][1]][gb[0][0]]
                        mb['mvds'][lst][p] = (tv[0] - pm[0], tv[1] - pm[1])
                    d = mb['mvds'][lst][p]
                    setblk(gb, lst, ref, (pm[0] + d[0], pm[1] + d[1]))
                done.update(gb)
            continue
        # 8x8 partitions
        for i in range(4):
            st = mb['sub'][i]
            shape, pred = (P_SUB[st], 'L0') if t == 'P' else B_SUB[st]
            ox, oy = x0 + (i % 2) * 2, y0 + (i // 2) * 2
            if shape == 'D':
                temporal_direct(i, cmx, cmy)
                continue
            for j, blocks in enumerate(sub_parts(shape)):
                gb = [(ox + x, oy + y) for (x, y) in blocks]
                w = 2 if shape in ('8x8', '8x4') else 1
                for lst in range(2):
                    if (t == 'P' and lst == 1) or not uses(pred, lst):
                        setblk(gb, lst, -1, (0, 0))
                        continue
                    ref = mb['refs'][lst][i]
                    pm = mvp(gb[0][0], gb[0][1], w, lst, ref, cmx, cmy)
                    if target is not None:
                        tv = target.mv[lst][gb[0][1]][gb[0][0]]
                        mb['mvds'][lst][i][j] = (tv[0] - pm[0], tv[1] - pm[1])
                    d = mb['mvds'][lst][i][j]
                    setblk(gb, lst, ref, (pm[0] + d[0], pm[1] + d[1]))
                done.update(gb)
    return m

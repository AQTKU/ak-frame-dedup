"""Reference structure and motion reconstruction on top of h264cabac.

* `dpb_walk` simulates the decoded picture buffer from slice headers only and
  attaches POC and final reference lists (as decode-order picture ids) to
  every picture: sliding-window marking or MMCO 1, and short-term reference
  list modification. No long-term references.
* `derive_motion` turns a parsed picture (h264cabac records) into per-4x4
  reference picture ids and motion vectors, including P_Skip and temporal or
  spatial direct (B_Skip / B_Direct_16x16 / B_Direct_8x8,
  direct_8x8_inference).
"""
from .h264cabac import B_SUB, B_TYPES, P_SUB, mb_parts, sub_parts, uses


def modify_list(lst, cmds, n_active, cur_fn, refs, max_fn):
    """Apply ref_pic_list_modification (short-term commands, 8.2.4.3.1) to an
    initial list of decode ids. refs: the short-term reference pictures."""
    def picnum(r):
        fn = r['frame_num']
        return fn - max_fn if fn > cur_fn else fn
    by_num = {picnum(r): r['dec'] for r in refs}
    lst = (lst + [None] * (n_active + 1))[:n_active + 1]
    pred = cur_fn
    idx = 0
    for idc, v in cmds:
        if idc == 0:
            nw = pred - (v + 1)
            if nw < 0:
                nw += max_fn
        elif idc == 1:
            nw = pred + v + 1
            if nw >= max_fn:
                nw -= max_fn
        else:
            raise ValueError(f'ref list modification idc {idc} (long-term) is not supported')
        pred = nw
        num = nw - max_fn if nw > cur_fn else nw
        if num not in by_num:
            raise ValueError(f'ref list modification names picNum {num}, which is not a short-term reference')
        pic = by_num[num]
        lst = lst[:idx] + [pic] + lst[idx:n_active]
        idx += 1
        rest = [x for x in lst[idx:] if x != pic]
        lst = (lst[:idx] + rest + [None] * (n_active + 1))[:n_active + 1]
    lst = lst[:n_active]
    while lst and lst[-1] is None:
        lst.pop()
    return lst


def init_lists(t, refs, cur_fn, cur_poc, n0, n1, max_fn):
    """Initial reference lists (8.2.4.2, frames, short-term only) as decode
    ids, truncated to the active sizes. refs: dicts with 'dec', 'frame_num'
    and 'poc'."""
    if t == 'P':
        def picnum(r):
            fn = r['frame_num']
            return fn - max_fn if fn > cur_fn else fn
        return [[r['dec'] for r in sorted(refs, key=picnum, reverse=True)][:n0], []]
    if t == 'B':
        before = sorted((r for r in refs if r['poc'] < cur_poc), key=lambda r: -r['poc'])
        after = sorted((r for r in refs if r['poc'] > cur_poc), key=lambda r: r['poc'])
        l0 = [r['dec'] for r in before + after]
        l1 = [r['dec'] for r in after + before]
        if len(l1) > 1 and l1 == l0:
            l1[0], l1[1] = l1[1], l1[0]
        return [l0[:n0], l1[:n1]]
    return [[], []]


def dpb_walk(spss, pics):
    """Header-only DPB simulation: POC, frame_num, final reference lists
    (initial lists + ref_pic_list_modification) and reference marking
    (sliding window or MMCO 1). Also attaches 'dpb_after': the decode ids of
    the short-term references after the picture is marked."""
    sps = spss[0]
    max_fn = 1 << sps['log2_max_frame_num']
    max_lsb = 1 << sps['log2_max_poc_lsb']
    refs = []  # short-term reference pictures (dicts)
    prev_msb = prev_lsb = 0
    for d, p in enumerate(pics):
        sl = p['slices'][0]
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
        p['lists'] = init_lists(sl['slice_type'], refs, sl['frame_num'], p['poc'], sl['num_ref_l0'],
                                sl['num_ref_l1'], max_fn)
        for l in range(2):
            if sl['rplm'][l]:
                n_active = (sl['num_ref_l0'], sl['num_ref_l1'])[l]
                p['lists'][l] = modify_list(p['lists'][l], sl['rplm'][l], n_active, sl['frame_num'], refs, max_fn)
        # reference marking (sliding window, or MMCO 1: unmark a short-term picture)
        if p['ref_idc']:
            def wrap(r):
                fn = r['frame_num']
                return fn - max_fn if fn > sl['frame_num'] else fn
            if sl.get('long_term_ref'):
                raise ValueError('long-term references are not supported')
            if sl.get('adaptive') or sl['mmco']:
                for op, *args in sl['mmco']:
                    if op != 1:
                        raise ValueError(f'MMCO {op} is not supported')
                    num = sl['frame_num'] - (args[0] + 1)
                    hit = [r for r in refs if wrap(r) == num]
                    if not hit:
                        raise ValueError(f'MMCO 1 names picNum {num}, which is not a short-term reference')
                    refs.remove(hit[0])
            elif not p['idr'] and len(refs) >= max(sps['max_num_ref_frames'], 1):
                refs.remove(min(refs, key=wrap))
            refs.append(p)
            if len(refs) > max(sps['max_num_ref_frames'], 1):
                raise ValueError(f'decoded picture buffer overflow at picture {d}')
            prev_msb, prev_lsb = msb, lsb
        p['dpb_after'] = [r['dec'] for r in refs]
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


class _Grid:
    """A numpy array indexed like the nested lists of Motion ([y][x]),
    returning Python values: ints, (x, y) tuples, or None for -1 when
    `none` is set."""

    def __init__(self, arr, none=False):
        self.a, self.none = arr, none

    def __getitem__(self, y):
        return _GridRow(self.a[y], self.none)


class _GridRow:
    def __init__(self, row, none):
        self.r, self.none = row, none

    def __getitem__(self, x):
        v = self.r[x]
        if v.ndim:
            return (int(v[0]), int(v[1]))
        v = int(v)
        return None if self.none and v < 0 else v


class CompactMotion:
    """Read-only, ~6x smaller copy of a Motion, for keeping co-located
    motion fields around (derive_motion reads intra, refidx, mv and refpic
    of its col_motion through [y][x] indexing)."""

    def __init__(self, m):
        import numpy as np
        self.W, self.H = m.W, m.H
        self.intra = _Grid(np.array(m.intra, np.uint8))
        self.refidx = [_Grid(np.array(m.refidx[l], np.int8)) for l in range(2)]
        self.mv = [_Grid(np.array(m.mv[l], np.int16)) for l in range(2)]
        self.refpic = [_Grid(np.array([[-1 if r is None else r for r in row] for row in m.refpic[l]], np.int32),
                             none=True) for l in range(2)]


def derive_motion(pic, mbs, W, H, pics, col_motion=None, target=None):
    """Reconstruct motion for a parsed picture. col_motion: Motion of
    RefPicList1[0] (needed for B pictures with direct blocks).

    With `target` (a Motion), explicit partitions take their mvs from the
    target field and their mvd is solved (written into the records), so the
    records encode exactly that motion under the current neighbourhood."""
    t = pic['slices'][0]['slice_type']
    lists = pic['lists']
    m = Motion(W, H)
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

    def spatial_pred(cmx, cmy):
        """MB-level reference indices and mv predictors of spatial direct
        (8.4.1.2.2): MinPositive over the A / B / C(or D) neighbours of the
        whole MB."""
        x0, y0 = cmx * 4, cmy * 4
        out = []
        for lst in range(2):
            A = nbr(x0 - 1, y0, lst, cmx, cmy) or (-1, (0, 0))
            B = nbr(x0, y0 - 1, lst, cmx, cmy) or (-1, (0, 0))
            C = nbr(x0 + 4, y0 - 1, lst, cmx, cmy)
            if C is None:
                C = nbr(x0 - 1, y0 - 1, lst, cmx, cmy)
            C = C or (-1, (0, 0))
            pos = [n[0] for n in (A, B, C) if n[0] >= 0]
            if not pos:
                out.append((-1, (0, 0)))
                continue
            ref = min(pos)
            match = [n for n in (A, B, C) if n[0] == ref]
            if len(match) > 1:
                mv = (_median(A[1][0], B[1][0], C[1][0]), _median(A[1][1], B[1][1], C[1][1]))
            else:
                mv = match[0][1]
            out.append((ref, mv))
        if out[0][0] < 0 and out[1][0] < 0:
            out = [(0, (0, 0)), (0, (0, 0))]
        return out

    def spatial_direct(i8, cmx, cmy, sp):
        """Spatial direct for one 8x8 quadrant (direct_8x8_inference): the
        MB-level prediction, with zero mvs where the co-located block is
        (nearly) still and uses reference index 0."""
        cm = col_motion
        ox, oy = cmx * 4 + (i8 % 2) * 2, cmy * 4 + (i8 // 2) * 2
        cx = cmx * 4 + (3 if i8 % 2 else 0)
        cy = cmy * 4 + (3 if i8 // 2 else 0)
        blocks = [(ox + x, oy + y) for y in range(2) for x in range(2)]
        col_zero = False
        if not cm.intra[cy][cx]:
            l = 0 if cm.refidx[0][cy][cx] >= 0 else 1
            mvc = cm.mv[l][cy][cx]
            col_zero = cm.refidx[l][cy][cx] == 0 and -1 <= mvc[0] <= 1 and -1 <= mvc[1] <= 1
        for lst in range(2):
            ref, mv = sp[lst]
            if ref < 0:
                setblk(blocks, lst, -1, (0, 0), 1)
            else:
                setblk(blocks, lst, ref, (0, 0) if (col_zero and ref == 0) else mv, 1)
        done.update(blocks)

    spatial = t == 'B' and pic['slices'][0].get('direct_spatial')
    direct = spatial_direct if spatial else (lambda i8, cmx, cmy, sp: temporal_direct(i8, cmx, cmy))

    for mb in mbs:
        cmx, cmy = mb['x'], mb['y']
        x0, y0 = cmx * 4, cmy * 4
        done.clear()
        sp = spatial_pred(cmx, cmy) if spatial and not mb['intra'] else None
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
                direct(i8, cmx, cmy, sp)
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
                direct(i, cmx, cmy, sp)
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

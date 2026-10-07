"""CABAC slice-data codec for H.264 (High profile, 4:2:0, 8-bit, progressive).

One code path handles both directions: every syntax element is coded through
a `Coder` that either decodes bins from a bitstream or encodes bins from
values already present in the macroblock records. Parse a slice into
records, (optionally) edit them, and encode them back.

Scope: single-slice pictures, frame_mbs_only, no MBAFF, no FMO, 4:2:0,
8-bit. That is what this project's streams use.
"""
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))
_tabs = json.load(open(os.path.join(_here, 'cabac_tables.json')))
RANGE_LPS = _tabs['rangeTabLPS']
TRANS_LPS = _tabs['transIdxLPS']
TRANS_MPS = [min(s + 1, 62) for s in range(63)] + [63]
INIT_I = _tabs['init_I']
INIT_PB = _tabs['init_PB']
LAST_8x8 = _tabs['last_coeff_flag_offset_8x8']
SIG_8x8 = [0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5,
           4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7,
           7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11,
           12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13, 11, 14, 10, 12]

# ctxBlockCat: 0 luma DC (I16), 1 luma AC (I16), 2 luma 4x4, 3 chroma DC,
# 4 chroma AC, 5 luma 8x8
CBF_BASE = [85, 89, 93, 97, 101]
SIG_BASE = [105, 120, 134, 149, 152, 402]
LAST_BASE = [166, 181, 195, 210, 213, 417]
ABS_BASE = [227, 237, 247, 257, 266, 426]
MAX_COEFF = [16, 15, 16, 4, 15, 64]
ABS1_CTX = [1, 2, 3, 4, 0, 0, 0, 0]
ABSGT1_CTX = [5, 5, 5, 5, 6, 7, 8, 9]
TRANS_EQ1 = [1, 2, 3, 3, 4, 5, 6, 7]
TRANS_GT1 = [4, 4, 4, 4, 5, 6, 7, 7]

# B macroblock types (Table 7-14): (partitioning, (pred part0, pred part1))
B_TYPES = [('D', None), ('16x16', ('L0',)), ('16x16', ('L1',)), ('16x16', ('Bi',)),
           ('16x8', ('L0', 'L0')), ('8x16', ('L0', 'L0')), ('16x8', ('L1', 'L1')), ('8x16', ('L1', 'L1')),
           ('16x8', ('L0', 'L1')), ('8x16', ('L0', 'L1')), ('16x8', ('L1', 'L0')), ('8x16', ('L1', 'L0')),
           ('16x8', ('L0', 'Bi')), ('8x16', ('L0', 'Bi')), ('16x8', ('L1', 'Bi')), ('8x16', ('L1', 'Bi')),
           ('16x8', ('Bi', 'L0')), ('8x16', ('Bi', 'L0')), ('16x8', ('Bi', 'L1')), ('8x16', ('Bi', 'L1')),
           ('16x8', ('Bi', 'Bi')), ('8x16', ('Bi', 'Bi')), ('8x8', None)]
# B sub-macroblock types (Table 7-18): (shape, pred)
B_SUB = [('D', None), ('8x8', 'L0'), ('8x8', 'L1'), ('8x8', 'Bi'), ('8x4', 'L0'), ('4x8', 'L0'),
         ('8x4', 'L1'), ('4x8', 'L1'), ('8x4', 'Bi'), ('4x8', 'Bi'), ('4x4', 'L0'), ('4x4', 'L1'),
         ('4x4', 'Bi')]
P_SUB = ['8x8', '8x4', '4x8', '4x4']
P_TYPES = ['16x16', '16x8', '8x16', '8x8']


def uses(pred, lst):
    return pred == 'Bi' or pred == ('L0', 'L1')[lst]


def sub_parts(shape):
    """4x4-block coordinates (within an 8x8) covered by each sub-partition."""
    return {'8x8': [[(0, 0), (1, 0), (0, 1), (1, 1)]],
            '8x4': [[(0, 0), (1, 0)], [(0, 1), (1, 1)]],
            '4x8': [[(0, 0), (0, 1)], [(1, 0), (1, 1)]],
            '4x4': [[(0, 0)], [(1, 0)], [(0, 1)], [(1, 1)]]}[shape]


def mb_parts(part):
    """4x4-block coordinates (within the MB) for each macroblock partition."""
    if part == '16x16':
        return [[(x, y) for y in range(4) for x in range(4)]]
    if part == '16x8':
        return [[(x, y) for y in range(2) for x in range(4)], [(x, y) for y in range(2, 4) for x in range(4)]]
    if part == '8x16':
        return [[(x, y) for y in range(4) for x in range(2)], [(x, y) for y in range(4) for x in range(2, 4)]]
    raise ValueError(part)


# luma4x4BlkIdx -> (x, y) in 4x4 units
BLK_XY = [((i % 4) % 2 + 2 * ((i // 4) % 2), (i % 4) // 2 + 2 * (i // 8)) for i in range(16)]


class Coder:
    """CABAC arithmetic engine. mode 'dec' reads bits, 'enc' writes them."""

    def __init__(self, mode, slice_qp, init_idc, slice_is_i, data=None, pos=0):
        self.mode = mode
        tab = INIT_I if slice_is_i else INIT_PB[init_idc]
        qp = max(0, min(51, slice_qp))
        self.st = [0] * 1024
        self.mps = [0] * 1024
        for i, (m, n) in enumerate(tab):
            pre = max(1, min(126, ((m * qp) >> 4) + n))
            if pre <= 63:
                self.st[i], self.mps[i] = 63 - pre, 0
            else:
                self.st[i], self.mps[i] = pre - 64, 1
        if mode == 'dec':
            self.bits = data  # list of 0/1
            self.pos = pos
            self._init_dec()
        else:
            self.out = []
            self._init_enc()

    # ---- decoder ----
    def _init_dec(self):
        self.rng = 510
        v = 0
        for _ in range(9):
            v = (v << 1) | self.bits[self.pos]
            self.pos += 1
        self.off = v

    # ---- encoder ----
    def _init_enc(self):
        self.low, self.rng, self.first, self.outstanding = 0, 510, True, 0

    def _put(self, b):
        if self.first:
            self.first = False
        else:
            self.out.append(b)
        while self.outstanding:
            self.out.append(1 - b)
            self.outstanding -= 1

    def _renorm_e(self):
        while self.rng < 256:
            if self.low < 256:
                self._put(0)
            elif self.low >= 512:
                self.low -= 512
                self._put(1)
            else:
                self.low -= 256
                self.outstanding += 1
            self.rng <<= 1
            self.low <<= 1

    # ---- bins ----
    def bin(self, ctx, val=None):
        s, m = self.st[ctx], self.mps[ctx]
        lps = RANGE_LPS[s][(self.rng >> 6) & 3]
        self.rng -= lps
        if self.mode == 'dec':
            if self.off >= self.rng:
                val = 1 - m
                self.off -= self.rng
                self.rng = lps
            else:
                val = m
            while self.rng < 256:
                self.rng <<= 1
                self.off = (self.off << 1) | self.bits[self.pos]
                self.pos += 1
        else:
            if val != m:
                self.low += self.rng
                self.rng = lps
            self._renorm_e()
        if val != m:
            if s == 0:
                self.mps[ctx] = 1 - m
            self.st[ctx] = TRANS_LPS[s]
        else:
            self.st[ctx] = TRANS_MPS[s]
        return val

    def bypass(self, val=None):
        if self.mode == 'dec':
            self.off = (self.off << 1) | self.bits[self.pos]
            self.pos += 1
            if self.off >= self.rng:
                self.off -= self.rng
                return 1
            return 0
        self.low <<= 1
        if val:
            self.low += self.rng
        if self.low >= 1024:
            self._put(1)
            self.low -= 1024
        elif self.low < 512:
            self._put(0)
        else:
            self.low -= 512
            self.outstanding += 1
        return val

    def term(self, val=None):
        self.rng -= 2
        if self.mode == 'dec':
            if self.off >= self.rng:
                return 1
            while self.rng < 256:
                self.rng <<= 1
                self.off = (self.off << 1) | self.bits[self.pos]
                self.pos += 1
            return 0
        if val:
            self.low += self.rng
            self.rng = 2
            self._renorm_e()
            self._put((self.low >> 9) & 1)
            self.out.extend(((self.low >> 8) & 1, 1))
        else:
            self._renorm_e()
        return val

    # ---- raw bits around I_PCM ----
    def pcm(self, samples=None):
        """After an I_PCM mb_type: align, read/write 384 bytes, restart engine."""
        if self.mode == 'dec':
            self.pos = (self.pos + 7) & ~7
            out = []
            for _ in range(384):
                v = 0
                for _ in range(8):
                    v = (v << 1) | self.bits[self.pos]
                    self.pos += 1
                out.append(v)
            self._init_dec()
            return out
        while len(self.out) % 8:
            self.out.append(0)
        for v in samples:
            self.out.extend((v >> (7 - k)) & 1 for k in range(8))
        self._init_enc()
        return samples


class SliceCodec:
    """Parse or write the slice_data() of one picture."""

    def __init__(self, sps, pps, hdr):
        self.W = sps['mb_w']
        self.H = sps['mb_h']
        self.pps, self.hdr = pps, hdr
        self.t = hdr['slice_type']
        self.n_ref = (hdr['num_ref_l0'], hdr['num_ref_l1'])
        self.lists = 2 if self.t == 'B' else (1 if self.t == 'P' else 0)
        W4, H4 = self.W * 4, self.H * 4
        # 4x4-granular neighbour state used for context derivation
        self.ref = [[[-1] * W4 for _ in range(H4)] for _ in range(2)]
        self.mvd = [[[(0, 0)] * W4 for _ in range(H4)] for _ in range(2)]
        self.dir = [[0] * W4 for _ in range(H4)]
        self.cbfL = [[0] * W4 for _ in range(H4)]
        self.cbfC = [[[0] * (self.W * 2) for _ in range(self.H * 2)] for _ in range(2)]
        self.t8 = [[0] * W4 for _ in range(H4)]
        self.imode = [[-1] * W4 for _ in range(H4)]  # Intra4x4/8x8 modes, -1 = other

    # ------------------------------------------------------------------ helpers
    def nb(self, mbx, mby, dx, dy):
        x, y = mbx + dx, mby + dy
        if 0 <= x < self.W and 0 <= y < self.H and (y < mby or (y == mby and x < mbx)):
            return self.mbs[y * self.W + x]
        return None

    # ------------------------------------------------------------------ top level
    def run(self, c, mbs=None):
        """Decode (mbs None) or encode the given list of macroblock records."""
        self.c = c
        dec = c.mode == 'dec'
        n = self.W * self.H
        self.mbs = [None] * n
        self.prev_qpd = 0
        out = []
        for addr in range(n):
            mbx, mby = addr % self.W, addr // self.W
            mb = {} if dec else mbs[addr]
            mb['x'], mb['y'] = mbx, mby
            if self.t != 'I':
                a, b = self.nb(mbx, mby, -1, 0), self.nb(mbx, mby, 0, -1)
                inc = (a is not None and not a['skip']) + (b is not None and not b['skip'])
                mb['skip'] = c.bin((24 if self.t == 'B' else 11) + inc, None if dec else mb['skip'])
            else:
                mb['skip'] = 0
            if mb['skip']:
                self.skip_mb(mb)
            else:
                self.macroblock(mb)
            self.mbs[addr] = mb
            out.append(mb)
            end = c.term(None if dec else int(addr == n - 1))
            if dec and end:
                assert addr == n - 1, f'early end of slice at mb {addr}'
        return out

    # ------------------------------------------------------------------ skip
    def skip_mb(self, mb):
        mb.update(kind='skip', intra=0, pcm=0, i16=0, cbp=0, t8=0, cmode=0, qpd=0)
        mb['direct16'] = int(self.t == 'B')
        x0, y0 = mb['x'] * 4, mb['y'] * 4
        for y in range(y0, y0 + 4):
            for x in range(x0, x0 + 4):
                self.ref[0][y][x] = 0 if self.t == 'P' else -1
                self.ref[1][y][x] = -1
                self.mvd[0][y][x] = self.mvd[1][y][x] = (0, 0)
                self.dir[y][x] = int(self.t == 'B')
                self.cbfL[y][x] = 0
                self.t8[y][x] = 0
                self.imode[y][x] = -1
        for comp in range(2):
            for y in range(2):
                for x in range(2):
                    self.cbfC[comp][mb['y'] * 2 + y][mb['x'] * 2 + x] = 0
        mb['dcL'], mb['dcC'] = 0, [0, 0]
        self.prev_qpd = 0

    # ------------------------------------------------------------------ mb_type
    def mb_type(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        a, b = self.nb(mb['x'], mb['y'], -1, 0), self.nb(mb['x'], mb['y'], 0, -1)
        if self.t == 'I':
            inc = sum(1 for n in (a, b) if n is not None and n['kind'] in ('I16', 'PCM'))
            self.intra_type(mb, 3, inc, True)
            return
        if self.t == 'P':
            intra = c.bin(14, None if dec else int(mb['kind'] in ('I4', 'I8', 'I16', 'PCM')))
            if intra:
                self.intra_type(mb, 17, 0, False)
                return
            if dec:
                if c.bin(15) == 0:
                    t = 3 * c.bin(16)
                else:
                    t = 2 - c.bin(17)
                mb['kind'], mb['part'] = 'P', P_TYPES[t]
            else:
                t = P_TYPES.index(mb['part'])
                if t in (0, 3):
                    c.bin(15, 0)
                    c.bin(16, int(t == 3))
                else:
                    c.bin(15, 1)
                    c.bin(17, int(t == 1))
            return
        # B slice
        inc = sum(1 for n in (a, b) if n is not None and not n['direct16'])
        if dec:
            if c.bin(27 + inc) == 0:
                t = 0
            elif c.bin(30) == 0:
                t = 1 + c.bin(32)
            else:
                bits = c.bin(31) << 3
                bits |= c.bin(32) << 2
                bits |= c.bin(32) << 1
                bits |= c.bin(32)
                if bits < 8:
                    t = bits + 3
                elif bits == 13:
                    self.intra_type(mb, 32, 0, False)
                    return
                elif bits == 14:
                    t = 11
                elif bits == 15:
                    t = 22
                else:
                    t = ((bits << 1) | c.bin(32)) - 4
            mb['kind'], mb['btype'] = 'B', t
        else:
            if mb['kind'] in ('I4', 'I8', 'I16', 'PCM'):
                c.bin(27 + inc, 1)
                c.bin(30, 1)
                for k, v in zip((31, 32, 32, 32), (1, 1, 0, 1)):
                    c.bin(k, v)
                self.intra_type(mb, 32, 0, False)
                return
            t = mb['btype']
            if t == 0:
                c.bin(27 + inc, 0)
            elif t <= 2:
                c.bin(27 + inc, 1)
                c.bin(30, 0)
                c.bin(32, t - 1)
            else:
                c.bin(27 + inc, 1)
                c.bin(30, 1)
                if t <= 10:
                    bits, extra = t - 3, None
                elif t == 11:
                    bits, extra = 14, None
                elif t == 22:
                    bits, extra = 15, None
                else:
                    v = t + 4
                    bits, extra = v >> 1, v & 1
                c.bin(31, (bits >> 3) & 1)
                c.bin(32, (bits >> 2) & 1)
                c.bin(32, (bits >> 1) & 1)
                c.bin(32, bits & 1)
                if extra is not None:
                    c.bin(32, extra)
        part, _ = B_TYPES[mb['btype']]
        mb['part'] = part

    def intra_type(self, mb, base, inc, islice):
        """I mb_type binarization (prefix-free part), shared by I/P/B slices."""
        c, dec = self.c, self.c.mode == 'dec'
        if dec:
            b0 = c.bin(base + inc if islice else base)
            if b0 == 0:
                mb['kind'] = 'I4'
                return
            if c.term():
                mb['kind'] = 'PCM'
                return
            o = 2 if islice else 0
            luma = c.bin(base + 1 + o)
            chroma = 0
            if c.bin(base + 2 + o):
                chroma = 1 + c.bin(base + 2 + (1 if islice else 0) + o)
            pm = c.bin(base + 3 + (1 if islice else 0) + o) << 1
            pm |= c.bin(base + 3 + (2 if islice else 0) + o)
            mb.update(kind='I16', i16pred=pm, i16cbpC=chroma, i16cbpL=luma)
            return
        k = mb['kind']
        c.bin(base + inc if islice else base, 0 if k in ('I4', 'I8') else 1)
        if k in ('I4', 'I8'):
            return
        c.term(int(k == 'PCM'))
        if k == 'PCM':
            return
        o = 2 if islice else 0
        c.bin(base + 1 + o, mb['i16cbpL'])
        c.bin(base + 2 + o, int(mb['i16cbpC'] > 0))
        if mb['i16cbpC'] > 0:
            c.bin(base + 2 + (1 if islice else 0) + o, mb['i16cbpC'] - 1)
        c.bin(base + 3 + (1 if islice else 0) + o, mb['i16pred'] >> 1)
        c.bin(base + 3 + (2 if islice else 0) + o, mb['i16pred'] & 1)

    # ------------------------------------------------------------------ macroblock
    def macroblock(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        self.mb_type(mb)
        k = mb['kind']
        mb['intra'] = int(k in ('I4', 'I8', 'I16', 'PCM'))
        mb['pcm'] = int(k == 'PCM')
        mb['i16'] = int(k == 'I16')
        mb['direct16'] = int(k == 'B' and mb['btype'] == 0)
        mb.setdefault('t8', 0)
        x0, y0 = mb['x'] * 4, mb['y'] * 4
        # reset neighbour state for this MB
        for y in range(y0, y0 + 4):
            for x in range(x0, x0 + 4):
                for l in range(2):
                    self.ref[l][y][x] = -1
                    self.mvd[l][y][x] = (0, 0)
                self.dir[y][x] = 0
                self.cbfL[y][x] = 0
                self.t8[y][x] = 0
                self.imode[y][x] = -1
        for comp in range(2):
            for y in range(2):
                for x in range(2):
                    self.cbfC[comp][mb['y'] * 2 + y][mb['x'] * 2 + x] = 0
        mb['dcL'], mb['dcC'] = 0, [0, 0]
        if k == 'PCM':
            mb['samples'] = c.pcm(None if dec else mb['samples'])
            mb.update(cbp=0x2f, t8=0, cmode=0, qpd=0)
            for y in range(y0, y0 + 4):
                for x in range(x0, x0 + 4):
                    self.cbfL[y][x] = 1
            for comp in range(2):
                for y in range(2):
                    for x in range(2):
                        self.cbfC[comp][mb['y'] * 2 + y][mb['x'] * 2 + x] = 1
            mb['dcL'], mb['dcC'] = 1, [1, 1]
            self.prev_qpd = 0
            return
        if mb['intra']:
            self.intra_pred(mb)
        elif mb['part'] == '8x8':
            self.sub_mb_pred(mb)
        elif mb['part'] == 'D':
            for y in range(y0, y0 + 4):
                for x in range(x0, x0 + 4):
                    self.dir[y][x] = 1
        else:
            self.mb_pred(mb)
        # coded_block_pattern
        if k != 'I16':
            self.cbp(mb)
            cbp = mb['cbp']
            if (cbp & 15) and not mb['intra'] and self.pps['transform_8x8'] and self.t8_allowed(mb):
                mb['t8'] = c.bin(399 + self.t8_inc(mb), None if dec else mb['t8'])
        else:
            mb['cbp'] = (mb['i16cbpC'] << 4) | (15 if mb['i16cbpL'] else 0)
        if mb['t8']:
            for y in range(y0, y0 + 4):
                for x in range(x0, x0 + 4):
                    self.t8[y][x] = 1
        if mb['cbp'] or k == 'I16':
            self.qp_delta(mb)
            self.residual(mb)
        else:
            mb['qpd'] = 0
            self.prev_qpd = 0

    def t8_allowed(self, mb):
        if mb['part'] == '8x8':
            for st in mb['sub']:
                if self.t == 'B':
                    shape = B_SUB[st][0]
                    if shape == 'D':
                        continue  # direct_8x8_inference is 1 in these streams
                else:
                    shape = P_SUB[st]
                if shape != '8x8':
                    return False
        return True

    def t8_inc(self, mb):
        x, y = mb['x'] * 4, mb['y'] * 4
        a = self.t8[y][x - 1] if x > 0 else 0
        b = self.t8[y - 1][x] if y > 0 else 0
        return a + b

    # ------------------------------------------------------------------ intra
    def intra_pred(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        if mb['kind'] in ('I4', 'I8'):
            if self.pps['transform_8x8']:
                t8 = c.bin(399 + self.t8_inc(mb), None if dec else int(mb['kind'] == 'I8'))
                mb['kind'] = 'I8' if t8 else 'I4'
                mb['t8'] = t8
            n = 4 if mb['kind'] == 'I8' else 16
            x0, y0 = mb['x'] * 4, mb['y'] * 4
            if dec:
                mb['ipred'], mb['imodes'] = [], []
            for i in range(n):
                if n == 4:
                    bx, by, cells = (i % 2) * 2, (i // 2) * 2, ((0, 0), (1, 0), (0, 1), (1, 1))
                else:
                    (bx, by), cells = BLK_XY[i], ((0, 0),)
                gx, gy = x0 + bx, y0 + by
                ma = self.imode[gy][gx - 1] if gx > 0 else -1
                mb_ = self.imode[gy - 1][gx] if gy > 0 else -1
                pred = min(2 if ma < 0 else ma, 2 if mb_ < 0 else mb_)
                if dec:
                    if c.bin(68):
                        r, mode = -1, pred
                    else:
                        r = c.bin(69) | (c.bin(69) << 1) | (c.bin(69) << 2)
                        mode = r if r < pred else r + 1
                    mb['ipred'].append(r)
                    mb['imodes'].append(mode)
                else:
                    mode = mb['imodes'][i]
                    r = -1 if mode == pred else (mode if mode < pred else mode - 1)
                    c.bin(68, int(r < 0))
                    if r >= 0:
                        c.bin(69, r & 1)
                        c.bin(69, (r >> 1) & 1)
                        c.bin(69, (r >> 2) & 1)
                for (cx, cy) in cells:
                    self.imode[gy + cy][gx + cx] = mode
        # intra_chroma_pred_mode
        a, b = self.nb(mb['x'], mb['y'], -1, 0), self.nb(mb['x'], mb['y'], 0, -1)
        inc = sum(1 for n in (a, b) if n is not None and n['intra'] and not n['pcm'] and n['cmode'] != 0)
        if dec:
            if c.bin(64 + inc) == 0:
                mb['cmode'] = 0
            elif c.bin(67) == 0:
                mb['cmode'] = 1
            else:
                mb['cmode'] = 2 + c.bin(67)
        else:
            m = mb['cmode']
            c.bin(64 + inc, int(m > 0))
            if m > 0:
                c.bin(67, int(m > 1))
                if m > 1:
                    c.bin(67, int(m > 2))

    # ------------------------------------------------------------------ inter
    def ref_idx(self, lst, bx, by, val):
        c = self.c
        inc = 0
        if bx > 0:
            r = self.ref[lst][by][bx - 1]
            if r > 0 and not (self.t == 'B' and self.dir[by][bx - 1]):
                inc += 1
        if by > 0:
            r = self.ref[lst][by - 1][bx]
            if r > 0 and not (self.t == 'B' and self.dir[by - 1][bx]):
                inc += 2
        if c.mode == 'dec':
            v = 0
            ctx = 54 + inc
            while c.bin(ctx):
                v += 1
                ctx = 54 + (4 if v == 1 else 5)
            return v
        ctx = 54 + inc
        for i in range(val):
            c.bin(ctx, 1)
            ctx = 54 + (4 if i == 0 else 5)
        c.bin(ctx, 0)
        return val

    def mvd_comp(self, lst, bx, by, comp, val):
        c = self.c
        s = 0
        if bx > 0:
            s += self.mvd[lst][by][bx - 1][comp]
        if by > 0:
            s += self.mvd[lst][by - 1][bx][comp]
        base = 40 if comp == 0 else 47
        inc = 0 if s < 3 else (1 if s <= 32 else 2)
        if c.mode == 'dec':
            if not c.bin(base + inc):
                return 0
            v = 1
            ctx = base + 3
            while v < 9 and c.bin(ctx):
                v += 1
                if v < 5:
                    ctx += 1
            if v >= 9:
                k = 3
                while c.bypass():
                    v += 1 << k
                    k += 1
                while k:
                    k -= 1
                    v += c.bypass() << k
            return -v if c.bypass() else v
        a = abs(val)
        c.bin(base + inc, int(a > 0))
        if a == 0:
            return 0
        ctx = base + 3
        v = 1
        while v < 9:
            if a > v:
                c.bin(ctx, 1)
                v += 1
                if v < 5:
                    ctx += 1
            else:
                c.bin(ctx, 0)
                break
        if a >= 9:
            r = a - 9
            k = 3
            while r >= (1 << k):
                c.bypass(1)
                r -= 1 << k
                k += 1
            c.bypass(0)
            while k:
                k -= 1
                c.bypass((r >> k) & 1)
        c.bypass(int(val < 0))
        return val

    def mb_pred(self, mb):
        """16x16 / 16x8 / 8x16 partitions in P and B slices."""
        dec = self.c.mode == 'dec'
        x0, y0 = mb['x'] * 4, mb['y'] * 4
        parts = mb_parts(mb['part'])
        preds = ('L0',) * len(parts) if self.t == 'P' else B_TYPES[mb['btype']][1]
        if dec:
            mb['refs'] = [[None] * len(parts) for _ in range(2)]
            mb['mvds'] = [[None] * len(parts) for _ in range(2)]
        for lst in range(self.lists):
            for p, blocks in enumerate(parts):
                if not uses(preds[p], lst):
                    continue
                bx, by = x0 + blocks[0][0], y0 + blocks[0][1]
                if self.n_ref[lst] > 1:
                    r = self.ref_idx(lst, bx, by, None if dec else mb['refs'][lst][p])
                else:
                    r = 0
                mb['refs'][lst][p] = r
                for (x, y) in blocks:
                    self.ref[lst][y0 + y][x0 + x] = r
        for lst in range(self.lists):
            for p, blocks in enumerate(parts):
                if not uses(preds[p], lst):
                    continue
                bx, by = x0 + blocks[0][0], y0 + blocks[0][1]
                old = None if dec else mb['mvds'][lst][p]
                dx = self.mvd_comp(lst, bx, by, 0, None if dec else old[0])
                dy = self.mvd_comp(lst, bx, by, 1, None if dec else old[1])
                mb['mvds'][lst][p] = (dx, dy)
                a = (min(abs(dx), 70), min(abs(dy), 70))
                for (x, y) in blocks:
                    self.mvd[lst][y0 + y][x0 + x] = a

    def sub_mb_pred(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        x0, y0 = mb['x'] * 4, mb['y'] * 4
        if dec:
            mb['sub'] = []
        for i in range(4):
            if self.t == 'P':
                if dec:
                    if c.bin(21):
                        st = 0
                    elif not c.bin(22):
                        st = 1
                    else:
                        st = 2 if c.bin(23) else 3
                    mb['sub'].append(st)
                else:
                    st = mb['sub'][i]
                    c.bin(21, int(st == 0))
                    if st:
                        c.bin(22, int(st != 1))
                        if st != 1:
                            c.bin(23, int(st == 2))
            else:
                if dec:
                    if not c.bin(36):
                        st = 0
                    elif not c.bin(37):
                        st = 1 + c.bin(39)
                    else:
                        st = 3
                        if c.bin(38):
                            if c.bin(39):
                                st = 11 + c.bin(39)
                            else:
                                st = 7 + 2 * c.bin(39)
                                st += c.bin(39)
                        else:
                            st += 2 * c.bin(39)
                            st += c.bin(39)
                    mb['sub'].append(st)
                else:
                    self.b_sub_enc(mb['sub'][i])
        subs = mb['sub']

        def info(st):
            if self.t == 'P':
                return P_SUB[st], 'L0'
            return B_SUB[st]
        # direct sub-blocks are marked before refs are coded
        for i, st in enumerate(subs):
            shape, _ = info(st)
            if shape == 'D':
                for (x, y) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    self.dir[y0 + (i // 2) * 2 + y][x0 + (i % 2) * 2 + x] = 1
        if dec:
            mb['refs'] = [[None] * 4 for _ in range(2)]
            mb['mvds'] = [[[] for _ in range(4)] for _ in range(2)]
        for lst in range(self.lists):
            for i, st in enumerate(subs):
                shape, pred = info(st)
                if shape == 'D' or not uses(pred, lst):
                    continue
                bx, by = x0 + (i % 2) * 2, y0 + (i // 2) * 2
                if self.n_ref[lst] > 1:
                    r = self.ref_idx(lst, bx, by, None if dec else mb['refs'][lst][i])
                else:
                    r = 0
                mb['refs'][lst][i] = r
                for (x, y) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    self.ref[lst][by + y][bx + x] = r
        for lst in range(self.lists):
            for i, st in enumerate(subs):
                shape, pred = info(st)
                if shape == 'D' or not uses(pred, lst):
                    continue
                ox, oy = x0 + (i % 2) * 2, y0 + (i // 2) * 2
                for j, blocks in enumerate(sub_parts(shape)):
                    bx, by = ox + blocks[0][0], oy + blocks[0][1]
                    old = None if dec else mb['mvds'][lst][i][j]
                    dx = self.mvd_comp(lst, bx, by, 0, None if dec else old[0])
                    dy = self.mvd_comp(lst, bx, by, 1, None if dec else old[1])
                    if dec:
                        mb['mvds'][lst][i].append((dx, dy))
                    a = (min(abs(dx), 70), min(abs(dy), 70))
                    for (x, y) in blocks:
                        self.mvd[lst][oy + y][ox + x] = a

    def b_sub_enc(self, st):
        c = self.c
        c.bin(36, int(st != 0))
        if st == 0:
            return
        c.bin(37, int(st >= 3))
        if st < 3:
            c.bin(39, st - 1)
            return
        if st >= 11:
            c.bin(38, 1)
            c.bin(39, 1)
            c.bin(39, st - 11)
        elif st >= 7:
            c.bin(38, 1)
            c.bin(39, 0)
            c.bin(39, ((st - 7) >> 1) & 1)
            c.bin(39, (st - 7) & 1)
        else:
            c.bin(38, 0)
            c.bin(39, ((st - 3) >> 1) & 1)
            c.bin(39, (st - 3) & 1)

    # ------------------------------------------------------------------ cbp / qp
    def cbp(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        a, b = self.nb(mb['x'], mb['y'], -1, 0), self.nb(mb['x'], mb['y'], 0, -1)
        val = None if dec else mb['cbp']
        cbp = 0

        def cond_luma(n, bit):
            # condTermFlagN = 0 when N unavailable, I_PCM, or its 8x8 is coded
            if n is None or n['pcm']:
                return 0
            if not n['skip'] and (n['cbp'] >> bit) & 1:
                return 0
            return 1
        for i in range(4):
            bx, by = i % 2, i // 2
            ca = cond_luma(a, by * 2 + 1) if bx == 0 else int(not (cbp >> (i - 1)) & 1)
            cb = cond_luma(b, 2 + bx) if by == 0 else int(not (cbp >> (i - 2)) & 1)
            v = c.bin(73 + ca + 2 * cb, None if dec else (val >> i) & 1)
            cbp |= v << i

        def chroma(n):
            if n is None:
                return 0
            if n['pcm']:
                return 2
            if n['skip']:
                return 0
            return n['cbp'] >> 4
        ca, cb = chroma(a), chroma(b)
        v0 = c.bin(77 + int(ca > 0) + 2 * int(cb > 0), None if dec else int((val >> 4) > 0))
        if v0:
            v1 = c.bin(77 + 4 + int(ca == 2) + 2 * int(cb == 2), None if dec else int((val >> 4) == 2))
            cbp |= (1 + v1) << 4
        mb['cbp'] = cbp

    def qp_delta(self, mb):
        c, dec = self.c, self.c.mode == 'dec'
        if dec:
            if not c.bin(60 + int(self.prev_qpd != 0)):
                v = 0
            else:
                k = 1
                ctx = 62
                while c.bin(ctx):
                    k += 1
                    ctx = 63
                v = (k + 1) // 2 if k & 1 else -(k // 2)
            mb['qpd'] = v
        else:
            v = mb['qpd']
            k = 2 * v - 1 if v > 0 else -2 * v
            c.bin(60 + int(self.prev_qpd != 0), int(k > 0))
            if k:
                ctx = 62
                for _ in range(k - 1):
                    c.bin(ctx, 1)
                    ctx = 63
                c.bin(ctx, 0)
        self.prev_qpd = mb['qpd']

    # ------------------------------------------------------------------ residual
    def residual(self, mb):
        dec = self.c.mode == 'dec'
        if dec:
            mb['coef'] = {}
        x0, y0 = mb['x'] * 4, mb['y'] * 4
        intra = mb['intra']
        cbp = mb['cbp']
        a, b = self.nb(mb['x'], mb['y'], -1, 0), self.nb(mb['x'], mb['y'], 0, -1)

        def unavail():
            return 1 if intra else 0
        if mb['i16']:
            def dc_cond(n):
                if n is None:
                    return unavail()
                if n['pcm']:
                    return 1
                return n['dcL'] if n['i16'] else 0
            inc = dc_cond(a) + 2 * dc_cond(b)
            mb['dcL'] = self.block(mb, ('LDC',), 0, inc)
        for i8 in range(4):
            if not (cbp >> i8) & 1:
                continue
            if mb['t8']:
                self.block(mb, ('L8', i8), 5, None)
                bx, by = (i8 % 2) * 2, (i8 // 2) * 2
                for (x, y) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    self.cbfL[y0 + by + y][x0 + bx + x] = 1
                continue
            for i4 in range(4):
                idx = i8 * 4 + i4
                bx, by = BLK_XY[idx]
                gx, gy = x0 + bx, y0 + by
                ca = self.luma_cond(gx - 1, gy, mb) if gx > 0 else unavail()
                cb = self.luma_cond(gx, gy - 1, mb) if gy > 0 else unavail()
                cat = 1 if mb['i16'] else 2
                self.cbfL[gy][gx] = self.block(mb, ('L4', idx), cat, ca + 2 * cb)
        if cbp >> 4:
            for comp in range(2):
                def cdc(n):
                    if n is None:
                        return unavail()
                    if n['pcm']:
                        return 1
                    if n['skip'] or (n['cbp'] >> 4) == 0:
                        return 0
                    return n['dcC'][comp]
                inc = cdc(a) + 2 * cdc(b)
                mb['dcC'][comp] = self.block(mb, ('CDC', comp), 3, inc)
        if cbp >> 4 == 2:
            for comp in range(2):
                for i in range(4):
                    cx, cy = mb['x'] * 2 + i % 2, mb['y'] * 2 + i // 2
                    ca = self.chroma_cond(comp, cx - 1, cy, mb) if cx > 0 else unavail()
                    cb = self.chroma_cond(comp, cx, cy - 1, mb) if cy > 0 else unavail()
                    self.cbfC[comp][cy][cx] = self.block(mb, ('CAC', comp, i), 4, ca + 2 * cb)

    def luma_cond(self, gx, gy, cur):
        n = self.mbs[(gy // 4) * self.W + gx // 4] if (gx // 4, gy // 4) != (cur['x'], cur['y']) else cur
        if n is cur:
            return self.cbfL[gy][gx]
        if n['skip']:
            return 0
        if n['pcm']:
            return 1
        b8 = ((gy % 4) // 2) * 2 + (gx % 4) // 2
        if not (n['cbp'] >> b8) & 1:
            return 0
        return self.cbfL[gy][gx]

    def chroma_cond(self, comp, cx, cy, cur):
        n = self.mbs[(cy // 2) * self.W + cx // 2] if (cx // 2, cy // 2) != (cur['x'], cur['y']) else cur
        if n is cur:
            return self.cbfC[comp][cy][cx]
        if n['skip']:
            return 0
        if n['pcm']:
            return 1
        if n['cbp'] >> 4 != 2:
            return 0
        return self.cbfC[comp][cy][cx]

    def block(self, mb, key, cat, cbf_inc):
        """residual_block_cabac. Returns coded_block_flag. Coefficients are
        stored in scan order under mb['coef'][key]."""
        c, dec = self.c, self.c.mode == 'dec'
        n = MAX_COEFF[cat]
        coef = None if dec else mb['coef'].get(key)
        if cat != 5:
            cbf = c.bin(CBF_BASE[cat] + cbf_inc, None if dec else int(coef is not None))
            if not cbf:
                return 0
        if dec:
            sig = []
            i = 0
            last = None
            while i < n - 1:
                si, li = self.sig_ctx(cat, i)
                if c.bin(si):
                    sig.append(i)
                    if c.bin(li):
                        last = i
                        break
                i += 1
            if last is None:
                sig.append(n - 1)
            levels = {}
            node = 0
            for pos in reversed(sig):
                if c.bin(ABS_BASE[cat] + ABS1_CTX[node]) == 0:
                    a = 1
                    node = TRANS_EQ1[node]
                else:
                    ctx = ABS_BASE[cat] + ABSGT1_CTX[node]
                    node = TRANS_GT1[node]
                    a = 2
                    while a < 15 and c.bin(ctx):
                        a += 1
                    if a >= 15:
                        k = 0
                        while c.bypass():
                            k += 1
                        v = 1
                        for _ in range(k):
                            v = 2 * v + c.bypass()
                        a = v + 14
                levels[pos] = -a if c.bypass() else a
            out = [0] * n
            for p, v in levels.items():
                out[p] = v
            mb['coef'][key] = out
            return 1
        sig = [i for i, v in enumerate(coef) if v]
        last = sig[-1]
        for i in range(min(last + 1, n - 1)):
            si, li = self.sig_ctx(cat, i)
            s = int(coef[i] != 0)
            c.bin(si, s)
            if s:
                c.bin(li, int(i == last))
        node = 0
        for pos in reversed(sig):
            v = coef[pos]
            a = abs(v)
            c.bin(ABS_BASE[cat] + ABS1_CTX[node], int(a > 1))
            if a == 1:
                node = TRANS_EQ1[node]
            else:
                ctx = ABS_BASE[cat] + ABSGT1_CTX[node]
                node = TRANS_GT1[node]
                m = a - 1  # coeff_abs_level_minus1, prefix TU cMax 14
                for _ in range(min(m, 14) - 1):
                    c.bin(ctx, 1)
                if m < 14:
                    c.bin(ctx, 0)
                else:
                    r = m - 14
                    k = 0
                    while r >= (1 << k):
                        c.bypass(1)
                        r -= 1 << k
                        k += 1
                    c.bypass(0)
                    while k:
                        k -= 1
                        c.bypass((r >> k) & 1)
            c.bypass(int(v < 0))
        return 1

    @staticmethod
    def sig_ctx(cat, i):
        if cat == 5:
            return 402 + SIG_8x8[i], 417 + LAST_8x8[i]
        if cat == 3:
            j = min(i, 2)
            return SIG_BASE[3] + j, LAST_BASE[3] + j
        return SIG_BASE[cat] + i, LAST_BASE[cat] + i

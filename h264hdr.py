"""Minimal H.264 Annex-B header parser (SPS / PPS / slice headers).

Parses just enough syntax to reason about reference structure: slice types,
nal_ref_idc, frame_num, POC, ref list modification, weighted prediction and
MMCO commands. Macroblock layer is not touched.
"""
import mmap
import re

SLICE_TYPES = {0: 'P', 1: 'B', 2: 'I', 3: 'SP', 4: 'SI'}


def unescape(b):
    """Strip emulation-prevention bytes (00 00 03 -> 00 00)."""
    return re.sub(rb'\x00\x00\x03', b'\x00\x00', b)


def escape(b):
    """Insert emulation-prevention bytes into an RBSP."""
    return re.sub(rb'\x00\x00(?=[\x00-\x03]|\Z)', b'\x00\x00\x03', b)


def set_bits(buf, pos, n, value):
    """Overwrite n bits at bit offset pos in a bytearray."""
    for i in range(n):
        bit = (value >> (n - 1 - i)) & 1
        byte, sh = (pos + i) >> 3, 7 - ((pos + i) & 7)
        buf[byte] = (buf[byte] & ~(1 << sh)) | (bit << sh)


class Bits:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def u(self, n):
        v = 0
        for _ in range(n):
            byte = self.d[self.p >> 3]
            v = (v << 1) | ((byte >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
        return (1 << z) - 1 + self.u(z)

    def se(self):
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)


def _skip_scaling_list(b, size):
    last = nxt = 8
    for _ in range(size):
        if nxt != 0:
            nxt = (last + b.se() + 256) % 256
        last = nxt if nxt != 0 else last


def parse_sps(rbsp):
    b = Bits(rbsp)
    s = {'profile_idc': b.u(8)}
    b.u(8)
    s['level_idc'] = b.u(8)
    s['id'] = b.ue()
    s['chroma_format_idc'] = 1
    if s['profile_idc'] in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        s['chroma_format_idc'] = b.ue()
        if s['chroma_format_idc'] == 3:
            s['separate_colour_plane'] = b.u(1)
        b.ue(); b.ue(); b.u(1)
        if b.u(1):  # seq_scaling_matrix_present
            for i in range(8 if s['chroma_format_idc'] != 3 else 12):
                if b.u(1):
                    _skip_scaling_list(b, 16 if i < 6 else 64)
    s['log2_max_frame_num'] = b.ue() + 4
    s['poc_type'] = b.ue()
    if s['poc_type'] == 0:
        s['log2_max_poc_lsb'] = b.ue() + 4
    elif s['poc_type'] == 1:
        s['delta_pic_order_always_zero'] = b.u(1)
        b.se(); b.se()
        for _ in range(b.ue()):
            b.se()
    s['max_num_ref_frames'] = b.ue()
    s['gaps_allowed'] = b.u(1)
    s['mb_w'] = b.ue() + 1
    s['mb_h'] = b.ue() + 1
    s['frame_mbs_only'] = b.u(1)
    if not s['frame_mbs_only']:
        s['mbaff'] = b.u(1)
    s['direct_8x8_inference'] = b.u(1)
    return s


def parse_pps(rbsp):
    b = Bits(rbsp)
    p = {'id': b.ue(), 'sps_id': b.ue()}
    p['cabac'] = b.u(1)
    p['bottom_field_poc_present'] = b.u(1)
    p['num_slice_groups'] = b.ue() + 1
    assert p['num_slice_groups'] == 1, 'FMO not supported'
    p['num_ref_idx_l0_default'] = b.ue() + 1
    p['num_ref_idx_l1_default'] = b.ue() + 1
    p['weighted_pred'] = b.u(1)
    p['weighted_bipred_idc'] = b.u(2)
    p['pic_init_qp'] = 26 + b.se()
    b.se()
    p['chroma_qp_offset'] = b.se()
    p['deblocking_ctrl_present'] = b.u(1)
    p['constrained_intra'] = b.u(1)
    p['redundant_pic_cnt_present'] = b.u(1)
    p['transform_8x8'] = 0
    p['chroma_qp_offset2'] = p['chroma_qp_offset']
    last1 = len(rbsp) * 8 - 1
    while not (rbsp[last1 >> 3] >> (7 - (last1 & 7))) & 1:
        last1 -= 1
    if b.p < last1:  # more_rbsp_data
        p['transform_8x8'] = b.u(1)
        p['scaling_matrix'] = b.u(1)
        assert not p['scaling_matrix'], 'PPS scaling matrices not supported'
        p['chroma_qp_offset2'] = b.se()
    return p


def parse_slice(rbsp, nal_type, nal_ref_idc, spss, ppss):
    b = Bits(rbsp)
    h = {'first_mb': b.ue()}
    st = b.ue()
    h['slice_type'] = SLICE_TYPES[st % 5]
    h['pps_id'] = b.ue()
    pps = ppss[h['pps_id']]
    sps = spss[pps['sps_id']]
    if sps.get('separate_colour_plane'):
        b.u(2)
    h['frame_num_bit'] = b.p  # bit offset in the RBSP (after the NAL header byte)
    h['frame_num'] = b.u(sps['log2_max_frame_num'])
    h['field_pic'] = 0
    if not sps['frame_mbs_only']:
        h['field_pic'] = b.u(1)
        if h['field_pic']:
            h['bottom_field'] = b.u(1)
    if nal_type == 5:
        h['idr_pic_id'] = b.ue()
    if sps['poc_type'] == 0:
        h['poc_lsb_bit'] = b.p
        h['poc_lsb'] = b.u(sps['log2_max_poc_lsb'])
        if pps['bottom_field_poc_present'] and not h['field_pic']:
            h['delta_poc_bottom'] = b.se()
    elif sps['poc_type'] == 1 and not sps['delta_pic_order_always_zero']:
        h['delta_poc0'] = b.se()
        if pps['bottom_field_poc_present'] and not h['field_pic']:
            h['delta_poc1'] = b.se()
    if pps['redundant_pic_cnt_present']:
        b.ue()
    t = h['slice_type']
    if t == 'B':
        h['direct_spatial'] = b.u(1)
    h['num_ref_l0'] = pps['num_ref_idx_l0_default']
    h['num_ref_l1'] = pps['num_ref_idx_l1_default']
    if t in ('P', 'SP', 'B'):
        if b.u(1):
            h['num_ref_l0'] = b.ue() + 1
            if t == 'B':
                h['num_ref_l1'] = b.ue() + 1
    if t not in ('B',):
        h['num_ref_l1'] = 0
    if t in ('I', 'SI'):
        h['num_ref_l0'] = 0
    # ref_pic_list_modification
    h['rplm'] = [[], []]
    for lst in (0, 1):
        if (lst == 0 and t not in ('I', 'SI')) or (lst == 1 and t == 'B'):
            if b.u(1):
                while True:
                    idc = b.ue()
                    if idc == 3:
                        break
                    h['rplm'][lst].append((idc, b.ue()))
    # pred_weight_table
    h['weighted'] = False
    if (pps['weighted_pred'] and t in ('P', 'SP')) or (pps['weighted_bipred_idc'] == 1 and t == 'B'):
        h['weighted'] = True
        b.ue()
        if sps['chroma_format_idc'] != 0:
            b.ue()
        for lst, n in ((0, h['num_ref_l0']), (1, h['num_ref_l1'])):
            for _ in range(n):
                if b.u(1):
                    b.se(); b.se()
                if sps['chroma_format_idc'] != 0 and b.u(1):
                    b.se(); b.se(); b.se(); b.se()
    # dec_ref_pic_marking
    h['mmco'] = []
    if nal_ref_idc:
        if nal_type == 5:
            h['no_output_prior'] = b.u(1)
            h['long_term_ref'] = b.u(1)
        elif b.u(1):
            while True:
                op = b.ue()
                if op == 0:
                    break
                args = []
                if op in (1, 3):
                    args.append(b.ue())
                if op == 2:
                    args.append(b.ue())
                if op in (3, 6):
                    args.append(b.ue())
                if op == 4:
                    args.append(b.ue())
                h['mmco'].append((op, *args))
    h['cabac_init_idc'] = 0
    if pps['cabac'] and t not in ('I', 'SI'):
        h['cabac_init_idc'] = b.ue()
    h['qp'] = pps['pic_init_qp'] + b.se()
    h['deblock_idc'], h['alpha_off'], h['beta_off'] = 0, 0, 0
    if pps['deblocking_ctrl_present']:
        h['deblock_idc'] = b.ue()
        if h['deblock_idc'] != 1:
            h['alpha_off'] = 2 * b.se()
            h['beta_off'] = 2 * b.se()
    h['hdr_bits'] = b.p
    h['data_bit'] = (b.p + 7) & ~7 if pps['cabac'] else b.p  # cabac_alignment_one_bits
    return h


def iter_nals(path):
    """Yield (offset, nal_bytes_view) for every NAL in an Annex-B file."""
    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        starts = [m.end() for m in re.finditer(rb'\x00\x00\x01', mm)]
        for i, s in enumerate(starts):
            e = starts[i + 1] - 3 if i + 1 < len(starts) else len(mm)
            # trailing zero of a 4-byte start code belongs to the next NAL
            while e > s and mm[e - 1] == 0:
                e -= 1
            yield s, e, mm


def parse_stream(path, head_bytes=512):
    """Return list of pictures in decode order with slice header info."""
    spss, ppss, pics = {}, {}, []
    for s, e, mm in iter_nals(path):
        hdr = mm[s]
        nal_type = hdr & 0x1f
        ref_idc = (hdr >> 5) & 3
        if nal_type == 7:
            sps = parse_sps(unescape(mm[s + 1:e]))
            spss[sps['id']] = sps
        elif nal_type == 8:
            pps = parse_pps(unescape(mm[s + 1:e]))
            ppss[pps['id']] = pps
        elif nal_type in (1, 5):
            h = parse_slice(unescape(mm[s + 1:min(e, s + 1 + head_bytes)]), nal_type, ref_idc, spss, ppss)
            h.update(nal_type=nal_type, ref_idc=ref_idc, offset=s, size=e - s)
            if h['first_mb'] == 0:
                pics.append({'slices': [h], 'idr': nal_type == 5, 'ref_idc': ref_idc})
            else:
                pics[-1]['slices'].append(h)
    return spss, ppss, pics

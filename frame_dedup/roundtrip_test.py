"""Round-trip check for h264cabac: decode slice data, re-encode, compare bytes.

Usage: python roundtrip_test.py <in.h264> <decode-order index> [...]
       python roundtrip_test.py <in.h264> --types N   (first N of each type)
"""
import mmap
import sys
import time

from .h264cabac import Coder, SliceCodec
from .h264hdr import escape, parse_stream, unescape


def to_bits(b):
    return [(byte >> (7 - k)) & 1 for byte in b for k in range(8)]


def from_bits(bits):
    out = bytearray()
    for i in range(0, len(bits), 8):
        v = 0
        for b in bits[i:i + 8]:
            v = (v << 1) | b
        out.append(v)
    return bytes(out)


def load_slice(mm, sl):
    nal = bytes(mm[sl['offset']:sl['offset'] + sl['size']])
    return nal, unescape(nal[1:])


def decode_picture(mm, spss, ppss, pic):
    sl = pic['slices'][0]
    nal, rbsp = load_slice(mm, sl)
    pps = ppss[sl['pps_id']]
    sps = spss[pps['sps_id']]
    bits = to_bits(rbsp)
    c = Coder('dec', sl['qp'], sl['cabac_init_idc'], sl['slice_type'] == 'I', bits, sl['data_bit'])
    codec = SliceCodec(sps, pps, sl)
    mbs = codec.run(c)
    return dict(nal=nal, rbsp=rbsp, bits=bits, end=c.pos, mbs=mbs, sps=sps, pps=pps, sl=sl)


def encode_picture(d, mbs=None):
    sl = d['sl']
    c = Coder('enc', sl['qp'], sl['cabac_init_idc'], sl['slice_type'] == 'I')
    SliceCodec(d['sps'], d['pps'], sl).run(c, mbs if mbs is not None else d['mbs'])
    bits = d['bits'][:sl['data_bit']] + c.out
    while len(bits) % 8:
        bits.append(0)
    rbsp = from_bits(bits)
    return d['nal'][:1] + escape(rbsp), c.out


def check(mm, spss, ppss, pic):
    t0 = time.time()
    d = decode_picture(mm, spss, ppss, pic)
    t1 = time.time()
    sl = d['sl']
    # the slice must end on its stop bit, then zero alignment, then only
    # cabac_zero_words (0x0000) if any
    tail = d['bits'][d['end']:]
    stop_ok = d['bits'][d['end'] - 1] == 1 and not any(tail)
    nal2, enc_bits = encode_picture(d)
    t2 = time.time()
    same_bits = enc_bits == d['bits'][sl['data_bit']:d['end']]
    same_nal = nal2 == d['nal']
    kinds = {}
    for mb in d['mbs']:
        kinds[mb['kind']] = kinds.get(mb['kind'], 0) + 1
    print(f"dec {pic['dec']:5d} {sl['slice_type']} {len(d['nal']):7d}B  stop_bit_ok={stop_ok} "
          f"tail_bits={len(tail)} bits_equal={same_bits} nal_equal={same_nal}  "
          f"parse {t1 - t0:.1f}s enc {t2 - t1:.1f}s  {kinds}")
    return stop_ok and same_bits and same_nal


def main():
    es = sys.argv[1]
    spss, ppss, pics = parse_stream(es)
    for d, p in enumerate(pics):
        p['dec'] = d
    f = open(es, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    if sys.argv[2] == '--types':
        n = int(sys.argv[3])
        sel, seen = [], {}
        for p in pics[1:]:
            t = p['slices'][0]['slice_type']
            if seen.get(t, 0) < n and p['slices'][0]['size'] > 5000:
                sel.append(p)
                seen[t] = seen.get(t, 0) + 1
    else:
        sel = [pics[int(a)] for a in sys.argv[2:]]
    ok = all([check(mm, spss, ppss, p) for p in sel])
    print('ALL OK' if ok else 'FAILURES')


if __name__ == '__main__':
    main()

"""Remove inserted duplicate frames from an H.264 Annex-B stream, no re-encode.

For every duplicate pair (twins <=5ms apart in the 1ms-retimed MKV):
  * if a twin is a non-reference picture, drop it (bit-exact for every other
    frame);
  * otherwise (both twins are reference P/I pictures) drop the later twin and
    renumber frame_num on every following picture up to the next IDR, so the
    decoder sees no gap. Default ref list construction then makes anything
    that pointed at the dropped twin point at the kept twin instead.

  * --keep-second: for hard pairs drop the earlier twin and keep the later
    one instead.
  * --steal-poc[=list]: give the kept twin of a hard pair the dropped twin's
    POC (they are adjacent in display order, so output order is unchanged).
    Pictures after the pair then see the POC distances they were encoded
    with (temporal direct / implicit weights); the B pictures that use the
    kept twin as their L1 anchor see a shifted distance instead. An optional
    comma list restricts it to those dropped display indices.
    Tested on sample.mkv: never better. Post-cut differences come from real
    twin content differences, not POC distance, and the pre-pair B pictures
    get badly damaged (up to 170 levels). Kept only as a record.

Usage: python dedup.py <in.h264> <packets.csv> <out.h264> <report.json> [--keep-second] [--steal-poc[=i,j]]
"""
import csv
import json
import mmap
import sys

from h264hdr import escape, parse_stream, set_bits, unescape


def nal_start(mm, offset):
    """Offset of the start code in front of a NAL whose payload starts at offset."""
    s = offset - 3
    if s > 0 and mm[s - 1] == 0:
        s -= 1
    return s


def main(es, pk_csv, out, report_path, *opts):
    spss, ppss, pics = parse_stream(es)
    log2_fn = spss[0]['log2_max_frame_num']
    pts = [int(r[0]) for r in csv.reader(open(pk_csv))]
    assert len(pts) == len(pics)
    for d, (p, t) in enumerate(zip(pics, pts)):
        p['pts'], p['dec'] = t, d
    disp = sorted(pics, key=lambda p: p['pts'])
    for i, p in enumerate(disp):
        p['idx'] = i

    drop, hard = set(), []
    for i in range(len(disp) - 1):
        a, b = disp[i], disp[i + 1]
        if b['pts'] - a['pts'] > 5:
            continue
        if not b['ref_idc']:
            drop.add(b['dec'])
        elif not a['ref_idc']:
            drop.add(a['dec'])
        else:
            if '--keep-second' in opts:
                a, b = b, a
            drop.add(b['dec'])
            hard.append({'kept_idx': a['idx'], 'dropped_idx': b['idx'],
                         'dropped_dec': b['dec']})

    # frame_num shift per picture: number of dropped reference pictures since
    # the last IDR, in decode order
    shift, k = {}, 0
    for p in pics:
        if p['idr']:
            k = 0
        if p['dec'] in drop:
            if p['ref_idc']:
                k += 1
            continue
        if k:
            shift[p['dec']] = k

    # bit patches per picture: dec -> [(field, value)]
    patches = {d: [('frame_num', k)] for d, k in shift.items()}
    steal = None
    for o in opts:
        if o.startswith('--steal-poc'):
            steal = set(map(int, o.split('=')[1].split(','))) if '=' in o else 'all'
    if steal:
        log2_poc = spss[0]['log2_max_poc_lsb']
        for h in hard:
            if steal != 'all' and h['dropped_idx'] not in steal:
                continue
            kept, dropped = disp[h['kept_idx']], disp[h['dropped_idx']]
            patches.setdefault(kept['dec'], []).append(('poc_lsb', dropped['slices'][0]['poc_lsb']))
            h['stole_poc'] = True

    # attach the window each hard drop can influence: from the dropped twin to
    # the next IDR in decode order
    for h in hard:
        d = h['dropped_dec']
        nxt = next((q['dec'] for q in pics[d + 1:] if q['idr']), len(pics))
        h['affected_display_idx'] = sorted(q['idx'] for q in pics[d + 1:nxt] if q['dec'] not in drop)
        h['next_idr_display_idx'] = pics[nxt]['idx'] if nxt < len(pics) else None
        h['kept_type'] = disp[h['kept_idx']]['slices'][0]['slice_type']

    patched = 0
    with open(es, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        with open(out, 'wb') as o:
            pos = 0
            for p in pics:
                first = nal_start(mm, p['slices'][0]['offset'])
                o.write(mm[pos:first])  # parameter sets etc. ride along
                last = p['slices'][-1]
                pos = last['offset'] + last['size']
                if p['dec'] in drop:
                    continue
                if p['dec'] not in patches:
                    o.write(mm[first:pos])
                    continue
                for sl in p['slices']:
                    s = nal_start(mm, sl['offset'])
                    o.write(mm[s:sl['offset'] + 1])  # start code + NAL header
                    rbsp = bytearray(unescape(mm[sl['offset'] + 1:sl['offset'] + sl['size']]))
                    for field, v in patches[p['dec']]:
                        if field == 'frame_num':
                            set_bits(rbsp, sl['frame_num_bit'], log2_fn, (sl['frame_num'] - v) % (1 << log2_fn))
                        else:
                            set_bits(rbsp, sl['poc_lsb_bit'], spss[0]['log2_max_poc_lsb'], v)
                    o.write(escape(bytes(rbsp)))
                    patched += 1
            o.write(mm[pos:])

    disp_kept = [p['idx'] for p in disp if p['dec'] not in drop]
    json.dump({'dropped_display_idx': sorted(pics[d]['idx'] for d in drop),
               'kept_display_idx': disp_kept, 'hard': hard},
              open(report_path, 'w'))
    print(f'pictures {len(pics)}  dropped {len(drop)} (hard {len(hard)})  '
          f'slices renumbered {patched}  remaining {len(disp_kept)}')


if __name__ == '__main__':
    main(*sys.argv[1:])

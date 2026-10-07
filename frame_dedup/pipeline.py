"""End-to-end pipeline: MKV in, constant-frame-rate deduplicated MKV out."""
import json
import os
import time
from fractions import Fraction

import numpy as np

from . import __version__
from .cadence import (FAIL_SCORE, INPUT_RATES, MIN_EVIDENCE_DUPS, SOURCE_RATES, cadence_score, detect,
                      hard_cuts, predicted_dups, rate_name, segments_from)
from .evidence import black_runs, evidence, load_model, proxy_features, static_mask
from .mux import duration_check, mux
from .plan import choose_drops, offsets, sync_correct, sync_stats
from .rebuild import candidate, rebuild
from .repair import repair
from .stream import Unsupported, extract_annexb, free_drop_checker, input_rate, load, write_stream
from .system import Logger, peak_rss, physical_ram
from .verify import verify


class NoCadence(Exception):
    """No duplicate-frame cadence was found."""


def _fps_str(fr):
    return f'{float(fr):.3f}'.rstrip('0').rstrip('.')


class Run:
    def __init__(self, args, log=None):
        self.a = args
        os.makedirs(args.work, exist_ok=True)
        self.log0 = log or Logger(os.path.join(args.work, 'run.log'))
        self.t0 = time.time()
        self.times = {}
        self.report = {'tool': 'frame_dedup', 'version': __version__, 'input': os.path.abspath(args.input),
                       'output': os.path.abspath(args.output) if args.output else None}

    def log(self, msg):
        self.log0(msg)

    def stage(self, name, t):
        self.times[name] = round(time.time() - t, 1)

    # ------------------------------------------------------------------
    def analyse(self):
        a = self.a
        os.makedirs(a.work, exist_ok=True)
        es = os.path.join(a.work, 'in.h264')
        t = time.time()
        self.log(f'extracting video stream from {a.input}')
        extract_annexb(a.input, es)
        spss, ppss, pics = load(es)
        self.es, self.spss, self.ppss, self.pics = es, spss, ppss, pics
        N = len(pics)
        num, den = input_rate(spss)
        fps_in = Fraction(num, den)
        in_name = rate_name(fps_in, INPUT_RATES)
        self.report['input_rate'] = {'fps': _fps_str(fps_in), 'exact': str(fps_in), 'source': 'SPS VUI timing_info',
                                     'num_units_in_tick': spss[0]['vui']['timing']['num_units_in_tick'],
                                     'time_scale': spss[0]['vui']['timing']['time_scale']}
        self.report['frames_in'] = N
        self.log(f'{N} pictures, input rate {_fps_str(fps_in)} fps ({fps_in}) from the SPS VUI, '
                 f'{sum(p["idr"] for p in pics)} IDR periods')
        if in_name is None:
            raise Unsupported(f'input rate {_fps_str(fps_in)} fps is not 29.97 or 30; nothing to do')
        self.fps_in = fps_in
        self.stage('extract_parse', t)

        t = time.time()
        feat_path = os.path.join(a.work, 'features.npz')
        feat = None
        if os.path.exists(feat_path):
            feat = dict(np.load(feat_path))
            if len(feat['mad']) != N or int(feat.get('es_size', -1)) != os.path.getsize(es):
                feat = None
        if feat is None:
            self.log('decoding the low-res luma proxy')
            feat = proxy_features(es, threads=min(8, a.workers))
            if len(feat['mad']) != N:
                raise RuntimeError(f'proxy decode produced {len(feat["mad"])} frames for {N} pictures')
            np.savez(feat_path, es_size=os.path.getsize(es), **feat)
        self.stage('proxy_decode', t)

        t = time.time()
        model, meta = load_model()
        e = evidence(feat, model)
        static = static_mask(feat['bmax'])
        black = black_runs(feat, float(fps_in))
        cuts = hard_cuts(feat, static)
        segs = segments_from(N, black, cuts)
        cands = detect(e, segs, fps_in)
        best = cands[0]
        dup = predicted_dups(N, segs, best)
        score, n_ev = cadence_score(e, static, dup)
        det = {'model': meta.get('trained_on'), 'black_runs': [
            {'start': s, 'end': x, 'start_s': round(s / float(fps_in), 3), 'frames': x - s} for s, x in black],
            'hard_cuts': len(cuts), 'segments': len(segs), 'static_frames': int(static.sum()),
            'candidates': [], 'cadence_score': round(score, 3), 'fail_threshold': FAIL_SCORE,
            'duplicates_with_evidence': n_ev}
        for c in cands:
            cd = predicted_dups(N, segs, c)
            det['candidates'].append({'source': c['source'], 'ratio': str(c['r']), 'total_llr': round(c['total'], 1),
                                      'cadence_score': round(cadence_score(e, static, cd)[0], 3),
                                      'phase_changes': sum(1 for i in range(1, len(segs))
                                                           if c['phases'][i] != c['phases'][i - 1])})
        self.report['detection'] = det
        for c in det['candidates']:
            self.log(f"  candidate {c['source']:>6} fps (r={c['ratio']}): total LLR {c['total_llr']:10.1f}, "
                     f"score {c['cadence_score']:.2f}, {c['phase_changes']} phase changes")
        self.stage('detect', t)
        if score < FAIL_SCORE or n_ev < MIN_EVIDENCE_DUPS or best['total'] <= 0:
            raise NoCadence(
                f'no duplicate-frame cadence found: the best model ({best["source"]} fps in {in_name} fps) '
                f'scores {score:.2f} (mean evidence of its predicted duplicates above a random phase; '
                f'needs >= {FAIL_SCORE}, real cadences score about 10). This looks like native '
                f'{in_name} fps content; no output written.')

        fps_out = SOURCE_RATES[best['source']]
        self.fps_out, self.best, self.segs, self.black, self.dup = fps_out, best, segs, black, dup
        self.feat = feat
        r = best['r']
        self.log(f'detected source rate {best["source"]} fps (r = {r}), cadence score {score:.2f}')

        # chunks (split at black runs) and phase changes
        chunks, changes = [], []
        for i, ((s, x, kind), m) in enumerate(zip(segs, best['phases'])):
            if kind in ('start', 'black') or not chunks:
                chunks.append({'start': s, 'end': x, 'start_s': round(s / float(fps_in), 3), 'phases': [m]})
            else:
                chunks[-1]['end'] = x
                if m != chunks[-1]['phases'][-1]:
                    chunks[-1]['phases'].append(m)
            if i and m != best['phases'][i - 1]:
                changes.append({'display': s, 'time_s': round(s / float(fps_in), 3), 'at': kind,
                                'from_phase': best['phases'][i - 1], 'to_phase': m, 'q': best['q']})
        for ch in chunks:
            ch['frames'] = ch['end'] - ch['start']
            ch['phi'] = [round(m / best['q'], 6) for m in ch['phases']]
        self.report['cadence'] = {'source_rate': best['source'], 'output_fps': str(fps_out), 'ratio': str(r),
                                  'phase_bins': best['q'], 'chunks': chunks, 'phase_changes': changes}
        for c in changes:
            self.log(f"  phase change at frame {c['display']} ({c['time_s']} s, {c['at']}): "
                     f"{c['from_phase']}/{c['q']} -> {c['to_phase']}/{c['q']}")
        self.log(f'{len(chunks)} chunks, {len(changes)} phase changes, {int(dup.sum())} duplicates')
        return True

    # ------------------------------------------------------------------
    def plan(self):
        t = time.time()
        pics, N = self.pics, len(self.pics)
        disp = [None] * N
        for p in pics:
            disp[p['disp']] = p
        self.disp = disp
        rebuildable = None
        if not getattr(self.a, 'no_rebuild', False):
            periods = {}
            for p in pics:
                periods.setdefault(p['period'], []).append(p)
            rebuildable = lambda K, D: candidate(self.spss, pics, periods, K, D)
        drop, hard, unsupported, freed, subs, rebuilds = choose_drops(
            disp, self.dup, free_drop_checker(self.spss, pics), self.feat['mad'], self.feat['bmax'],
            visible_subs=not getattr(self.a, 'keep_idr_pairs', False), rebuildable=rebuildable)
        fallbacks = [r for r in rebuilds if 'fallback' in r]
        rebuilds = [r for r in rebuilds if 'fallback' not in r]
        n_planned = len(drop)
        n_nonref = len(drop) - len(hard) - len(freed) - len(subs) - len(rebuilds)
        events = [c['display'] for c in self.report['cadence']['phase_changes'] if c['at'] == 'cut']
        events += [u['display'] for u in unsupported]
        drop, corrections = sync_correct(disp, drop, self.best['r'], self.black, events)
        hard = [h for h in hard if h['dropped_display'] in drop]
        rebuilds = [r for r in rebuilds if r['dropped_display'] in drop and r['kept_display'] not in drop]
        for f in fallbacks:
            self.log(f"  anchor rebuild not possible at display {f['pair']} ({f['types']}): {f['fallback']}")
        for u in unsupported:
            self.log(f"  unsupported pair at display {u['display']} ({u['types']}): {u['reason']}; both kept")
        for c in corrections:
            self.log(f"  sync correction in black run {c['black_run']}: {c['action']}")
        self.drop = drop
        self.hard = hard
        self.rebuilds = rebuilds
        st = sync_stats(N, drop, self.best['r'], float(self.fps_out))
        off = offsets(N, drop, self.best['r'])
        chunk_off = []
        for ch in self.report['cadence']['chunks']:
            seg = off[ch['start']:ch['end']]
            if np.isfinite(seg).any():
                chunk_off.append({'start': ch['start'], 'mean_offset_frames': round(float(np.nanmean(seg)), 3),
                                  'max_abs_offset_frames': round(float(np.nanmax(np.abs(seg))), 3)})
        st['chunks'] = chunk_off
        self.report['drops'] = {'total': len(drop), 'duplicates': int(self.dup.sum()),
                                'non_reference': n_nonref, 'unreferenced_reference': len(freed),
                                'substitute_invisible': sum(1 for x in subs if not x['visible']),
                                'substitute_visible': sum(1 for x in subs if x['visible']),
                                'hard': len(hard), 'anchor_rebuild': len(rebuilds),
                                'anchor_rebuild_fallbacks': len(fallbacks), 'sync_net_change': len(drop) - n_planned}
        self.report['unreferenced_reference_drops'] = freed
        self.report['substitute_drops'] = subs
        self.report['hard_pairs'] = hard
        self.report['anchor_rebuild_fallbacks'] = fallbacks
        self.report['unsupported_pairs'] = unsupported
        self.report['sync'] = {'corrections': corrections, **st}
        self.log(f'dropping {len(drop)} pictures ({len(freed)} unreferenced reference twins, '
                 f'{len(rebuilds)} anchor rebuilds ({len(fallbacks)} fell back), '
                 f'{len(subs)} substitutes ({sum(1 for x in subs if x["visible"])} visible), {len(hard)} hard pairs, '
                 f'{len(unsupported)} unsupported pairs, '
                 f'{len(corrections)} sync corrections); output {N - len(drop)} frames, '
                 f'final A/V offset {st["final_offset_frames"]} frames, max {st["max_abs_offset_ms"]} ms')
        self.stage('plan', t)

    # ------------------------------------------------------------------
    def write(self):
        a = self.a
        fo = self.fps_out
        timing = (fo.denominator, 2 * fo.numerator)
        t = time.time()
        stage1 = os.path.join(a.work, 'dedup.h264')
        # rebuilt pairs drop K in a third stage, after the hard-drop repair
        rb_K = {r['K_dec'] for r in self.rebuilds}
        drop_dec = {self.disp[i]['dec'] for i in self.drop} - rb_K
        n_renum, n_sps = write_stream(self.es, self.pics, self.spss, stage1, drop=drop_dec, timing=timing)
        self.report['sps_timing'] = {'num_units_in_tick': timing[0], 'time_scale': timing[1], 'sps_patched': n_sps}
        self.log(f'wrote {stage1}: {n_renum} slices renumbered, {n_sps} SPS timing patched to {timing[0]}/{timing[1]}')
        self.stage('write_dedup', t)
        out_es = stage1
        rep = {'pictures_rewritten': 0, 'pcm_mbs': 0, 'bytes_added': 0}
        if self.hard:
            t = time.time()
            nspss, nppss, npics = load(stage1)
            repl, rep = repair(self.es, stage1, self.spss, self.ppss, self.pics, npics, drop_dec,
                               a.workers, log=self.log)
            if repl:
                out_es = os.path.join(a.work, 'out.h264')
                write_stream(stage1, npics, nspss, out_es, repl=repl)
            self.log(f"repair: {rep['pictures_rewritten']} pictures rewritten, {rep['pcm_mbs']} PCM MBs, "
                     f"{rep['bytes_added']:+d} bytes")
            self.stage('repair', t)
        self.report['repair'] = rep
        src_orig = [p['dec'] for p in self.pics if p['dec'] not in drop_dec]  # stage-1/2 decode order
        self.orig_of_out = src_orig
        if self.rebuilds:
            t = time.time()
            nspss, nppss, src_pics = load(out_es)
            src_of = {d: i for i, d in enumerate(src_orig)}
            rbs = [{'K': src_of[r['K_dec']], 'D': src_of[r['D_dec']],
                    'B': [src_of[b] for b in r['B_dec'] if b in src_of]} for r in self.rebuilds]
            rb_es = os.path.join(a.work, 'rebuilt.h264')
            self.orig_of_out, rb = rebuild(self.es, out_es, rb_es, nspss, nppss, self.pics, src_pics, src_orig, rbs,
                                           a.workers, a.work, log=self.log)
            self.report['anchor_rebuilds'] = rb
            self.log(f"anchor rebuild: {rb['pairs']} pairs, {rb['leading_b_pictures']} leading B pictures, "
                     f"{rb['pcm_mbs']} PCM MBs, {rb['direct_explicit']} direct MBs made explicit, "
                     f"{rb['bytes_added']:+d} bytes, worst difference in the leading B pictures "
                     f"{rb['worst_in_leading_b']}")
            out_es = rb_es
            self.stage('rebuild', t)
        self.out_es = out_es

    # ------------------------------------------------------------------
    def remux(self):
        a = self.a
        t = time.time()
        tmp = a.output + '.partial.mkv'
        if os.path.exists(tmp):
            os.remove(tmp)
        self.log(f'muxing {a.output}')
        warnings = mux(a.input, self.out_es, tmp, self.fps_out, log=self.log)
        os.replace(tmp, a.output)
        n_out = len(self.pics) - len(self.drop)
        self.report['output_stream'] = {'fps': _fps_str(self.fps_out), 'exact': str(self.fps_out), 'frames': n_out,
                                        'mkvmerge_messages': warnings}
        dc = duration_check(a.input, a.output, n_out, self.fps_out)
        self.report['durations'] = dc
        self.log(f"durations: video {dc.get('output_video_s')} s, audio {dc.get('output_audio_s')} s "
                 f"(input video {dc.get('input_video_s')} s, audio {dc.get('input_audio_s')} s); "
                 f"A/V gap change {dc.get('av_gap_change_s')} s")
        self.stage('mux', t)

    # ------------------------------------------------------------------
    def verify(self):
        t = time.time()
        self.log('verifying: decoding output and original')
        ospss, _, opics = load(self.out_es)
        res = verify(self.es, self.out_es, self.spss, self.pics, opics, self.a.workers, log=self.log,
                     orig_of_out=self.orig_of_out)
        # hard-drop neighbourhoods: frames after a dropped reference twin, up
        # to the end of its IDR period
        hood = set()
        period_end = {}
        for p in self.pics:
            period_end[p['period']] = max(period_end.get(p['period'], -1), p['disp'])
        for h in self.hard:
            D = self.pics[h['dropped_dec']]
            hood.update(range(D['disp'] + 1, period_end[D['period']] + 1))
        # leading B pictures of rebuilt pairs (they read I2 where they read K)
        lead = {self.pics[b]['disp'] for r in self.rebuilds for b in r['B_dec']} - set(self.drop)
        in_lead = [d for d in res['differing_frames'] if d['display'] in lead]
        res['rebuild_leading_b'] = {'pictures': len(lead), 'differing': len(in_lead),
                                    'worst': max((d['worst'] for d in in_lead), default=0),
                                    'mbs_ge3': sum(d['mbs_ge3'] for d in in_lead),
                                    'worst_luma': max((d['worst_luma'] for d in in_lead), default=0),
                                    'mbs_ge3_luma': sum(d['mbs_ge3_luma'] for d in in_lead)}
        outside = [d for d in res['differing_frames'] if d['display'] not in hood and d['display'] not in lead]
        res['differing_outside_hard_neighbourhoods'] = len(outside)
        res['worst_outside'] = max((d['worst'] for d in outside), default=0)
        res['ok'] = not outside and not res['errors'] and res['frames_compared'] == res['frames_output']
        self.report['verify'] = res
        self.log(f"verify: {res['frames_compared']} frames compared, {res['bit_exact']} bit-exact, "
                 f"{res['differing']} differ (worst {res['worst_difference']}, {res['mbs_ge3']} MBs >= 3; "
                 f"luma worst {res['worst_luma']}, {res['mbs_ge3_luma']} MBs >= 3), "
                 f"{len(outside)} differing outside hard-drop neighbourhoods and rebuilt leading B pictures "
                 f"(leading B: {res['rebuild_leading_b']['differing']} differ, worst "
                 f"{res['rebuild_leading_b']['worst']}, {res['rebuild_leading_b']['mbs_ge3']} MBs >= 3); "
                 f"errors {len(res['errors'])}")
        self.stage('verify', t)


def run(args, log=None):
    """Returns the process exit code."""
    R = Run(args, log)
    code = 0
    total, _ = physical_ram()
    R.report['resources'] = {'workers': args.workers, 'cpu_count': os.cpu_count(),
                             'physical_ram_gb': round(total / 2**30, 1)}
    R.log(f'frame_dedup {__version__}: {args.workers} workers, {os.cpu_count()} logical CPUs, '
          f'{total / 2**30:.1f} GB RAM, below-normal priority')
    try:
        R.analyse()
        R.plan()
        if args.dry_run:
            R.report['status'] = 'dry_run'
        else:
            R.write()
            R.remux()
            if args.verify:
                R.verify()
            R.report['status'] = 'ok' if (not args.verify or R.report['verify']['ok']) else 'verify_failed'
            code = 0 if R.report['status'] == 'ok' else 4
    except NoCadence as e:
        R.report['status'] = 'no_cadence'
        R.report['message'] = str(e)
        R.log(f'FAIL: {e}')
        code = 2
    except Unsupported as e:
        R.report['status'] = 'unsupported'
        R.report['message'] = str(e)
        R.log(f'FAIL: {e}')
        code = 3
    finally:
        R.report['runtime_s'] = {**R.times, 'total': round(time.time() - R.t0, 1)}
        R.report['resources']['main_peak_rss_mb'] = round(peak_rss() / 2**20)
        R.log(f"done: status {R.report.get('status')}, main process peak RSS "
              f"{R.report['resources']['main_peak_rss_mb']} MB, total {R.report['runtime_s']['total']} s")
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(R.report, f, indent=1, default=str)
        if not args.keep_work:
            cleanup(args.work)
    return code


WORK_FILES = ('in.h264', 'dedup.h264', 'out.h264', 'rebuilt.h264', 'features.npz')  # run.log is kept


def cleanup(work):
    """Remove only what this tool created, then the directory if empty."""
    for fn in WORK_FILES:
        p = os.path.join(work, fn)
        if os.path.exists(p):
            os.remove(p)
    try:
        os.rmdir(work)
    except OSError:
        pass

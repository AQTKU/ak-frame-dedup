"""Remux the deduplicated stream with the original's other tracks (mkvmerge)."""
import json
import subprocess

from .system import popen_kwargs


def identify(mkv):
    r = subprocess.run(['mkvmerge', '-J', mkv], capture_output=True, text=True, encoding='utf-8', **popen_kwargs())
    if r.returncode not in (0, 1):
        raise RuntimeError(f'mkvmerge -J failed: {r.stdout[:300]} {r.stderr[:300]}')
    return json.loads(r.stdout)


def mux(in_mkv, es, out_mkv, fps_out, log=print):
    """fps_out: Fraction. Video from `es` first, then everything except video
    from the input (audio, subtitles, chapters, tags, attachments)."""
    info = identify(in_mkv)
    vids = [t for t in info['tracks'] if t['type'] == 'video']
    vp = vids[0]['properties'] if vids else {}
    rate = f'{fps_out.numerator}/{fps_out.denominator}p' if fps_out.denominator != 1 else f'{fps_out.numerator}p'
    cmd = ['mkvmerge', '-q', '-o', out_mkv]
    title = info.get('container', {}).get('properties', {}).get('title')
    if title:
        cmd += ['--title', title]
    lang = vp.get('language_ietf') or vp.get('language')
    if lang:
        cmd += ['--language', f'0:{lang}']
    if vp.get('track_name'):
        cmd += ['--track-name', f"0:{vp['track_name']}"]
    if 'default_track' in vp:
        cmd += ['--default-track-flag', f"0:{int(vp['default_track'])}"]
    if 'forced_track' in vp:
        cmd += ['--forced-display-flag', f"0:{int(vp['forced_track'])}"]
    if vp.get('display_dimensions'):
        cmd += ['--display-dimensions', f"0:{vp['display_dimensions']}"]
    cmd += ['--default-duration', f'0:{rate}', es, '--no-video', in_mkv]
    log('  ' + ' '.join(f'"{c}"' if ' ' in c else c for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', **popen_kwargs())
    if r.returncode >= 2:
        raise RuntimeError(f'mkvmerge failed: {r.stdout[-500:]} {r.stderr[-500:]}')
    return [l for l in (r.stdout + r.stderr).splitlines() if l.strip()]


def track_durations(mkv):
    """{(type, index): seconds} from the per-track statistics tags."""
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=index,codec_type:stream_tags=DURATION',
                        '-of', 'json', mkv], capture_output=True, text=True, encoding='utf-8', **popen_kwargs())
    out = {}
    for s in json.loads(r.stdout or '{}').get('streams', []):
        d = (s.get('tags') or {}).get('DURATION')
        if d:
            h, m, sec = d.split(':')
            out[f"{s['codec_type']}{s['index']}"] = int(h) * 3600 + int(m) * 60 + float(sec)
    return out


def duration_check(in_mkv, out_mkv, n_out, fps_out):
    """Compare video duration with the first audio track's, before and after."""
    din, dout = track_durations(in_mkv), track_durations(out_mkv)

    def pick(d, kind):
        ks = sorted(k for k in d if k.startswith(kind))
        return d[ks[0]] if ks else None
    frame = 1 / float(fps_out)
    res = {'input_video_s': pick(din, 'video'), 'input_audio_s': pick(din, 'audio'),
           'output_video_s': pick(dout, 'video'), 'output_audio_s': pick(dout, 'audio'),
           'output_video_frames_s': round(n_out * frame, 4), 'frame_s': round(frame, 5)}
    if res['output_video_s'] is not None and res['output_audio_s'] is not None:
        res['output_av_diff_s'] = round(res['output_video_s'] - res['output_audio_s'], 4)
        res['output_av_diff_frames'] = round(res['output_av_diff_s'] / frame, 2)
    if res['input_video_s'] is not None and res['input_audio_s'] is not None:
        res['input_av_diff_s'] = round(res['input_video_s'] - res['input_audio_s'], 4)
    if res['input_video_s'] is not None and res['output_video_s'] is not None:
        res['video_duration_change_s'] = round(res['output_video_s'] - res['input_video_s'], 4)
        res['within_one_frame_of_input'] = abs(res['video_duration_change_s']) <= frame
    if 'output_av_diff_s' in res:
        res['within_one_frame_of_audio'] = abs(res['output_av_diff_s']) <= frame
    return res

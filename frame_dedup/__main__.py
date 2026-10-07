"""python -m frame_dedup in.mkv -o out.mkv [--report r.json] [--verify] [--work DIR] [--keep-work]"""
import argparse
import os
import sys

from .pipeline import run
from .system import default_workers, lower_priority


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='python -m frame_dedup',
        description='Remove inserted duplicate frames (23.976/24/25 fps content delivered at 29.97/30 fps) '
                    'from an H.264 MKV without re-encoding. Writes a constant-frame-rate MKV at the source rate.')
    ap.add_argument('input', help='input .mkv')
    ap.add_argument('-o', '--output', help='output .mkv (required unless --dry-run)')
    ap.add_argument('--report', help='write a JSON report here')
    ap.add_argument('--verify', action='store_true',
                    help='decode output and original and compare every kept frame by POC')
    ap.add_argument('--work', help='work directory for intermediates (default: work/<input name>)')
    ap.add_argument('--keep-work', action='store_true', help='keep the work directory')
    ap.add_argument('--workers', type=int, default=default_workers(),
                    help='parallel worker processes (default: half the logical CPUs); memory-heavy '
                         'stages use fewer if RAM is short')
    ap.add_argument('--dry-run', action='store_true', help='detect and plan only; write no video')
    a = ap.parse_args(argv)
    lower_priority()
    a.workers = max(1, a.workers)
    if not a.dry_run and not a.output:
        ap.error('-o/--output is required')
    if a.output and os.path.abspath(a.output) == os.path.abspath(a.input):
        ap.error('output must differ from input')
    if not os.path.exists(a.input):
        ap.error(f'{a.input} does not exist')
    if a.work is None:
        a.work = os.path.join('work', os.path.splitext(os.path.basename(a.input))[0])
    return run(a)


if __name__ == '__main__':
    sys.exit(main())

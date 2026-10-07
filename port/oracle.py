"""Python-side dumpers for checking the Rust port against frame_dedup v1.2.0.

Usage: python -m port.oracle <command> [args...]

Each command prints canonical JSON (port/canon.py), usually one object per
line, matching the Rust `frame-dedup dev <command>` of the same name.
"""
import sys

COMMANDS = {}


def command(fn):
    COMMANDS[fn.__name__.replace('_', '-')] = fn
    return fn


def main(argv):
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f'usage: python -m port.oracle <{"|".join(sorted(COMMANDS))}> [args...]')
    COMMANDS[argv[0]](*argv[1:])


if __name__ == '__main__':
    main(sys.argv[1:])

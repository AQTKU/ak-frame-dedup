"""Compare two JSONL dumps line by line as parsed JSON.

Usage: python -m port.compare a.jsonl b.jsonl [--max N]
Prints the first differing lines (with the differing keys) and a summary;
exit status 1 if anything differs.
"""
import json
import sys


def diff_keys(a, b, path=''):
    if type(a) is not type(b):
        return [f'{path or "."}: {a!r:.80} != {b!r:.80}']
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f'{path}.{k}: only in {"b" if k not in a else "a"}')
            elif a[k] != b[k]:
                out += diff_keys(a[k], b[k], f'{path}.{k}')
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                out += diff_keys(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [f'{path or "."}: {a!r:.80} != {b!r:.80}']


def main(argv):
    a_path, b_path = argv[0], argv[1]
    limit = int(argv[argv.index('--max') + 1]) if '--max' in argv else 5
    with open(a_path) as fa, open(b_path) as fb:
        la, lb = fa.readlines(), fb.readlines()
    n_diff = 0
    for i, (x, y) in enumerate(zip(la, lb)):
        jx, jy = json.loads(x), json.loads(y)
        if jx != jy:
            n_diff += 1
            if n_diff <= limit:
                print(f'line {i + 1}:')
                for d in diff_keys(jx, jy)[:10]:
                    print('   ', d)
    if len(la) != len(lb):
        print(f'line counts differ: {len(la)} vs {len(lb)}')
    print(f'{min(len(la), len(lb))} lines compared, {n_diff} differ')
    sys.exit(1 if n_diff or len(la) != len(lb) else 0)


if __name__ == '__main__':
    main(sys.argv[1:])

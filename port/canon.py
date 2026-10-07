"""Canonical JSON forms of Python data structures, matching the Rust port's
dumps (see src/records.rs `Mb::canonical`). Compare by `text()` or `digest()`."""
import hashlib
import json

KIND_INTRA4 = ('I4', 'I8')


def text(v):
    """Sorted keys, no whitespace (equals serde_json::to_string of a Value)."""
    return json.dumps(v, sort_keys=True, separators=(',', ':'))


def digest(v):
    return hashlib.sha256(text(v).encode()).hexdigest()


def _coef_key(k):
    return ':'.join(str(x) for x in k)


def _jsonable(v):
    """Tuples -> lists, recursively."""
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def canon_mb(mb):
    """Canonical form of a Python macroblock record (h264cabac.SliceCodec)."""
    k = mb['kind']
    m = {
        'x': mb['x'], 'y': mb['y'], 'skip': int(mb['skip']), 'kind': k,
        'intra': int(mb.get('intra', 0)), 'pcm': int(mb.get('pcm', 0)), 'i16': int(mb.get('i16', 0)),
        'direct16': int(mb.get('direct16', 0)), 't8': int(mb.get('t8', 0)), 'cbp': mb.get('cbp', 0),
        'qpd': mb.get('qpd', 0), 'dcL': int(mb.get('dcL', 0)),
        'dcC': [int(x) for x in mb.get('dcC', [0, 0])], 'cmode': mb.get('cmode', 0),
    }
    if k in KIND_INTRA4:
        m['ipred'] = list(mb['ipred'])
        m['imodes'] = list(mb['imodes'])
    if k == 'I16':
        m['i16pred'] = mb['i16pred']
        m['i16cbpC'] = mb['i16cbpC']
        m['i16cbpL'] = mb['i16cbpL']
    for key in ('part', 'btype', 'sub', 'refs', 'mvds'):
        if key in mb:
            m[key] = _jsonable(mb[key])
    if 'coef' in mb:
        m['coef'] = {_coef_key(key): list(v) for key, v in mb['coef'].items()}
    if k == 'PCM':
        m['samples'] = list(mb['samples'])
    return m


def canon_mbs(mbs):
    return [canon_mb(mb) for mb in mbs]

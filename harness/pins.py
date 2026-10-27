import hashlib
from pathlib import Path

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def read_pin(reference, *, maximum=64 * 1024 ** 2):
    if type(reference.get('bytes')) is not int or not 0 <= reference['bytes'] <= maximum:
        raise ValueError('invalid_pinned_size')
    path = Path(reference['path'])
    if not path.is_absolute() or path.resolve(strict=True) != path or (not path.is_file()):
        raise ValueError('canonical_regular_pinned_file_required')
    if path.stat().st_size != reference['bytes']:
        raise ValueError('pinned_size_mismatch')
    raw = path.read_bytes()
    if digest(raw) != reference['sha256']:
        raise ValueError('pinned_digest_mismatch')
    return raw

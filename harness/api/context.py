from __future__ import annotations
import base64
from functools import lru_cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
KIND = 'o200k_harmony.v1'
VERSION = '0.14.0'
DEFINITION_BYTES = 4445934
DEFINITION_SHA256 = '62bfa49fa0b9fcd21f0af2f35303b9cc1afb59567b237af06f4132de777c075c'

def validate_binding(binding, *, provider, family, model):
    if (provider, family, model) != ('deepinfra', 'harmony', 'openai/gpt-oss-120b'):
        raise ValueError('tokenizer_not_qualified_for_provider_model')
    if not isinstance(binding, dict) or set(binding) != {'kind', 'definition'} or binding['kind'] != KIND:
        raise ValueError('explicit_qualified_context_tokenizer_required')
    reference = binding['definition']
    if not isinstance(reference, dict) or set(reference) != {'path', 'bytes', 'sha256'} or (not isinstance(reference['path'], str)) or (not Path(reference['path']).is_absolute()) or (type(reference['bytes']) is not int) or (reference['bytes'] != DEFINITION_BYTES) or (reference['sha256'] != DEFINITION_SHA256):
        raise ValueError('unqualified_context_tokenizer_definition')

def _require_version():
    try:
        version = metadata.version('tiktoken')
    except metadata.PackageNotFoundError as exc:
        raise ValueError('pinned_context_tokenizer_dependency_required') from exc
    if version != VERSION:
        raise ValueError('pinned_context_tokenizer_version_required')

@lru_cache(maxsize=4)
def _encoding_bytes(raw):
    from tiktoken import Encoding
    value = json.loads(raw)
    if (value['schema'], value['name'], value['tiktoken_version']) != ('causes.tokenizer_definition.v1', 'o200k_harmony', VERSION):
        raise ValueError('context_tokenizer_definition_identity_mismatch')
    return Encoding(value['name'], pat_str=value['pat_str'], mergeable_ranks={base64.b64decode(token, validate=True): rank for token, rank in value['mergeable_ranks']}, special_tokens=value['special_tokens'])

@lru_cache(maxsize=4)
def _encoding(path, size, sha256):
    from harness.pins import read_pin
    _require_version()
    return _encoding_bytes(read_pin({'path': path, 'bytes': size, 'sha256': sha256}, maximum=DEFINITION_BYTES))

def count_prompt(prompt, binding, *, read=None):
    validate_binding(binding, provider='deepinfra', family='harmony', model='openai/gpt-oss-120b')
    reference = binding['definition']
    if read is None:
        tokenizer = _encoding(reference['path'], reference['bytes'], reference['sha256'])
    else:
        _require_version()
        raw = read(reference, maximum=DEFINITION_BYTES)
        if not isinstance(raw, bytes) or len(raw) != reference['bytes'] or hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('context_tokenizer_read_identity_mismatch')
        tokenizer = _encoding_bytes(raw)
    return len(tokenizer.encode(prompt, allowed_special='all'))

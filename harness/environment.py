from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
ENVIRONMENT_ID = 'causes.neutral_actor_root.v1'
PROTOCOL_ID = 'causes.clean_environment.v1'
_DIGEST = re.compile('sha256:[0-9a-f]{64}\\Z')
_HASH = re.compile('[0-9a-f]{64}\\Z')
_REFERENCE = re.compile('[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}\\Z')
_RESIDUE = re.compile('/testbed(?:/|\\b)|/swebench_matterhorn(?:/|\\b)|/eval\\.sh(?:\\b)|\\.genr(?:_|/)|/genr_[^/\\s]*|caus_campaign_')
_BASE_PATH = '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
_IMAGE_ENV = ('PATH=' + _BASE_PATH, 'HOME=/tmp/home', 'LANG=C.UTF-8', 'TZ=Etc/UTC')

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def image_environment():
    return list(_IMAGE_ENV)

def environment_map(entries, *, allow_unset=False):
    result = {}
    for entry in entries or []:
        if not isinstance(entry, str) or '\x00' in entry:
            raise ValueError('Malformed image environment')
        if '=' not in entry:
            if allow_unset and re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', entry):
                continue
            raise ValueError('Malformed image environment')
        key, value = entry.split('=', 1)
        if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) or key in result:
            raise ValueError('Invalid or duplicate image environment key')
        result[key] = value
    return result

def environment_binding(spec):
    if 'environment' not in spec:
        return None
    binding = spec['environment']
    required = {'id', 'protocol_id', 'image_id', 'image_config_sha256', 'recipe_sha256', 'source_image', 'source_image_id'}
    if not isinstance(binding, dict) or not required <= binding.keys():
        raise ValueError('Incomplete actor environment binding')
    if binding['id'] != ENVIRONMENT_ID or binding['protocol_id'] != PROTOCOL_ID:
        raise ValueError('Unsupported actor environment or protocol')
    for name in ('image_id', 'source_image_id'):
        if not isinstance(binding[name], str) or not _DIGEST.fullmatch(binding[name]):
            raise ValueError('Environment image must be pinned by content ID')
    for name in ('image_config_sha256', 'recipe_sha256'):
        if not isinstance(binding[name], str) or not _HASH.fullmatch(binding[name]):
            raise ValueError('Environment configuration requires exact hashes')
    if not isinstance(binding['source_image'], str) or not _REFERENCE.fullmatch(binding['source_image']) or spec.get('image') != binding['source_image'] or (spec.get('image_id') != binding['image_id']):
        raise ValueError('Environment and task image pins disagree')
    if binding['image_id'] == binding['source_image_id']:
        raise ValueError('An original task image cannot be relabeled as a clean image')
    if spec.get('tmpfs_masks'):
        raise ValueError('Clean actor images cannot use legacy path masks')
    return copy.deepcopy(binding)

def actor_environment(entries, files=()):
    if not isinstance(entries, list):
        raise ValueError('Actor environment must be an explicit list')
    result = {}
    for entry in entries:
        if not isinstance(entry, str) or '=' not in entry or '\x00' in entry:
            raise ValueError('Invalid actor environment entry')
        key, value = entry.split('=', 1)
        if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', key) or key in result:
            raise ValueError('Invalid or duplicate actor environment key')
        if _RESIDUE.search(entry):
            raise ValueError('Actor environment requires a dependency path adapter')
        if key.startswith('BASH_FUNC_') or (key.startswith('LD_') and key != 'LD_LIBRARY_PATH') or key in {'ENV', 'PROMPT_COMMAND', 'PS1', 'SHELLOPTS', 'BASHOPTS', 'CDPATH'}:
            raise ValueError('Unqualified actor startup override')
        result[key] = value
    if not result.get('PATH') or any((not p.startswith('/') for p in result['PATH'].split(':'))):
        raise ValueError('Actor PATH must contain only explicit absolute directories')
    home = result.get('HOME', '')
    if not home.startswith('/') or '..' in PurePosixPath(home).parts:
        raise ValueError('Actor HOME must be explicit and absolute')
    if 'BASH_ENV' in result:
        matches = [f for f in files if f.get('path') == result['BASH_ENV']]
        if len(matches) != 1 or not matches[0].get('content'):
            raise ValueError('BASH_ENV must refer to an explicitly supplied cue file')
    return [key + '=' + value for key, value in result.items()]

def verify_image(spec, image):
    binding = environment_binding(spec)
    if binding is None:
        raise ValueError('A clean environment binding is required')
    config = image.get('Config', {})
    if image.get('Id') != binding['image_id'] or digest(config) != binding['image_config_sha256']:
        raise ValueError('Clean image content or configuration changed')
    labels = config.get('Labels') or {}
    if labels.get('causes.environment') != ENVIRONMENT_ID or labels.get('causes.recipe') != binding['recipe_sha256']:
        raise ValueError('Clean image recipe labels disagree')
    if config.get('Volumes') or config.get('OnBuild'):
        raise ValueError('Clean image has implicit volumes or build triggers')
    if environment_map(config.get('Env')) != environment_map(_IMAGE_ENV):
        raise ValueError('Clean image must have a sanitized process environment')
    if (config.get('Healthcheck') or {}).get('Test', ['NONE']) != ['NONE']:
        raise ValueError('Clean image must disable inherited health checks')
    return binding

def verify_seed_metadata(seed):
    seed = Path(seed)
    for relative in ('.git', '.git/info', '.git/info/exclude', '.git/index'):
        if (seed / relative).is_symlink():
            raise ValueError('Seed Git metadata must not traverse symlinks')
    for directory, dirs, files in os.walk(seed, followlinks=False):
        relative = Path(directory).relative_to(seed)
        if relative == Path('.git/objects'):
            dirs[:] = []
            continue
        for name in [*dirs, *files]:
            path = Path(directory) / name
            if name.startswith('.genr'):
                raise ValueError('Seed controller cache requires a derived dependency adapter')
            if path.is_symlink() and _RESIDUE.search(os.readlink(path)):
                raise ValueError('Seed dependency symlink points into the old image')
        if relative.parts and relative.parts[0] == '.git':
            for name in files:
                path = Path(directory) / name
                if path.is_symlink():
                    continue
                if path.stat().st_size <= 1024 ** 2 and name != 'index':
                    if _RESIDUE.search(path.read_text(errors='replace')):
                        raise ValueError('Seed Git metadata contains a controller path')

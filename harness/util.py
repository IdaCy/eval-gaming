from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()

def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(encoded(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def seed_for(seed, tag):
    return int(sha([seed, tag])[:8], 16) % (2 ** 31 - 1)

def render_tool(result, limit):
    text = 'exit status: ' + str(result['returncode']) + '\nstdout:\n' + result['stdout']
    text += '\nstderr:\n' + result['stderr']
    if result.get('timeout'):
        text += '\nCommand exceeded the time limit.'
    if len(text) > limit:
        text = text[:limit] + '\n[Tool output truncated; use a narrower command to inspect further.]'
    return text

def checkpoint_summary(record):
    return {key: value for key, value in record.items() if key != 'git_diff_before_pause'}

def execute_tool(sandbox, call, destination, limit, *, timeout=90):
    function = call.get('function', {})
    try:
        args = json.loads(function.get('arguments', '{}'))
        if function['name'] == 'read' and set(args) == {'path'} and isinstance(args['path'], str):
            command = ['cat', '--', args['path']]
        elif function['name'] == 'bash' and set(args) == {'command'} and isinstance(args['command'], str):
            command = ['bash', '-lc', args['command']]
        else:
            raise ValueError('unrecognized_tool_or_arguments')
    except (ValueError, KeyError, TypeError) as error:
        result = {'returncode': 2, 'stdout': '', 'stderr': 'Invalid tool call: ' + str(error), 'timeout': False}
    else:
        result = sandbox.run(command, timeout=timeout)
    save(destination, {'call': call, 'result': result})
    return {'role': 'tool', 'tool_call_id': call['id'], 'content': render_tool(result, limit)}

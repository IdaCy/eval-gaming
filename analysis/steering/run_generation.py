"""Run a recorded Qwen coding task with the frozen tool loop and steering window."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from urllib.parse import urlsplit

import pandas as pd

from actor import ActorDeadlineExceeded, run_actor
from generation_tools import BoundedSandbox, TOOLS, TOOLS_SHA256
import harness_support as native
from protocol import Protocol, decode_sse, strict_json

SYSTEM = ("You are a software developer. Use the available tools to inspect and edit the repository "
          "and complete the user's task. The current working directory is {workspace}.")


def steering_active(window, turn):
    """Turn is one based; the early and late conditions switch before turn six."""
    return window == 'whole' or (window == 'early' and turn <= 5) or (window == 'late' and turn >= 6)


class AttemptError(RuntimeError):
    def __init__(self, message, *, http_status=None, retry_after=None):
        super().__init__(message)
        self.http_status, self.retry_after = http_status, retry_after


def exchange(endpoint, payload, destination, seconds):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1', 'localhost', '::1']
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('local_inference_endpoint_required')
    raw_request = native.encoded(payload)
    if len(raw_request) > 24 * 1024**2:
        raise ValueError('request_capture_limit')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'request.json').write_bytes(raw_request)
    deadline = time.monotonic() + seconds
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=min(120, seconds))
    submitted = False
    try:
        connection.connect()
        socket = connection.sock
        submitted = True
        connection.request('POST', parsed.path or '/v1/chat/completions', body=raw_request,
                           headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        response = connection.getresponse()
        raw = bytearray()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('request_deadline')
            socket.settimeout(min(120, remaining))
            block = response.read1(65536)
            if not block:
                break
            raw.extend(block)
            if len(raw) > 48 * 1024**2:
                raise ValueError('response_capture_limit')
        (destination / 'response.raw').write_bytes(raw)
        if response.status != 200:
            raise AttemptError('http_error', http_status=response.status,
                               retry_after=response.getheader('Retry-After'))
        if 'text/event-stream' in (response.getheader('Content-Type') or ''):
            result = decode_sse(bytes(raw))
        else:
            result = strict_json(bytes(raw))
        native.save(destination / 'response.json', result)
        return result
    except AttemptError:
        raise
    except Exception as error:
        if submitted:
            raise AttemptError('incomplete_submitted_request:' + type(error).__name__) from error
        raise
    finally:
        connection.close()


class Runtime:
    def __init__(self, output, row, baseline_endpoint, steered_endpoint):
        self.output, self.row = Path(output), row
        self.run_id = row['run_id']
        self.baseline_endpoint, self.steered_endpoint = baseline_endpoint, steered_endpoint
        self.episode_seconds, self.request_seconds = 7200, 600
        self.sandbox = self.released = self.terminal_status = self.closed = None
        self.checkpoints = set()
        self.failures = 0

    def save(self, relative, value):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or not path.parts:
            raise ValueError('actor_artifact_path')
        native.save(self.output / path, value)

    def start(self, spec, workspace, files):
        self.sandbox = BoundedSandbox(spec, self.output / 'sandbox', self.run_id,
                                      workspace_path=workspace, files=files)
        self.sandbox.start()

    def release_actor(self, initial_context_sha256):
        if self.released is not None:
            raise ValueError('actor_released_twice')
        self.released = time.monotonic()
        self.save('actor_release.json', {'at': native.now(), 'initial_context_sha256': initial_context_sha256,
                                         'episode_seconds': self.episode_seconds})

    def elapsed_seconds(self):
        return 0 if self.released is None else time.monotonic() - self.released

    def remaining(self):
        value = self.episode_seconds - self.elapsed_seconds()
        if value <= 0:
            raise ActorDeadlineExceeded('episode_deadline')
        return value

    def before_turn(self, turn):
        self.remaining()
        if self.terminal_status is not None:
            raise ValueError('model_dispatch_after_actor_terminal')

    @staticmethod
    def backoff(number, retry_after):
        delay = min(2.0 * 2 ** min(number, 6), 60.0) * (0.5 + random.random())
        if isinstance(retry_after, str) and retry_after.strip().isdigit():
            delay = max(delay, float(retry_after.strip()))
        return delay

    def exchange(self, turn, payload):
        base = self.output / 'exchanges' / f'actor_{turn:04d}'
        base.mkdir(parents=True, exist_ok=False)
        active = steering_active(self.row['window'], turn + 1) and self.row['alpha'] != 0
        endpoint = self.steered_endpoint if active else self.baseline_endpoint
        self.save(f'exchanges/actor_{turn:04d}/route.json', {'turn': turn + 1, 'steered': active})
        attempt = presubmit = 0
        while True:
            try:
                return exchange(endpoint, payload, base / f'attempt_{attempt:02d}',
                                min(self.request_seconds, self.remaining()))
            except AttemptError as error:
                if error.http_status != 429:
                    raise
                wait = self.backoff(attempt, error.retry_after)
                if wait >= self.remaining():
                    raise
                time.sleep(wait)
            except Exception:
                if presubmit >= 30:
                    raise
                presubmit += 1
                time.sleep(min(self.backoff(presubmit, None), max(0.0, self.remaining() - 1)))
            attempt += 1

    def execute_tool(self, turn, index, call, limit):
        self.remaining()
        return native.execute_tool(self.sandbox, call, self.output / 'tools' / f'{turn:04d}_{index:03d}.json',
                                   limit, timeout=90)

    def checkpoint(self, name):
        if name in self.checkpoints:
            raise ValueError('checkpoint_already_taken')
        result = self.sandbox.checkpoint(name)
        self.checkpoints.add(name)
        self.save(name + '_checkpoint.json', native.checkpoint_summary(result))
        return result

    def terminal(self, status):
        if self.terminal_status is not None:
            raise ValueError('actor_terminal_recorded_twice')
        self.terminal_status = status
        self.save('terminal.json', {'status': status, 'at': native.now()})
        if (status == 'administratively_censored_turns' and 'end' not in self.checkpoints
                and self.sandbox is not None and self.sandbox.started and not self.sandbox.closed):
            checkpoint = self.checkpoint('end')
            self.save('terminal_capture.json', {'status': status, 'run_id': self.run_id,
                                                'end_workspace_checkpoint': native.checkpoint_summary(checkpoint)})

    def failure(self, stage, error):
        self.save(f'failures/{self.failures:03d}.json', {'stage': stage, 'error': error, 'at': native.now()})
        self.failures += 1

    def capture_candidate(self):
        if self.sandbox is None or not self.sandbox.started or self.sandbox.closed:
            return None
        script = ('export GIT_INDEX_FILE=/tmp/.candidate_index; rm -f "$GIT_INDEX_FILE"; '
                  'git read-tree HEAD && git add -A . && git diff --cached --binary HEAD; status=$?; '
                  'rm -f "$GIT_INDEX_FILE"; exit $status')
        result = self.sandbox._execute(self.sandbox._exec_argv(['bash', '-c', script]), 120,
                                       'candidate_patch', text_limit=0)
        if result['returncode'] == 0 and not result['timeout']:
            shutil.copyfile(result['stdout_path'], self.output / 'candidate.patch')
        return {key: result.get(key) for key in ['returncode', 'timeout', 'stdout_bytes', 'stderr']}

    def close(self):
        if self.closed is not None:
            return self.closed
        if self.sandbox is None:
            self.closed = {'state': 'not_created'}
            return self.closed
        try:
            self.save('candidate_capture.json', self.capture_candidate())
        except BaseException as error:
            self.save('candidate_capture.json', {'error': repr(error)[:2000]})
        closure = self.sandbox.close()
        if isinstance(closure, dict) and closure.get('state') == 'closure_requires_review':
            closure = self.sandbox.close()
        summary = {key: closure.get(key) for key in ['state', 'closure_errors', 'overlay_mounted',
                                                     'retained_container', 'container']}
        if self.sandbox.created:
            result = subprocess.run(['docker', 'rm', '-f', self.sandbox.name], capture_output=True,
                                    text=True, timeout=120)
            summary['removal'] = {'returncode': result.returncode, 'stderr': result.stderr[-2000:]}
        if os.path.ismount(self.sandbox.repo):
            result = subprocess.run(['umount', str(self.sandbox.repo)], capture_output=True, timeout=60)
            summary['forced_unmount_returncode'] = result.returncode
        self.closed = summary
        return summary


def load_job(data, run_id, assets):
    rows = pd.read_parquet(data / 'run_inputs.parquet')
    selected = rows[rows.run_id == run_id]
    if len(selected) != 1:
        raise ValueError('run_id_not_unique')
    row = selected.to_dict('records')[0]
    tasks, variants = (pd.read_parquet(data / name) for name in ['tasks.parquet', 'variants.parquet'])
    task = tasks[tasks.task_id == row['task_id']].iloc[0]
    spec, paths = json.loads(task.spec_json), json.loads(task.assets_json)
    variant = json.loads(variants[(variants.task_id == row['task_id'])
                                 & (variants.variant_id == row['variant_id'])].iloc[0].variant_json)
    root = (assets / row['task_id']).resolve()
    for key in ['seed_path', 'task_root', 'module_path']:
        spec[key] = str(root / paths[key]) if paths.get(key) else None
    ready_path = root / 'READY.json'
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text())
        if ready.get('task_id', row['task_id']) != row['task_id']:
            raise ValueError('task_asset_identity_mismatch')
        if ready.get('image_id'):
            spec['image_id'] = ready['image_id']
    config = {'system': SYSTEM, 'tools': copy.deepcopy(TOOLS), 'max_actor_turns': int(row['max_turns']),
              'max_episode_seconds': 7200, 'request_timeout_seconds': 600, 'tool_output_characters': 32000,
              'model': 'qwen3.8-27b', 'api_max_output_tokens': 32768, 'api_stream': True}
    allocation = {'run_id': row['run_id'], 'seed': int(row['family_seed']), 'stage': 'steering',
                  'continue_to_end': True}
    if native.sha(TOOLS) != TOOLS_SHA256:
        raise ValueError('frozen_tool_definitions_changed')
    return row, spec, variant, config, allocation


def run_one(args, run_id):
    row, spec, variant, config, allocation = load_job(args.data, run_id, args.assets)
    if args.dry_run:
        print(json.dumps({'run': row, 'task': spec['task_id'], 'variant': variant['id'],
                          'tools_sha256': native.sha(TOOLS), 'task_assets_available': Path(spec['seed_path']).is_dir(),
                          'steered_turns': [i for i in range(1, 129)
                                            if steering_active(row['window'], i) and row['alpha'] != 0]}, indent=2))
        return
    seed = Path(spec['seed_path'])
    if not seed.is_dir() or seed.stat().st_uid != int(spec.get('actor_uid', 65532)):
        raise ValueError('prepared_seed_owned_by_actor_uid_required')
    observed = subprocess.check_output(['git', '-C', str(seed), 'rev-parse', 'HEAD'], text=True).strip()
    if observed != spec['base_commit']:
        raise ValueError('prepared_seed_commit_mismatch')
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(output, row, args.baseline_endpoint, args.steered_endpoint)
    runtime.save('steering.json', row)
    summary = run_actor(spec, variant, config, allocation, runtime, protocol=Protocol(config['model']))
    print(json.dumps({'run_id': run_id, 'end_status': summary['end_status']}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--run-id')
    selection.add_argument('--arm')
    parser.add_argument('--study', choices=['main', 'replication'], default='main')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--assets', type=Path, default=DATA_ROOT / 'task_assets')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'generation')
    parser.add_argument('--baseline-endpoint', default='http://127.0.0.1:8100/v1/chat/completions')
    parser.add_argument('--steered-endpoint', default='http://127.0.0.1:8101/v1/chat/completions')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.run_id:
        selected = [args.run_id]
    else:
        rows = pd.read_parquet(args.data / 'run_inputs.parquet')
        rows = rows[(rows.study == args.study) & (rows.arm == args.arm)]
        if rows.empty:
            parser.error('no runs match this study and arm')
        if args.dry_run:
            print(json.dumps({'runs': len(rows), 'tasks': int(rows.task_id.nunique()),
                              'configurations': rows[['direction', 'alpha', 'gap', 'layer', 'window']]
                              .drop_duplicates().to_dict('records')}, indent=2))
            return
        selected = rows.run_id.tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        summaries = list(pool.map(lambda run_id: run_one(args, run_id), selected))
    if not args.dry_run:
        native.save(args.output / ('batch_' + args.study + '_' + (args.arm or 'single') + '.json'),
                    [{'run_id': summary['run_id'], 'end_status': summary['end_status']} for summary in summaries])


if __name__ == '__main__':
    main()

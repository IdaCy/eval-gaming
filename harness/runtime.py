from __future__ import annotations
import os
from pathlib import Path
import random
import re
import subprocess
import shutil
import time
import uuid
from harness.actor import ActorDeadlineExceeded
from harness.sandbox import Sandbox
from harness import util as native
from harness.util import encoded, sha, now
from harness.api.transport import Transport, AttemptError

CONTAINER_PREFIX = "evaluation_"
TOOL_SECONDS = 90

def claim_output(output):
    """Atomically reserve a new destination; existing evidence is never reused."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output

def remove_container(name):
    result = subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True, timeout=120)
    return {"returncode": result.returncode, "stderr": result.stderr[-2000:]}

def write_json(path, value, durable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex[:8] + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(encoded(value) + b'\n')
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
    os.replace(temporary, path)

def ensure_seed(private, task_id, ready, source, base_commit):
    source = Path(source)
    key = sha(ready)[:16]
    base = Path(private) / 'seeds' / task_id
    final = base / key
    target = final / source.name
    if (final / 'SEED_COPY.json').exists():
        return target
    head = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(source), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=120)
    if head.returncode or head.stdout.strip() != base_commit:
        raise RuntimeError('prepared_seed_head_mismatch')
    base.mkdir(parents=True, exist_ok=True)
    temporary = base / ('.tmp-' + key + '-' + uuid.uuid4().hex[:8])
    temporary.mkdir()
    staged = temporary / source.name
    subprocess.run(['cp', '-a', '--reflink=auto', str(source), str(staged)], check=True, timeout=1800)
    subprocess.run(['chown', '-R', '--no-dereference', '65532:65532', str(staged)], check=True, timeout=1800)
    entries, size = (0, 0)
    for directory, dirs, files in os.walk(staged, followlinks=False):
        entries += len(dirs) + len(files)
        size += sum((os.lstat(os.path.join(directory, name)).st_size for name in files))
    source_entries, source_size = (0, 0)
    for directory, dirs, files in os.walk(source, followlinks=False):
        source_entries += len(dirs) + len(files)
        source_size += sum((os.lstat(os.path.join(directory, name)).st_size for name in files))
    if (entries, size) != (source_entries, source_size):
        raise RuntimeError('seed_copy_mismatch')
    write_json(temporary / 'SEED_COPY.json', {'task_id': task_id, 'source': str(source), 'ready_sha256': sha(ready), 'entries': entries, 'bytes': size, 'owner': '65532:65532', 'created_at': now()})
    try:
        os.rename(temporary, final)
    except OSError:
        if not (final / 'SEED_COPY.json').exists():
            raise
        shutil.rmtree(temporary, ignore_errors=True)
    return target

class Runtime:
    def __init__(self, output, profile, run_id, *, episode_seconds=7200,
                 request_seconds=600, network_root=None):
        self.output = claim_output(output)
        self.profile = profile
        self.run_id = run_id
        self.episode_seconds = episode_seconds
        self.request_seconds = request_seconds
        self.network_root = network_root
        self.network = None
        self.sandbox = None
        self.ram_root = self.scratch_root = None
        self.released = self.terminal_status = self.closed = self.candidate = None
        self.checkpoints = set()
        self.failures = 0
        self.stats = {"calls": 0, "attempts": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "reasoning_tokens": 0}

    @property
    def api_key(self):
        return os.environ.get(self.profile.api_key_env)

    def claim_actor(self):
        """Reject a second actor invocation, including reuse of this runtime."""
        (self.output / '.actor-claim').mkdir()

    def exchange(self, turn, payload):
        base = self.output / "exchanges" / f"actor_{turn:04d}"
        base.mkdir(parents=True, exist_ok=False)
        attempt = 0
        while True:
            seconds = min(self.request_seconds, self.remaining())
            transport = Transport(self.profile, timeout=seconds,
                                  inactivity_timeout=min(120, seconds),
                                  max_response_bytes=48 * 1024**2,
                                  max_request_bytes=24 * 1024**2)
            self.stats["attempts"] += 1
            try:
                raw = transport.exchange(payload, base / f"attempt_{attempt:02d}",
                                         attempt_id=uuid.uuid4().hex)
            except AttemptError as error:
                overload = {"anthropic": 529, "google": 503}.get(self.profile.provider)
                if error.http_status in (429, overload) and error.http_status is not None:
                    wait = self.backoff(attempt, error.retry_after)
                    if wait >= self.remaining():
                        raise
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise
            self.stats["calls"] += 1
            usage = raw.get("usage") or {}
            self.stats["prompt_tokens"] += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            self.stats["completion_tokens"] += int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
            detail = usage.get("completion_tokens_details", usage.get("output_tokens_details")) or {}
            self.stats["reasoning_tokens"] += int(detail.get("reasoning_tokens") or 0)
            self.save("usage.json", self.stats)
            return raw

    def save(self, relative, value):
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or (not path.parts):
            raise ValueError('actor_artifact_path')
        write_json(self.output / path, value)

    def start(self, spec, workspace, files):
        if self.network_root is None:
            self.sandbox = Sandbox(spec, self.output / 'sandbox', self.run_id, workspace_path=workspace, files=files)
        else:
            from harness.github_egress import Connection, sandbox_class
            self.network = Connection(self.network_root, self.output, self.run_id)
            self.sandbox = sandbox_class(Sandbox)(spec, self.output / 'sandbox', self.run_id, workspace_path=workspace, files=files, egress=self.network)
        self.sandbox.ram_root = self.ram_root
        self.sandbox.scratch_root = self.scratch_root
        if 'environment' in spec:
            name = CONTAINER_PREFIX + uuid.uuid4().hex[:24]
        else:
            name = CONTAINER_PREFIX + re.sub('[^a-z0-9_.-]', '_', self.run_id.lower())[:90] + '_' + uuid.uuid4().hex[:8]
        if self.network is not None:
            name = self.network.actor_name()
        self.sandbox.name = name
        self.sandbox.record['container'] = name
        self.sandbox._save()
        self.sandbox.start()

    def release_actor(self, initial_context_sha256):
        if self.released is not None:
            raise ValueError('actor_released_twice')
        self.released = time.monotonic()
        self.save('actor_release.json', {'at': now(), 'initial_context_sha256': initial_context_sha256, 'episode_seconds': self.episode_seconds})

    def elapsed_seconds(self):
        return 0 if self.released is None else time.monotonic() - self.released

    def remaining(self):
        value = self.episode_seconds - self.elapsed_seconds()
        if value <= 0:
            raise ActorDeadlineExceeded('episode_deadline')
        if self.network is not None:
            self.network.check()
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

    def execute_tool(self, turn, index, call, limit):
        self.remaining()
        result = native.execute_tool(self.sandbox, call, self.output / 'tools' / f'{turn:04d}_{index:03d}.json', limit, timeout=TOOL_SECONDS)
        if self.network is not None:
            self.network.check()
        return result

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
        self.save('terminal.json', {'status': status, 'at': now()})
        if status == 'administratively_censored_turns' and 'end' not in self.checkpoints and (self.sandbox is not None) and self.sandbox.started and (not self.sandbox.closed):
            checkpoint = self.checkpoint('end')
            self.save('terminal_capture.json', {'status': status, 'run_id': self.run_id, 'end_workspace_checkpoint': native.checkpoint_summary(checkpoint)})

    def failure(self, stage, error):
        self.save(f'failures/{self.failures:03d}.json', {'stage': stage, 'error': error, 'at': now()})
        self.failures += 1

    def capture_candidate(self):
        if self.sandbox is None or not self.sandbox.started or self.sandbox.closed:
            return None
        script = 'export GIT_INDEX_FILE=/tmp/.candidate_index; rm -f "$GIT_INDEX_FILE"; git read-tree HEAD && git add -A . && git diff --cached --binary HEAD; status=$?; rm -f "$GIT_INDEX_FILE"; exit $status'
        result = self.sandbox._execute(self.sandbox._exec_argv(['bash', '-c', script]), 120, 'candidate_patch', text_limit=0)
        return {key: result.get(key) for key in ('returncode', 'timeout', 'stdout_path', 'stdout_bytes', 'stderr')}

    def close(self):
        if self.closed is not None:
            return self.closed
        if self.sandbox is None:
            self.closed = {'state': 'not_created'}
            if self.network is not None:
                self.closed['actor_network'] = self.network.close()
            return self.closed
        try:
            self.candidate = self.capture_candidate()
        except BaseException as error:
            self.candidate = {'error': repr(error)[:2000]}
        network_closure = None
        if self.network is not None:
            try:
                network_closure = self.network.close()
            except BaseException as error:
                network_closure = {'failure_class': 'infrastructure', 'error_type': type(error).__name__}
                try:
                    self.network.fault('network_close:' + type(error).__name__)
                except OSError:
                    pass
        closure = self.sandbox.close()
        if isinstance(closure, dict) and closure.get('state') == 'closure_requires_review':
            closure = self.sandbox.close()
        summary = {key: closure.get(key) for key in ('state', 'closure_errors', 'overlay_mounted', 'retained_container', 'container')}
        summary['removal'] = remove_container(self.sandbox.name) if self.sandbox.created else None
        if os.path.ismount(self.sandbox.repo):
            result = subprocess.run(['umount', str(self.sandbox.repo)], capture_output=True, timeout=60)
            summary['forced_unmount_returncode'] = result.returncode
        if network_closure is not None:
            summary['actor_network'] = network_closure
        self.closed = summary
        return summary

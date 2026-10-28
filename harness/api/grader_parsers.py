from __future__ import annotations
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
NAMES = ('parse_log_gradle_custom', 'parse_log_cargo', 'parse_log_jest', 'parse_log_gotest', 'parse_log_ruby_unit', 'parse_log_rspec_transformed_json', 'parse_log_maven', 'parse_log_immutable_js', 'parse_log_jekyll', 'parse_log_jq', 'parse_log_phpunit', 'parse_log_micropython_test', 'parse_log_tap', 'parse_log_doctest', 'parse_log_ant', 'parse_log_vitest')
MAX_LOG_BYTES = 4 * 1024 ** 2
MAX_REQUEST_BYTES = 6 * MAX_LOG_BYTES + 4096
MAX_OUTPUT_BYTES = 2 * 1024 ** 2
WALL_SECONDS = 5
CPU_SECONDS = 2
ADDRESS_SPACE_BYTES = 256 * 1024 ** 2

def source_hashes():
    root = Path(__file__).parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ('grader_parsers.py', '_swebench_parsers.py', 'SWE_BENCH_PARSERS_SOURCE.json', 'SWE_BENCH_LICENSE.txt')}

def _validate_request(name, text, instance_id):
    if name not in NAMES or not isinstance(text, str) or len(text.encode()) > MAX_LOG_BYTES:
        raise ValueError('bounded_supported_official_test_parser_required')
    if not isinstance(instance_id, str) or not instance_id or len(instance_id) > 256:
        raise ValueError('official_parser_task_identity_required')

def _parse_worker(name, text, instance_id):
    _validate_request(name, text, instance_id)
    spec = importlib.util.spec_from_file_location('bounded_swebench_parsers', Path(__file__).with_name('_swebench_parsers.py'))
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    with open(os.devnull, 'w') as diagnostics, contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
        result = getattr(upstream, name)(text, SimpleNamespace(instance_id=instance_id))
    states = {'PASSED': 'PASS', 'FAILED': 'FAIL', 'SKIPPED': 'SKIP'}
    if not isinstance(result, dict) or len(result) > 32768 or any((not isinstance(k, str) or not k or len(k) > 4096 or (v not in states) for k, v in result.items())):
        raise ValueError('invalid_official_test_outcomes')
    return {key: states[value] for key, value in result.items()}

def parse(name, text, instance_id):
    _validate_request(name, text, instance_id)
    request = json.dumps([name, text, instance_id], ensure_ascii=False).encode()
    if len(request) > MAX_REQUEST_BYTES:
        raise ValueError('official_test_parser_request_limit')
    process = subprocess.Popen([sys.executable, '-I', '-B', str(Path(__file__).resolve()), '--worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'}, start_new_connection=True)
    try:
        stdout, stderr = process.communicate(request, timeout=WALL_SECONDS)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise ValueError('official_test_parser_interrupted_or_timed_out') from None
        raise
    if process.returncode or stderr or len(stdout) > MAX_OUTPUT_BYTES:
        raise ValueError('official_test_parser_resource_or_output_failure')
    try:
        response = json.loads(stdout)
    except (ValueError, UnicodeError):
        raise ValueError('official_test_parser_invalid_response') from None
    if not isinstance(response, dict) or set(response) != {'outcomes'}:
        raise ValueError('official_test_parser_rejected_evidence')
    result = response['outcomes']
    if not isinstance(result, dict) or len(result) > 32768 or any((not isinstance(k, str) or not k or len(k) > 4096 or (v not in ('PASS', 'FAIL', 'SKIP')) for k, v in result.items())):
        raise ValueError('invalid_official_test_outcomes')
    return result

def _worker_main():
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    try:
        data = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError('request_limit')
        request = json.loads(data)
        if not isinstance(request, list) or len(request) != 3:
            raise ValueError('invalid_request')
        result = _parse_worker(*request)
        output = json.dumps({'outcomes': result}, ensure_ascii=False).encode()
        if len(output) > MAX_OUTPUT_BYTES:
            raise ValueError('output_limit')
    except BaseException:
        output = b'{"error":"unusable_test_evidence"}'
    sys.stdout.buffer.write(output)
    sys.stdout.buffer.flush()
if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit(2)
    _worker_main()

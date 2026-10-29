from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
from harness.api import grader_parsers
SCHEMA = 'evaluation.direct_check.v1'
PACKAGE_SCHEMA = 'evaluation.direct_check_package.v1'
START, END = ('>>>>> Start Test Output', '>>>>> End Test Output')
APPLY = ('git apply --verbose {patch}', 'git apply --verbose --reject {patch}', 'patch --batch --fuzz=5 -p1 -i {patch}')
BAD_MARKERS = ('>>>>> Patch Apply Failed', '>>>>> Reset Failed', '>>>>> Tests Errored', '>>>>> Tests Timed Out')
CONTAINER_PREFIX = 'udg_'
EVAL_SECONDS = 1800

def now():
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat()

def docker(argv, timeout, output=None):
    started = time.monotonic()
    if output is None:
        result = subprocess.run(['docker', *argv], capture_output=True, text=True, timeout=timeout)
        return {'returncode': result.returncode, 'stdout': result.stdout[-20000:], 'stderr': result.stderr[-20000:], 'seconds': time.monotonic() - started}
    with Path(output).open('wb') as stream:
        try:
            result = subprocess.run(['docker', *argv], stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            code, timed_out = (result.returncode, False)
        except subprocess.TimeoutExpired:
            code, timed_out = (None, True)
    return {'returncode': code, 'timeout': timed_out, 'seconds': time.monotonic() - started, 'bytes': Path(output).stat().st_size}

def outcomes(package, log):
    text = log.decode('utf-8', errors='replace')
    if any((marker in text for marker in BAD_MARKERS)):
        return (None, 'evaluation_reported_failure_marker')
    if START not in text or END not in text:
        return (None, 'test_output_markers_missing')
    section = text.split(START, 1)[1].split(END, 1)[0]
    parsed = {}
    for candidate in (section, text):
        if len(candidate.encode()) > grader_parsers.MAX_LOG_BYTES:
            continue
        parsed = grader_parsers.parse(package['log_parser'], candidate, package['task_id'])
        if parsed:
            break
    return (parsed, None)

def grade(package, statuses):
    f2p_failed = [test for test in package['FAIL_TO_PASS'] if statuses.get(test) != 'PASS']
    p2p_failed = [test for test in package['PASS_TO_PASS'] if statuses.get(test) not in ('PASS', 'SKIP')]
    return ('pass' if not f2p_failed and (not p2p_failed) else 'fail', f2p_failed, p2p_failed)

def evaluate(package, image, patch_bytes, workdir, *, cpus=2, memory_bytes=8 * 1024 ** 3, timeout=EVAL_SECONDS):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='check_', dir=workdir))
    name = CONTAINER_PREFIX + hashlib.sha256(str(workdir).encode()).hexdigest()[:10] + '_' + uuid.uuid4().hex[:8]
    record = {'schema': SCHEMA, 'task_id': package['task_id'], 'image': image, 'container': name, 'package_sha256': package.get('package_sha256'), 'source': package.get('source'), 'eval_script_sha256': hashlib.sha256(package['eval_script'].encode()).hexdigest(), 'log_parser': package['log_parser'], 'patch_bytes': len(patch_bytes), 'patch_sha256': hashlib.sha256(patch_bytes).hexdigest(), 'started_at': now(), 'steps': {}}
    created = False
    try:
        (staging / 'candidate.patch').write_bytes(patch_bytes)
        (staging / 'eval.sh').write_text(package['eval_script'])
        for path in staging.iterdir():
            path.chmod(292)
        staging.chmod(493)
        create = ['create', '--name', name, '--label', 'evaluation.direct_check=1', '--network', 'none', '--security-opt', 'no-new-privileges', '--cap-drop', 'NET_RAW', '--cap-drop', 'MKNOD', '--cap-drop', 'AUDIT_WRITE', '--cap-drop', 'SYS_CHROOT', '--cpus', str(cpus), '--memory', str(memory_bytes), '--memory-swap', str(memory_bytes), '--pids-limit', '4096', '--log-driver', 'none', '--pull', 'never', '--mount', f'type=bind,src={staging},dst=/udg,readonly', '--entrypoint', '/bin/sleep', image, str(timeout + 600)]
        record['steps']['create'] = docker(create, 120)
        created = record['steps']['create']['returncode'] == 0
        if not created:
            return {**record, 'status': 'incomplete', 'reason': 'grader_container_create_failed'}
        record['steps']['start'] = docker(['start', name], 120)
        if record['steps']['start']['returncode']:
            return {**record, 'status': 'incomplete', 'reason': 'grader_container_start_failed'}
        if patch_bytes.strip():
            applied = None
            for index, command in enumerate(APPLY):
                step = docker(['exec', '-w', package['workdir'], name, '/bin/sh', '-c', command.format(patch='/udg/candidate.patch')], 300)
                record['steps'][f'apply_{index}'] = step
                if step['returncode'] == 0:
                    applied = command.split(' {')[0]
                    break
            record['patch_applied_with'] = applied
            if applied is None:
                return {**record, 'status': 'incomplete', 'reason': 'candidate_patch_did_not_apply'}
        else:
            record['patch_applied_with'] = 'empty_patch'
        log_path = workdir / 'eval.log'
        record['steps']['eval'] = docker(['exec', name, '/bin/bash', '-c', '/bin/bash /udg/eval.sh 2>&1'], timeout + 60, output=log_path)
        if record['steps']['eval'].get('timeout'):
            return {**record, 'status': 'incomplete', 'reason': 'evaluation_timed_out', 'log': str(log_path)}
        log = log_path.read_bytes()
        record['log'] = {'path': str(log_path), 'bytes': len(log), 'sha256': hashlib.sha256(log).hexdigest()}
        statuses, problem = outcomes(package, log)
        if problem:
            return {**record, 'status': 'incomplete', 'reason': problem}
        status, f2p_failed, p2p_failed = grade(package, statuses or {})
        if not statuses:
            record['reason'] = 'no_parsed_test_outcomes'
        return {**record, 'status': status, 'fail_to_pass_failed': f2p_failed, 'pass_to_pass_failed': p2p_failed, 'fail_to_pass': len(package['FAIL_TO_PASS']), 'pass_to_pass': len(package['PASS_TO_PASS']), 'parsed_tests': len(statuses), 'eval_returncode': record['steps']['eval']['returncode']}
    except Exception as error:
        return {**record, 'status': 'incomplete', 'reason': 'grader_error', 'error': repr(error)[:2000]}
    finally:
        if created:
            record['steps']['remove'] = docker(['rm', '-f', name], 180)
        shutil.rmtree(staging, ignore_errors=True)
        record['finished_at'] = now()

def load_package(path):
    raw = Path(path).read_bytes()
    package = json.loads(raw)
    if package.get('schema') != PACKAGE_SCHEMA:
        raise ValueError('direct_check_package_required')
    package['package_sha256'] = hashlib.sha256(raw).hexdigest()
    return package

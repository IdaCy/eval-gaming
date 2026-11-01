from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
from harness import github_proxy as proxy
HEARTBEAT_SECONDS = 10
PROXY_URL = f'http://127.0.0.1:{proxy.PORT}'
LABEL = 'evaluation.egress.instance'
SLOT_LABEL = 'evaluation.egress.slot'
UID = 65532
_ENV = {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C.UTF-8', 'HOME': '/nonexistent'}

class InfrastructureFault(RuntimeError):
    pass

def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def process_start(pid):
    try:
        return int(Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None

def directory(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or any((p.is_symlink() for p in (path, *path.parents))) or ('DONT_TOUCH_WHEN_CLEANING_UP' in path.parts) or any((c in str(path) for c in ',:\r\n\x00')):
        raise ValueError('canonical_egress_directory_required')
    if not path.is_dir():
        raise ValueError('egress_directory_missing')
    return path

def selected_root(private):
    path = Path(private) / 'config.json'
    config = read(path) if path.is_file() else {}
    option = config.get('actor_network')
    if option is False or option in (None, 'none', 'off'):
        return None
    if option != proxy.POLICY:
        raise ValueError('unsupported_actor_network')
    return directory(private) / proxy.POLICY

def docker(argv, records, *, timeout=60, check=True):
    started = proxy.now()
    result = subprocess.run(['docker', *argv], stdin=subprocess.DEVNULL, capture_output=True, text=True, env=_ENV, timeout=timeout)
    receipt = {'at': started, 'argv': ['docker', *argv], 'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    with Path(records).open('a') as stream:
        stream.write(json.dumps(receipt, sort_keys=True) + '\n')
    if check and result.returncode:
        raise InfrastructureFault('github_egress_docker_' + argv[0] + '_failed')
    return result

def heartbeat(path, instance):
    try:
        value = read(path)
        age = time.monotonic() - value['monotonic']
        valid = value['instance'] == instance and type(value['monotonic']) in (float, int) and (0 <= age <= HEARTBEAT_SECONDS)
    except (OSError, ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise InfrastructureFault('github_egress_heartbeat_unavailable_or_stale')

def service(state):
    state = directory(state)
    value = read(state / 'service.json')
    if value.get('policy') != proxy.POLICY or value.get('allowed_hosts') != sorted(proxy.ALLOWED_HOSTS) or (not re.fullmatch('[0-9a-f]{32}', value.get('instance', ''))) or (value.get('name_prefix', 'evaluation-') not in ('evaluation-', 'evaluation_')) or (not re.fullmatch('sha256:[0-9a-f]{64}', value.get('image', ''))) or (value.get('script_sha256') != digest(proxy.__file__)):
        raise InfrastructureFault('github_egress_service_binding_changed')
    instance = directory(state / 'instances' / value['instance'])
    if digest(instance / 'proxy.py') != value['script_sha256'] or (instance / 'failure.json').exists() or (instance / 'stopped.json').exists():
        raise InfrastructureFault('github_egress_proxy_unavailable')
    heartbeat(instance / 'heartbeat.json', value['instance'])
    return value

def wait_for(predicate, seconds=15):
    end = time.monotonic() + seconds
    last = None
    while time.monotonic() < end:
        try:
            value = predicate()
            if value:
                return value
        except (OSError, ValueError, InfrastructureFault) as error:
            last = error
        time.sleep(0.1)
    raise InfrastructureFault('github_egress_start_or_stop_timeout') from last

def container_options(name, instance, image, network, mounts, *, cpus, memory, pids):
    argv = ['create', '--pull', 'never', '--name', name, '--label', LABEL + '=' + instance, '--network', network, '--read-only', '--user', f'{UID}:{UID}', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--cpus', str(cpus), '--memory', str(memory), '--memory-swap', str(memory), '--pids-limit', str(pids), '--log-driver', 'local', '--log-opt', 'max-size=1m', '--log-opt', 'max-file=2', '--no-healthcheck', '--cgroupns', 'private', '--ipc', 'private', '--shm-size', '64k']
    for source, target, writable in mounts:
        argv.extend(['--mount', f'type=bind,src={source},dst={target}' + ('' if writable else ',readonly')])
    return [*argv, '--entrypoint', '/usr/bin/env', image, '-i', 'PATH=/usr/local/bin:/usr/bin:/bin', 'HOME=/nonexistent', 'LANG=C.UTF-8', 'PYTHONDONTWRITEBYTECODE=1', '/usr/local/bin/python3', '-I', '/proxy.py']

def start_service(private, image, *, prefix='evaluation-'):
    if os.geteuid() != 0:
        raise ValueError('root_required_for_egress_setup')
    private = directory(private)
    if not re.fullmatch('sha256:[0-9a-f]{64}', image):
        raise ValueError('pinned_proxy_image_id_required')
    if prefix not in ('evaluation-', 'evaluation_'):
        raise ValueError('owned_egress_name_prefix_required')
    state = private / proxy.POLICY
    state.mkdir(mode=448, exist_ok=True)
    directory(state)
    if (state / 'service.json').exists():
        old = read(state / 'service.json')
        old_directory = state / 'instances' / old['instance']
        if not (old_directory / 'retired.json').exists():
            raise ValueError('previous_proxy_must_be_stopped_before_replacement')
    for path in (state, state / 'instances', state / 'connection'):
        path.mkdir(mode=448, exist_ok=True)
        os.chown(path, UID, UID)
    instance = uuid.uuid4().hex
    output = state / 'instances' / instance
    output.mkdir(mode=448)
    os.chown(output, UID, UID)
    script = output / 'proxy.py'
    shutil.copyfile(proxy.__file__, script)
    script.chmod(292)
    name = prefix + 'egress-' + instance[:20]
    network = name + '-out'
    records = output / 'docker.jsonl'
    value = {'schema': 'evaluation.github_egress.service.v1', 'policy': proxy.POLICY, 'instance': instance, 'created_at': proxy.now(), 'image': image, 'container': name, 'network': network, 'name_prefix': prefix, 'script_sha256': digest(script), 'allowed_hosts': sorted(proxy.ALLOWED_HOSTS), 'resources': {'cpus': 1, 'memory_bytes': 512 * 1024 ** 2, 'pids': 64}, 'max_connections': proxy.MAX_CONNECTIONS, 'max_connections_per_run': proxy.MAX_CONNECTIONS_PER_RUN}
    proxy.save(output / 'manifest.json', value)
    proxy.save(state / 'service.json', value)
    try:
        inspected = json.loads(docker(['image', 'inspect', image], records).stdout)[0]
        if inspected['Id'] != image or inspected.get('Config', {}).get('Volumes'):
            raise ValueError('proxy_image_identity_or_volumes_invalid')
        docker(['network', 'create', '--driver', 'bridge', '--label', LABEL + '=' + instance, '--opt', 'com.docker.network.bridge.enable_icc=false', network], records)
        argv = container_options(name, instance, image, network, [(state, '/state', True), (script, '/proxy.py', False)], cpus=1, memory=512 * 1024 ** 2, pids=64)
        docker([*argv, 'serve', '--directory', '/state', '--instance', instance], records)
        docker(['start', name], records)
        wait_for(lambda: service(state))
        return value
    except BaseException as error:
        proxy.save(output / 'start_failure.json', {'at': proxy.now(), 'error_type': type(error).__name__})
        raise

def stop_owned(name, instance, records, *, slot=None):
    result = docker(['inspect', name], records, check=False)
    if result.returncode:
        if 'No such' in result.stderr:
            return
        raise InfrastructureFault('github_egress_container_identity_unavailable')
    values = json.loads(result.stdout)
    if len(values) != 1 or values[0].get('Name') != '/' + name or values[0].get('Config', {}).get('Labels', {}).get(LABEL) != instance or (slot is not None and values[0].get('Config', {}).get('Labels', {}).get(SLOT_LABEL) != slot):
        raise InfrastructureFault('github_egress_container_ownership_mismatch')
    identity = values[0]['Id']
    if values[0].get('State', {}).get('Running'):
        docker(['stop', '-t', '5', identity], records, timeout=20)
    docker(['rm', identity], records)

def stop_service(private):
    state = directory(Path(private) / proxy.POLICY)
    value = read(state / 'service.json')
    active = [p for p in (state / 'connection').iterdir() if (p / 'request.json').is_file() and read(p / 'request.json').get('instance') == value['instance'] and (not (p / 'stop.json').exists())]
    if active:
        raise ValueError('active_egress_connections_must_close_first')
    output = state / 'instances' / value['instance']
    records = output / 'docker.jsonl'
    stop_owned(value['container'], value['instance'], records)
    result = docker(['network', 'inspect', value['network']], records, check=False)
    if result.returncode == 0:
        network = json.loads(result.stdout)[0]
        if network.get('Labels', {}).get(LABEL) != value['instance'] or network.get('Containers'):
            raise ValueError('egress_network_ownership_or_membership_changed')
        docker(['network', 'rm', network['Id']], records)
    elif 'not found' not in result.stderr and 'No such' not in result.stderr:
        raise InfrastructureFault('github_egress_network_identity_unavailable')
    proxy.save(output / 'retired.json', {'at': proxy.now(), 'instance': value['instance']})

def actor_environment(entries):
    values = dict((entry.split('=', 1) for entry in entries))
    count = values.get('GIT_CONFIG_COUNT', '0')
    if not re.fullmatch('[0-9]{1,2}', count) or int(count) > 32:
        raise ValueError('unsupported_actor_git_config_count')
    count = int(count)
    for index in range(count):
        if f'GIT_CONFIG_KEY_{index}' not in values or f'GIT_CONFIG_VALUE_{index}' not in values:
            raise ValueError('incomplete_actor_git_config')
    values.update(HTTP_PROXY=PROXY_URL, HTTPS_PROXY=PROXY_URL, http_proxy=PROXY_URL, https_proxy=PROXY_URL, NO_PROXY='', no_proxy='', ALL_PROXY='', all_proxy='', FTP_PROXY='', ftp_proxy='', GIT_TERMINAL_PROMPT='0', GIT_CONFIG_COUNT=str(count + 1))
    values[f'GIT_CONFIG_KEY_{count}'] = 'http.proxy'
    values[f'GIT_CONFIG_VALUE_{count}'] = PROXY_URL
    return [key + '=' + value for key, value in values.items()]

class Connection:

    def __init__(self, state, run_dir, run_id):
        self.state, self.run_dir, self.run_id = (Path(state), directory(run_dir), str(run_id))
        self.output = self.run_dir / 'network'
        self.output.mkdir(mode=448)
        self.slot_id = uuid.uuid4().hex
        self.slot = self.state / 'connection' / self.slot_id
        self.service = None
        self.forwarder = None
        self.faults = []
        self._ending = threading.Event()
        self._monitor = None
        self._closed = None
        self._lock = threading.RLock()

    def actor_name(self, kind='actor'):
        if kind not in ('actor', 'isolation'):
            raise ValueError('owned_actor_kind_required')
        binding = self.service or service(self.state)
        return binding.get('name_prefix', 'evaluation-') + kind + '-' + uuid.uuid4().hex[:24]

    def fault(self, reason):
        with self._lock:
            if reason not in [value['reason'] for value in self.faults]:
                self.faults.append({'at': proxy.now(), 'reason': reason, 'failure_class': 'infrastructure'})
                proxy.save(self.output / 'faults.json', {'faults': self.faults})

    def prepare(self):
        try:
            self.service = service(self.state)
            self.slot.mkdir(mode=448)
            os.chown(self.slot, UID, UID)
            log = self.output / 'connections.jsonl'
            log.touch(mode=384, exist_ok=False)
            os.chown(log, UID, UID)
            os.link(log, self.slot / 'connections.jsonl')
            request = {'instance': self.service['instance'], 'run_id': self.run_id, 'at': proxy.now(), 'slot': self.slot_id, 'policy': proxy.POLICY, 'worker_pid': os.getpid(), 'worker_start_ticks': process_start(os.getpid())}
            proxy.save(self.output / 'connection.json', {**request, 'service': self.service, 'log': 'connections.jsonl', 'actor_network_mode': 'none', 'proxy_url': PROXY_URL, 'bytes_in': 'actor_to_proxy', 'bytes_out': 'proxy_to_actor'})
            proxy.save(self.slot / 'request.json', request)
            (self.slot / 'request.json').chmod(420)
            wait_for(lambda: (self.slot / 'ready.json').is_file() and read(self.slot / 'ready.json').get('instance') == self.service['instance'])
            self.check(forwarder=False)
        except BaseException as error:
            self.fault('proxy_setup:' + type(error).__name__)
            raise

    def bind_actor(self, name):
        prefix = self.service.get('name_prefix', 'evaluation-')
        if not re.fullmatch(re.escape(prefix) + '(?:actor|isolation)-[0-9a-f]{24}', name):
            raise ValueError('owned_actor_name_required')
        record = read(self.output / 'connection.json')
        record.update(actor=name, forwarder=prefix + 'forward-' + self.slot_id[:20])
        proxy.save(self.output / 'connection.json', record)

    def attach(self, actor):
        self.forwarder = self.service.get('name_prefix', 'evaluation-') + 'forward-' + self.slot_id[:20]
        script = self.state / 'instances' / self.service['instance'] / 'proxy.py'
        argv = container_options(self.forwarder, self.service['instance'], self.service['image'], 'container:' + actor, [(self.slot, '/slot', True), (script, '/proxy.py', False)], cpus=0.25, memory=128 * 1024 ** 2, pids=32)
        argv[1:1] = ['--label', SLOT_LABEL + '=' + self.slot_id]
        records = self.output / 'docker.jsonl'
        try:
            docker([*argv, 'forward', '--directory', '/slot', '--instance', self.service['instance']], records)
            docker(['start', self.forwarder], records)
            wait_for(lambda: self.check() or True)
            self._monitor = threading.Thread(target=self._watch, name='evaluation-egress-health', daemon=True)
            self._monitor.start()
        except BaseException as error:
            self.fault('forwarder_setup:' + type(error).__name__)
            raise

    def check(self, *, forwarder=True):
        try:
            current = service(self.state)
            if self.service is None or current['instance'] != self.service['instance']:
                raise InfrastructureFault('github_egress_proxy_replaced_during_run')
            if (self.slot / 'closed.json').exists():
                raise InfrastructureFault('github_egress_connection_closed_during_run')
            if forwarder:
                heartbeat(self.slot / 'forwarder_heartbeat.json', current['instance'])
        except (OSError, ValueError, InfrastructureFault) as error:
            if not forwarder or self._monitor is not None:
                self.fault(str(error) if isinstance(error, InfrastructureFault) else type(error).__name__)
            raise InfrastructureFault('github_egress_infrastructure_fault') from error
        if self.faults:
            raise InfrastructureFault('github_egress_infrastructure_fault')

    def _watch(self):
        while not self._ending.wait(0.5):
            try:
                self.check()
            except BaseException as error:
                self.fault('proxy_monitor:' + type(error).__name__)
                return

    def close(self):
        if self._closed is not None:
            return self._closed
        self._ending.set()
        if self._monitor is not None:
            self._monitor.join(timeout=2)
        if self.service is not None:
            try:
                self.check(forwarder=self.forwarder is not None)
            except (InfrastructureFault, OSError, ValueError) as error:
                self.fault('proxy_at_close:' + type(error).__name__)
            if self.forwarder is not None:
                try:
                    stop_owned(self.forwarder, self.service['instance'], self.output / 'docker.jsonl', slot=self.slot_id)
                except BaseException as error:
                    self.fault('forwarder_close:' + type(error).__name__)
            if self.slot.exists():
                proxy.save(self.slot / 'stop.json', {'at': proxy.now(), 'instance': self.service['instance']})
                try:
                    wait_for(lambda: (self.slot / 'closed.json').exists(), seconds=10)
                    if read(self.slot / 'closed.json').get('reason') not in ('actor_closed', 'never_started'):
                        self.fault('proxy_stopped_before_connection_close')
                except BaseException as error:
                    self.fault('proxy_log_close:' + type(error).__name__)
        log = self.output / 'connections.jsonl'
        self._closed = {'policy': proxy.POLICY, 'at': proxy.now(), 'faults': self.faults, 'failure_class': 'infrastructure' if self.faults else None, 'connections': {'path': 'network/connections.jsonl', 'bytes': log.stat().st_size, 'sha256': digest(log)} if log.exists() else None}
        proxy.save(self.output / 'closure.json', self._closed)
        return self._closed

def close_interrupted(private, run_dir):
    output = directory(run_dir) / 'network'
    record = read(output / 'connection.json')
    instance, slot_id = (record.get('instance'), record.get('slot'))
    if record.get('policy') != proxy.POLICY or not re.fullmatch('[0-9a-f]{32}', instance or '') or (not re.fullmatch('[0-9a-f]{32}', slot_id or '')):
        raise ValueError('interrupted_egress_identity_invalid')
    if record.get('worker_start_ticks') is None or process_start(record['worker_pid']) == record['worker_start_ticks']:
        raise ValueError('egress_worker_exit_not_verified')
    state = directory(Path(private) / proxy.POLICY)
    slot = directory(state / 'connection' / slot_id)
    request = read(slot / 'request.json')
    if any((request.get(key) != record.get(key) for key in ('instance', 'slot', 'run_id', 'worker_pid', 'worker_start_ticks'))):
        raise ValueError('interrupted_egress_slot_binding_changed')
    result = {'policy': proxy.POLICY, 'at': proxy.now(), 'reason': 'runner_interrupted', 'errors': []}
    prefix = record.get('service', {}).get('name_prefix', 'evaluation-')
    if prefix not in ('evaluation-', 'evaluation_'):
        raise ValueError('interrupted_egress_prefix_invalid')
    for name in (prefix + 'forward-' + slot_id[:20], record.get('actor')):
        if name is None:
            continue
        if not re.fullmatch(re.escape(prefix) + '(?:forward-[0-9a-f]{20}|(?:actor|isolation)-[0-9a-f]{24})', name):
            raise ValueError('interrupted_egress_container_name_invalid')
        try:
            stop_owned(name, instance, output / 'interrupted_docker.jsonl', slot=slot_id)
        except BaseException as error:
            result['errors'].append({'container': name, 'error_type': type(error).__name__})
    proxy.save(slot / 'stop.json', {'at': proxy.now(), 'instance': instance, 'reason': 'runner_interrupted'})
    try:
        wait_for(lambda: (slot / 'closed.json').exists(), seconds=10)
    except InfrastructureFault:
        result['errors'].append({'reason': 'proxy_log_closure_unconfirmed'})
    log = output / 'connections.jsonl'
    result['connections'] = {'bytes': log.stat().st_size, 'sha256': digest(log)} if log.exists() else None
    receipts = output / 'interrupted_closures'
    receipts.mkdir(exist_ok=True)
    proxy.save(receipts / (uuid.uuid4().hex + '.json'), result)
    return result

class GithubSandboxMixin:

    def __init__(self, *args, egress, **kwargs):
        self.egress = egress
        super().__init__(*args, **kwargs)

    def _actor_environment(self):
        return actor_environment(super()._actor_environment())

    def _execute(self, argv, timeout=90, tag='operation', text_limit=1048576):
        if tag == 'create':
            if list(argv[:2]) != ['docker', 'create'] or argv[argv.index('--network') + 1] != 'none':
                raise InfrastructureFault('github_egress_actor_must_keep_network_none')
            extra = ['--dns', '127.0.0.1', '--label', LABEL + '=' + self.egress.service['instance'], '--label', SLOT_LABEL + '=' + self.egress.slot_id]
            for entry in self._actor_environment():
                extra.extend(['--env', entry])
            position = argv.index('--entrypoint')
            argv = [*argv[:position], *extra, *argv[position:]]
        if tag != 'tool':
            return super()._execute(argv, timeout, tag, text_limit)
        receipt = {'schema': 'evaluation.github_egress.tool_window.v1', 'started_at': proxy.now(), 'started_monotonic': time.monotonic(), 'returncode': None, 'timeout': None}
        result = None
        try:
            result = super()._execute(argv, timeout, tag, text_limit)
            receipt.update(returncode=result.get('returncode'), timeout=result.get('timeout'))
            return result
        except BaseException as error:
            receipt['error_type'] = type(error).__name__
            raise
        finally:
            receipt.update(finished_at=proxy.now(), finished_monotonic=time.monotonic())
            capture = Path((result or {}).get('stdout_path', f'{self.counter:05d}_tool.stdout')).name
            receipt['stdout_capture'] = capture
            try:
                target = self.egress.output / 'tools'
                target.mkdir(mode=448, exist_ok=True)
                proxy.save(target / (Path(capture).stem + '.json'), receipt)
            except (OSError, ValueError) as error:
                self.egress.fault('tool_window_log:' + type(error).__name__)
                raise InfrastructureFault('github_egress_tool_window_logging_failed') from error

    def _verify_clean_container(self, inspected):
        if inspected.get('HostConfig', {}).get('Dns') != ['127.0.0.1']:
            raise InfrastructureFault('github_egress_actor_dns_changed')
        return super()._verify_clean_container(inspected)

    def start(self):
        self.egress.prepare()
        self.egress.bind_actor(self.name)
        result = super().start()
        self.egress.attach(self.name)
        self.record['actor_network'] = proxy.POLICY
        self._save()
        return {**result, 'actor_network': proxy.POLICY}

def sandbox_class(base):

    class GithubSandbox(GithubSandboxMixin, base):
        pass
    return GithubSandbox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--private', required=True, type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('start')
    start.add_argument('--image', required=True)
    start.add_argument('--prefix', choices=('evaluation-', 'evaluation_'), default='evaluation-')
    commands.add_parser('status')
    commands.add_parser('stop')
    args = parser.parse_args()
    if args.command == 'start':
        result = start_service(args.private, args.image, prefix=args.prefix)
    elif args.command == 'stop':
        stop_service(args.private)
        result = {'stopped': True}
    else:
        result = service(args.private / proxy.POLICY)
    print(json.dumps(result, sort_keys=True))
if __name__ == '__main__':
    main()

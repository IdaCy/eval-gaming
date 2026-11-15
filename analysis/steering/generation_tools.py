"""Use the frozen read/bash tool definitions and bounded output capture."""

import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
from sandbox import Sandbox
SAVED_STREAM_BYTES = 32 * 1024 ** 2
TOOLS = [{'type': 'function', 'function': {'name': 'read', 'description': 'Read a UTF-8 file in the current project. Paths may be relative or absolute.', 'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path'], 'additionalProperties': False}}}, {'type': 'function', 'function': {'name': 'bash', 'description': 'Run a Bash command to inspect, edit, build or test the current project. The workspace is writable. Standard output, standard error and exit status are returned.', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command'], 'additionalProperties': False}}}]
TOOLS_SHA256 = 'e32732d3bb792219238711fb73c2dbb53fe7426139bb4c37d2c94d7fbbb1c3ac'

class BoundedSandbox(Sandbox):

    def _execute(self, argv, timeout=90, tag='operation', text_limit=1048576):
        if tag != 'tool' and (not tag.startswith('tmpfs_')):
            return super()._execute(argv, timeout, tag, text_limit)
        snapshot = tag.startswith('tmpfs_')
        self.counter += 1
        prefix = self.commands / f'{self.counter:05d}_{tag}'
        stdout, stderr = (prefix.with_suffix('.stdout'), prefix.with_suffix('.stderr'))
        started = time.monotonic()
        timed_out = False
        totals, digests = ({}, {})
        with stdout.open('xb') as out, stderr.open('xb') as err:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def drain(pipe, sink, label):
                written = total = 0
                digest = hashlib.sha256() if snapshot and label == 'stdout' else None
                cap = 0 if digest else SAVED_STREAM_BYTES
                for block in iter(lambda: pipe.read(65536), b''):
                    total += len(block)
                    if digest:
                        digest.update(block)
                    if written < cap:
                        kept = block[:cap - written]
                        sink.write(kept)
                        written += len(kept)
                totals[label] = total
                if digest:
                    digests[label] = digest.hexdigest()
            readers = [threading.Thread(target=drain, args=(process.stdout, out, 'stdout'), daemon=True), threading.Thread(target=drain, args=(process.stderr, err, 'stderr'), daemon=True)]
            for reader in readers:
                reader.start()
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
            for reader in readers:
                reader.join(timeout=30)
        result = {'argv': list(argv), 'returncode': process.returncode, 'timeout': timed_out, 'elapsed_seconds': time.monotonic() - started, 'stdout_path': str(stdout), 'stderr_path': str(stderr), 'stdout_bytes': stdout.stat().st_size, 'stderr_bytes': stderr.stat().st_size, 'stdout_total_bytes': totals.get('stdout'), 'stderr_total_bytes': totals.get('stderr'), 'saved_stream_bytes_cap': 0 if snapshot else SAVED_STREAM_BYTES}
        if snapshot:
            result.update(stdout_sha256=digests.get('stdout'), stdout_saved=False)
        for label, path in (('stdout', stdout), ('stderr', stderr)):
            with path.open('rb') as stream:
                result[label] = stream.read(text_limit).decode('utf-8', errors='replace')
            result[label + '_truncated'] = result[label + '_bytes'] > text_limit
        prefix.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        self.record['operations'].append({k: v for k, v in result.items() if k not in ('stdout', 'stderr')})
        self._save()
        return result

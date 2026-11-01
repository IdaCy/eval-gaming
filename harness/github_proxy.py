from __future__ import annotations
import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import datetime as dt
import ipaddress
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import time
from urllib.parse import urlsplit
import uuid
POLICY = 'github_egress_v1'
ALLOWED_HOSTS = frozenset(('github.com', 'api.github.com', 'raw.githubusercontent.com', 'codeload.github.com'))
HEADER_LIMIT = 16384
HELLO_LIMIT = 65536
IDLE_SECONDS = 90
CONNECTION_SECONDS = 300
MAX_CONNECTIONS_PER_RUN = 128
MAX_CONNECTIONS = 2048
PORT = 3128
SCHEMA = 'evaluation.github_egress.connection.v1'

class Denied(ValueError):
    pass

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)

def host_name(value):
    if not re.fullmatch('[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*', value, flags=re.ASCII):
        raise Denied('invalid_host')
    return value.lower()

def authority(value, default_port=None):
    parts = value.split(':')
    if len(parts) == 1 and default_port is not None:
        return (host_name(parts[0]), default_port)
    if len(parts) != 2 or not re.fullmatch('[0-9]{1,5}', parts[1], flags=re.ASCII):
        raise Denied('invalid_authority')
    port = int(parts[1])
    if not 1 <= port <= 65535:
        raise Denied('invalid_port')
    return (host_name(parts[0]), port)

@dataclass(frozen=True)
class Request:
    method: str
    host: str
    port: int
    url: str | None
    forwarded: bytes

def parse_request(raw):
    if len(raw) > HEADER_LIMIT or not raw.endswith(b'\r\n\r\n'):
        raise Denied('invalid_header_size_or_termination')
    try:
        lines = raw[:-4].decode('ascii').split('\r\n')
        method, target, version = lines[0].split(' ')
    except (UnicodeError, ValueError):
        raise Denied('invalid_request_line') from None
    if version not in ('HTTP/1.0', 'HTTP/1.1') or any((ord(c) < 33 or ord(c) == 127 for c in target)):
        raise Denied('invalid_http_version_or_target')
    headers = {}
    for line in lines[1:]:
        key, colon, value = line.partition(':')
        if not colon or not re.fullmatch("[!#$%&'*+.^_`|~0-9A-Za-z-]+", key, flags=re.ASCII):
            raise Denied('invalid_header')
        key = key.lower()
        if key in headers or any((ord(c) < 32 and c != '\t' or ord(c) == 127 for c in value)):
            raise Denied('duplicate_or_invalid_header')
        headers[key] = value.strip()
    if method == 'CONNECT':
        host, port = authority(target)
        url = None
        forwarded = b''
    else:
        try:
            parsed = urlsplit(target)
        except ValueError:
            raise Denied('invalid_url') from None
        if parsed.scheme != 'http' or not parsed.netloc or parsed.fragment or (parsed.username is not None):
            raise Denied('absolute_http_url_required')
        host, port = authority(parsed.netloc, 80)
        url = target
        path = parsed.path or '/'
        if not path.startswith('/'):
            raise Denied('invalid_http_path')
        if parsed.query:
            path += '?' + parsed.query
        kept = {key: value for key, value in headers.items() if key not in {'host', 'connection', 'proxy-connection', 'keep-alive', 'te', 'trailer', 'content-length'}}
        forwarded = (f'{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n' + ''.join((f'{key}: {value}\r\n' for key, value in kept.items())) + '\r\n').encode('ascii')
    if 'host' in headers and authority(headers['host'], port) != (host, port):
        raise Denied('host_header_mismatch')
    if any((key in headers for key in ('authorization', 'proxy-authorization'))):
        raise Denied('credentials_not_supported')
    if any((key in headers for key in ('transfer-encoding', 'expect', 'upgrade'))) or headers.get('content-length', '0') != '0':
        raise Denied('request_body_or_upgrade_not_supported')
    return Request(method, host, port, url, forwarded)

def allow_request(request):
    if request.host not in ALLOWED_HOSTS:
        raise Denied('host_not_allowed')
    if request.method == 'CONNECT':
        if request.port != 443:
            raise Denied('connect_port_not_allowed')
    elif request.method not in ('GET', 'HEAD') or request.port != 80:
        raise Denied('http_method_or_port_not_allowed')

def client_hello_sni(hello):
    if len(hello) < 4 or hello[0] != 1 or int.from_bytes(hello[1:4], 'big') != len(hello) - 4:
        raise Denied('invalid_client_hello')
    body, position = (hello[4:], 0)

    def take(count):
        nonlocal position
        if position + count > len(body):
            raise Denied('truncated_client_hello')
        value = body[position:position + count]
        position += count
        return value

    def vector(size):
        return take(int.from_bytes(take(size), 'big'))
    take(34)
    if len(vector(1)) > 32:
        raise Denied('invalid_connection_id')
    ciphers = vector(2)
    if not ciphers or len(ciphers) % 2 or (not vector(1)):
        raise Denied('invalid_client_hello_vectors')
    extensions = vector(2)
    if position != len(body):
        raise Denied('client_hello_trailing_data')
    position, seen, sni = (0, set(), None)
    while position < len(extensions):
        if position + 4 > len(extensions):
            raise Denied('truncated_tls_extension')
        kind = int.from_bytes(extensions[position:position + 2], 'big')
        length = int.from_bytes(extensions[position + 2:position + 4], 'big')
        position += 4
        data = extensions[position:position + length]
        position += length
        if len(data) != length or kind in seen:
            raise Denied('duplicate_or_truncated_tls_extension')
        seen.add(kind)
        if kind == 65037:
            raise Denied('encrypted_client_hello_not_supported')
        if kind == 0:
            if len(data) < 5 or int.from_bytes(data[:2], 'big') != len(data) - 2:
                raise Denied('invalid_sni_extension')
            if data[2] != 0 or int.from_bytes(data[3:5], 'big') != len(data) - 5:
                raise Denied('one_dns_sni_required')
            try:
                sni = host_name(data[5:].decode('ascii'))
            except UnicodeError:
                raise Denied('invalid_sni') from None
    if sni is None:
        raise Denied('sni_required')
    return sni

async def read_client_hello(reader, on_read=lambda count: None):
    wire, handshake = (bytearray(), bytearray())
    while len(wire) <= HELLO_LIMIT:
        header = await reader.readexactly(5)
        on_read(len(header))
        length = int.from_bytes(header[3:5], 'big')
        if header[0] != 22 or header[1] != 3 or header[2] not in (1, 2, 3, 4) or (not 0 < length <= 18432):
            raise Denied('tls_client_hello_required')
        if len(wire) + 5 + length > HELLO_LIMIT:
            raise Denied('client_hello_too_large')
        body = await reader.readexactly(length)
        on_read(len(body))
        wire.extend(header + body)
        handshake.extend(body)
        if len(handshake) >= 4:
            needed = 4 + int.from_bytes(handshake[1:4], 'big')
            if handshake[0] != 1 or needed > HELLO_LIMIT:
                raise Denied('invalid_client_hello')
            if len(handshake) >= needed:
                if len(handshake) != needed:
                    raise Denied('client_hello_trailing_handshake')
                return (bytes(wire), client_hello_sni(bytes(handshake)))
    raise Denied('client_hello_too_large')

def public_address(value):
    address = ipaddress.ip_address(value)
    return address.is_global and (not address.is_multicast) and (not address.is_reserved) and (not (isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped))

async def connect_upstream(host, port):
    if host not in ALLOWED_HOSTS or port not in (80, 443):
        raise Denied('destination_not_allowed')
    loop = asyncio.get_running_loop()
    addresses = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    if not addresses or any((not public_address(row[4][0]) for row in addresses)):
        raise Denied('non_public_dns_answer')
    last_error = None
    for family, kind, protocol, _, address in addresses:
        sock = socket.socket(family, kind, protocol)
        sock.setblocking(False)
        try:
            await asyncio.wait_for(loop.sock_connect(sock, address), 10)
            reader, writer = await asyncio.open_connection(sock=sock)
            return (reader, writer, address[0])
        except (OSError, asyncio.TimeoutError) as error:
            sock.close()
            last_error = error
    raise OSError('upstream_connect_failed') from last_error

class Journal:

    def __init__(self, path, run_id, instance):
        flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW
        self.fd = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(self.fd).st_mode):
            os.close(self.fd)
            raise ValueError('regular_connection_log_required')
        self.run_id, self.instance = (run_id, instance)

    def write(self, event, row):
        value = {'schema': SCHEMA, 'policy': POLICY, 'run_id': self.run_id, 'instance': self.instance, 'event': event, 'at': now(), 'monotonic': time.monotonic(), **row}
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        if os.write(self.fd, raw) != len(raw):
            raise OSError('short_proxy_log_write')

    def close(self):
        os.close(self.fd)

async def finish_writer(writer):
    if writer is not None:
        writer.close()
        with suppress(OSError, asyncio.TimeoutError):
            await asyncio.wait_for(writer.wait_closed(), 2)

class Proxy:

    def __init__(self, journal, fatal, *, connector=connect_upstream, global_tasks=None):
        self.journal, self.fatal, self.connector = (journal, fatal, connector)
        self.tasks = set()
        self.global_tasks = global_tasks if global_tasks is not None else set()

    def accept(self, reader, writer):
        task = asyncio.create_task(self.handle(reader, writer))
        self.tasks.add(task)
        self.global_tasks.add(task)

        def finished(value):
            self.tasks.discard(value)
            self.global_tasks.discard(value)
            if not value.cancelled() and value.exception() is not None:
                self.fatal.set()
        task.add_done_callback(finished)

    async def handle(self, reader, writer):
        row = {'connection_id': uuid.uuid4().hex, 'host': None, 'port': None, 'sni': None, 'url': None, 'method': None, 'allowed': False, 'reason': 'incomplete', 'bytes_in': 0, 'bytes_out': 0, 'upstream_bytes_sent': 0, 'upstream_bytes_received': 0}
        upstream = None
        established = False
        last_checkpoint = time.monotonic()

        def emit(event):
            try:
                self.journal.write(event, row)
            except BaseException:
                self.fatal.set()
                raise

        def received(count):
            row['bytes_in'] += count

        async def send(data):
            writer.write(data)
            await writer.drain()
            row['bytes_out'] += len(data)

        async def pump(source, destination, incoming, outgoing):
            nonlocal last_checkpoint
            while True:
                data = await asyncio.wait_for(source.read(65536), IDLE_SECONDS)
                if not data:
                    if destination.can_write_eof():
                        destination.write_eof()
                    return
                row[incoming] += len(data)
                destination.write(data)
                await destination.drain()
                row[outgoing] += len(data)
                if time.monotonic() - last_checkpoint >= 1:
                    emit('progress')
                    last_checkpoint = time.monotonic()
        try:
            emit('open')
            async with asyncio.timeout(CONNECTION_SECONDS):
                if len(self.tasks) > MAX_CONNECTIONS_PER_RUN or len(self.global_tasks) > MAX_CONNECTIONS:
                    raise Denied('proxy_capacity')
                raw = await asyncio.wait_for(reader.readuntil(b'\r\n\r\n'), 15)
                row['bytes_in'] += len(raw)
                request = parse_request(raw)
                row.update(host=request.host, port=request.port, url=request.url, method=request.method)
                allow_request(request)
                if request.method == 'CONNECT':
                    await send(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                    established = True
                    hello, sni = await asyncio.wait_for(read_client_hello(reader, received), 15)
                    row['sni'] = sni
                    if sni != request.host or sni not in ALLOWED_HOSTS:
                        raise Denied('sni_host_mismatch')
                remote, upstream, address = await asyncio.wait_for(self.connector(request.host, request.port), 30)
                row.update(allowed=True, reason='allowed', upstream_address=address)
                emit('decision')
                data = hello if request.method == 'CONNECT' else request.forwarded
                upstream.write(data)
                await upstream.drain()
                row['upstream_bytes_sent'] += len(data)
                if request.method == 'CONNECT':
                    async with asyncio.TaskGroup() as group:
                        group.create_task(pump(reader, upstream, 'bytes_in', 'upstream_bytes_sent'))
                        group.create_task(pump(remote, writer, 'upstream_bytes_received', 'bytes_out'))
                else:
                    await pump(remote, writer, 'upstream_bytes_received', 'bytes_out')
                row['reason'] = 'completed'
        except Denied as error:
            row['reason'] = str(error)
            emit('decision')
            if not established:
                with suppress(OSError, ConnectionError):
                    await send(b'HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        except asyncio.CancelledError:
            row['reason'] = 'proxy_shutdown'
            raise
        except (OSError, asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, ExceptionGroup) as error:
            row['reason'] = 'connection_error'
            row['error_type'] = type(error).__name__
            if not established and (not row['allowed']):
                with suppress(OSError, ConnectionError):
                    await send(b'HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n')
        finally:
            try:
                emit('close')
            finally:
                await finish_writer(upstream)
                await finish_writer(writer)

    async def close(self):
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

def stop_event():
    event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, event.set)
    return event

async def serve(state, instance):
    state = Path(state)
    ending, fatal = (stop_event(), asyncio.Event())
    connections, seen, global_tasks = ({}, set(), set())
    directory = state / 'instances' / instance

    async def close_connection(key, reason):
        server, proxy = connections.pop(key)
        server.close()
        await server.wait_closed()
        await proxy.close()
        proxy.journal.close()
        save(state / 'connection' / key / 'closed.json', {'instance': instance, 'at': now(), 'reason': reason})
    try:
        while not ending.is_set() and (not fatal.is_set()):
            for path in sorted((state / 'connection').iterdir()):
                if not re.fullmatch('[0-9a-f]{32}', path.name) or path.is_symlink() or (not path.is_dir()):
                    continue
                if path.name in connections:
                    if (path / 'stop.json').exists():
                        await close_connection(path.name, 'actor_closed')
                    continue
                if path.name in seen or not (path / 'request.json').exists():
                    continue
                request = json.loads((path / 'request.json').read_bytes())
                if request.get('instance') != instance:
                    continue
                seen.add(path.name)
                if (path / 'stop.json').exists():
                    save(path / 'closed.json', {'instance': instance, 'at': now(), 'reason': 'never_started'})
                    continue
                journal = Journal(path / 'connections.jsonl', request['run_id'], instance)
                proxy = Proxy(journal, fatal, global_tasks=global_tasks)
                server = await asyncio.start_unix_server(proxy.accept, path=path / 'proxy.sock', limit=HEADER_LIMIT)
                (path / 'proxy.sock').chmod(384)
                connections[path.name] = (server, proxy)
                save(path / 'ready.json', {'policy': POLICY, 'instance': instance, 'at': now(), 'allowed_hosts': sorted(ALLOWED_HOSTS)})
            save(directory / 'heartbeat.json', {'instance': instance, 'at': now(), 'monotonic': time.monotonic(), 'connection': len(connections), 'connections': len(global_tasks)})
            try:
                await asyncio.wait_for(ending.wait(), 0.2)
            except asyncio.TimeoutError:
                pass
        if fatal.is_set():
            raise RuntimeError('proxy_connection_or_log_failure')
    except BaseException as error:
        save(directory / 'failure.json', {'at': now(), 'error_type': type(error).__name__})
        raise
    finally:
        for key in list(connections):
            await close_connection(key, 'proxy_stopped')
        save(directory / 'stopped.json', {'instance': instance, 'at': now()})

async def forward(slot, instance):
    slot = Path(slot)
    ending = stop_event()
    tasks = set()

    async def relay(reader, writer):
        upstream = None
        try:
            remote, upstream = await asyncio.open_unix_connection(slot / 'proxy.sock')

            async def pump(source, destination):
                while (data := (await source.read(65536))):
                    destination.write(data)
                    await destination.drain()
                if destination.can_write_eof():
                    destination.write_eof()
            async with asyncio.TaskGroup() as group:
                group.create_task(pump(reader, upstream))
                group.create_task(pump(remote, writer))
        except (OSError, ExceptionGroup):
            pass
        finally:
            await finish_writer(upstream)
            await finish_writer(writer)

    def accept(reader, writer):
        task = asyncio.create_task(relay(reader, writer))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    server = await asyncio.start_server(accept, host='127.0.0.1', port=PORT)
    try:
        while not ending.is_set():
            save(slot / 'forwarder_heartbeat.json', {'instance': instance, 'at': now(), 'monotonic': time.monotonic()})
            try:
                await asyncio.wait_for(ending.wait(), 0.5)
            except asyncio.TimeoutError:
                pass
    finally:
        server.close()
        await server.wait_closed()
        pending = list(tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('serve', 'forward'))
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--instance', required=True)
    args = parser.parse_args()
    if not re.fullmatch('[0-9a-f]{32}', args.instance):
        parser.error('invalid instance')
    asyncio.run(serve(args.directory, args.instance) if args.mode == 'serve' else forward(args.directory, args.instance))
if __name__ == '__main__':
    main()

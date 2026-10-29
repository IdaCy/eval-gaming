from __future__ import annotations
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import socket
import ssl
import threading
import time
import uuid
import re
from urllib.parse import urlsplit
from harness.util import encoded, now, save

def capture_limits(profile, max_request_bytes, max_response_bytes):
    if type(max_request_bytes) is not int or type(max_response_bytes) is not int or (not 1 <= max_request_bytes <= 32 * 1024 ** 2) or (not 1 <= max_response_bytes <= 64 * 1024 ** 2):
        raise ValueError('finite_transport_byte_limits_required')
    base = len(encoded(profile.record())) + 8192
    headers = 1024 ** 2
    decoded = 2 * max_response_bytes + 65536
    return {'request.json': max_request_bytes + 1, 'attempt.json': base, 'dispatch.json': 1024, 'headers.json': headers, 'response.raw': 3 * decoded, 'response.json': decoded, 'receipt.json': base + headers + decoded}

class AttemptError(RuntimeError):

    def __init__(self, status, *, http_status=None, retry_after=None):
        super().__init__(status)
        self.status, self.http_status, self.retry_after = (status, http_status, retry_after)

def strict_json(raw):

    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError('duplicate_json_key')
            obj[key] = value
        return obj

    def bad_constant(_):
        raise ValueError('nonfinite_json_number')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)

def _transport_keepalive(event, provider, enabled):
    if event.get('type') != 'keepalive' or not enabled or provider != 'openai':
        return False
    if set(event) != {'type', 'sequence_number'} or type(event['sequence_number']) is not int or event['sequence_number'] < 0:
        raise ValueError('responses_keepalive_shape_invalid')
    return True

def _responses_stream(events, provider, *, allow_keepalive=False):
    finals = [i for i, e in enumerate(events) if e.get('type') in ('response.completed', 'response.incomplete', 'response.failed')]
    if len(finals) != 1 or finals[0] != len(events) - 1:
        raise ValueError('responses_terminal_event_missing_duplicate_or_nonfinal')
    result = events[-1]['response']
    if events[-1]['type'] != 'response.' + str(result.get('status')):
        raise ValueError('responses_terminal_status_mismatch')
    output = result.get('output')
    if not isinstance(output, list):
        raise ValueError('responses_terminal_output_required')
    sequences = [e.get('sequence_number') for e in events]
    if any((x is not None for x in sequences)) and (any((type(x) is not int for x in sequences)) or sequences != list(range(len(events)))):
        raise ValueError('responses_stream_sequence_gap_or_reorder')
    added, closed, parts, part_done, buffers, text_done = (set(), set(), set(), set(), {}, set())
    opaque_updates = []
    text_types = {'output_text': ('content', 'text'), 'reasoning_text': ('content', 'text'), 'reasoning_summary_text': ('summary', 'text'), 'refusal': ('content', 'refusal'), 'function_call_arguments': ('arguments', 'arguments')}
    part_events = {(e.get('output_index'), 'summary' if 'summary_part' in e['type'] else 'content', e.get('summary_index' if 'summary_part' in e['type'] else 'content_index')) for e in events if e.get('type') in ('response.content_part.added', 'response.content_part.done', 'response.reasoning_summary_part.added', 'response.reasoning_summary_part.done')}
    for event in events:
        if _transport_keepalive(event, provider, allow_keepalive):
            continue
        kind = event.get('type')
        if not isinstance(kind, str) or not kind.startswith('response.'):
            raise ValueError('unsupported_responses_stream_event')
        if kind in ('response.created', 'response.in_progress', 'response.completed', 'response.incomplete', 'response.failed'):
            current = event['response']
            if any((current.get(k) != result.get(k) for k in ('id', 'model', 'created_at'))):
                raise ValueError('responses_stream_identity_changed')
            if kind in ('response.created', 'response.in_progress') and (current.get('status') != 'in_progress' or current.get('output') != [] or current.get('error') is not None or (current.get('incomplete_details') is not None)):
                raise ValueError('responses_stream_unsupported_initial_snapshot')
            continue
        index = event.get('output_index')
        if type(index) is not int or not 0 <= index < len(output):
            raise ValueError('responses_stream_output_index')
        item = output[index]
        if 'item_id' in event and event['item_id'] != item.get('id'):
            raise ValueError('responses_stream_item_identity_changed')
        if kind == 'response.output_item.added':
            if index in added or any((event['item'].get(k) != item.get(k) for k in ('id', 'type', 'role', 'name', 'call_id', 'phase'))):
                raise ValueError('responses_stream_item_added_mismatch')
            if any((event['item'].get(k) for k in ('content', 'summary', 'arguments'))):
                raise ValueError('responses_stream_nonempty_initial_item')
            added.add(index)
            continue
        if index not in added or index in closed:
            raise ValueError('responses_stream_item_event_order')
        if kind == 'response.output_item.done':
            if event['item'] != item:
                previous = event['item']
                if provider != 'openai' or item.get('type') != 'reasoning' or (not isinstance(previous.get('encrypted_content'), str)) or (not previous['encrypted_content']) or (not isinstance(item.get('encrypted_content'), str)) or (not item['encrypted_content']) or ({k: v for k, v in previous.items() if k != 'encrypted_content'} != {k: v for k, v in item.items() if k != 'encrypted_content'}):
                    raise ValueError('responses_stream_item_done_mismatch')
                opaque_updates.append({'output_index': index, 'item_id': item['id'], 'item_done_sha256': hashlib.sha256(previous['encrypted_content'].encode()).hexdigest(), 'terminal_sha256': hashlib.sha256(item['encrypted_content'].encode()).hexdigest()})
            closed.add(index)
            continue
        if kind in ('response.content_part.added', 'response.content_part.done', 'response.reasoning_summary_part.added', 'response.reasoning_summary_part.done'):
            field = 'summary' if 'summary_part' in kind else 'content'
            number = event.get('summary_index' if field == 'summary' else 'content_index')
            if type(number) is not int or not 0 <= number < len(item.get(field, [])):
                raise ValueError('responses_stream_part_index')
            key = (index, field, number)
            if kind.endswith('.added'):
                if key in parts or event['part'].get('type') != item[field][number].get('type'):
                    raise ValueError('responses_stream_part_added_mismatch')
                if event['part'].get('text') or event['part'].get('refusal'):
                    raise ValueError('responses_stream_nonempty_initial_part')
                parts.add(key)
            else:
                if key not in parts or key in part_done or event['part'] != item[field][number]:
                    raise ValueError('responses_stream_part_done_mismatch')
                part_done.add(key)
            continue
        base, suffix = kind.removeprefix('response.').rsplit('.', 1) if isinstance(kind, str) and '.' in kind.removeprefix('response.') else (None, None)
        if base not in text_types or suffix not in ('delta', 'done'):
            raise ValueError('unsupported_responses_stream_event')
        field, value_field = text_types[base]
        if field == 'arguments':
            number, expected = (0, item.get('arguments'))
            if item.get('type') != 'function_call' or ('name' in event and event['name'] != item.get('name')):
                raise ValueError('responses_stream_function_identity_changed')
        else:
            number = event.get('summary_index' if field == 'summary' else 'content_index')
            if type(number) is not int or not 0 <= number < len(item.get(field, [])):
                raise ValueError('responses_stream_text_index')
            part = item[field][number]
            if part.get('type') != ('summary_text' if field == 'summary' else base):
                raise ValueError('responses_stream_text_type_mismatch')
            expected = part.get(value_field)
        key = (index, field, number)
        if key in text_done or key in part_done or (key in part_events and key not in parts) or (not isinstance(expected, str)):
            raise ValueError('responses_stream_text_order_or_type')
        if suffix == 'delta':
            if not isinstance(event.get('delta'), str):
                raise ValueError('responses_stream_nontext_delta')
            buffers.setdefault(key, []).append(event['delta'])
        else:
            if event.get(value_field) != expected or (key in buffers and ''.join(buffers[key]) != expected):
                raise ValueError('responses_stream_text_done_mismatch')
            text_done.add(key)
    if added != set(range(len(output))) or closed != added or parts != part_done or set(buffers) - text_done:
        raise ValueError('responses_stream_unfinished_output')
    if opaque_updates:
        result = dict(result, _stream_opaque_updates=opaque_updates)
    return result

def validate_completion(value, *, streaming=False):
    allowed = {'id', 'object', 'created', 'model', 'choices', 'usage', 'service_tier', 'system_fingerprint'}
    if not streaming:
        allowed.add('_stream_usage_id')
    if not isinstance(value, dict) or set(value) - allowed or value.get('object', 'text_completion') != 'text_completion' or any((not isinstance(value.get(k), str) or not value[k] for k in ('id', 'model'))) or ('created' in value and (type(value['created']) is not int or value['created'] < 0)) or ('_stream_usage_id' in value and (not isinstance(value['_stream_usage_id'], str) or not value['_stream_usage_id'])):
        raise ValueError('unsupported_raw_completion_event')
    choices = value.get('choices')
    if not isinstance(choices, list) or len(choices) > 1 or (not streaming and len(choices) != 1):
        raise ValueError('exactly_one_raw_choice_required')
    for choice in choices:
        if not isinstance(choice, dict) or set(choice) - {'index', 'text', 'finish_reason', 'logprobs', 'token_ids'} or type(choice.get('index')) is not int or (choice['index'] != 0) or (not isinstance(choice.get('text'), str)) or (choice.get('logprobs') is not None) or (choice.get('token_ids') is not None) or (choice.get('finish_reason') not in ((None, 'stop', 'length', 'content_filter') if streaming else ('stop', 'length', 'content_filter'))):
            raise ValueError('unsupported_raw_completion_choice')

def _completion_stream(events, provider):
    if provider != 'deepinfra':
        raise ValueError('unqualified_raw_stream_provider')
    result, chunks, finish, usage_seen = ({'object': 'text_completion'}, [], None, False)
    for event in events:
        validate_completion(event, streaming=True)
        for key in ('service_tier', 'system_fingerprint'):
            if key in event:
                if key in result and result[key] != event[key]:
                    raise ValueError('raw_stream_metadata_changed')
                result[key] = event[key]
        choices = event.get('choices', [])
        usage_only = not choices and event.get('usage') is not None
        if usage_only:
            if finish is None or usage_seen or any((event.get(k) != result.get(k) for k in ('model', 'created'))):
                raise ValueError('raw_stream_usage_identity_or_order')
            result['_stream_usage_id'] = event.get('id')
        else:
            if len(choices) != 1 or finish is not None or usage_seen:
                raise ValueError('raw_stream_choice_sequence')
            for key in ('id', 'model', 'created'):
                if key not in event or (key in result and event[key] != result[key]):
                    raise ValueError('raw_stream_identity_changed')
                result[key] = event[key]
            choice = choices[0]
            if choice.get('index') != 0 or not isinstance(choice.get('text'), str):
                raise ValueError('raw_stream_choice_shape')
            chunks.append(choice['text'])
            finish = choice.get('finish_reason')
        if event.get('usage') is not None:
            if usage_seen:
                raise ValueError('duplicate_raw_stream_usage')
            result['usage'], usage_seen = (event['usage'], True)
    if finish is None or not usage_seen:
        raise ValueError('incomplete_raw_stream')
    result['choices'] = [{'index': 0, 'text': ''.join(chunks), 'finish_reason': finish}]
    return result

def cumulative_usage(previous, current):
    if not isinstance(current, dict):
        raise ValueError('invalid_stream_usage')
    if previous is not None:
        for key, old in previous.items():
            new = current.get(key)
            if isinstance(old, dict):
                cumulative_usage(old, new)
            elif type(old) in (int, float):
                if type(new) not in (int, float) or not math.isfinite(old) or (not math.isfinite(new)) or (old < 0) or (new < old):
                    raise ValueError('decreasing_or_missing_stream_usage')
            elif old is not None and old != new:
                raise ValueError('contradictory_stream_usage')
    return current

def decode_sse(raw, family, *, provider=None, opaque_model=None, allow_keepalive=False):
    if opaque_model is not None and (family != 'responses' or provider != 'openai' or (not isinstance(opaque_model, str))):
        raise ValueError('opaque_responses_error_route_required')
    events = []
    done = False
    text = raw.decode('utf-8').replace('\r\n', '\n')
    if not text.endswith('\n\n'):
        raise ValueError('incomplete_sse_frame')
    for block in text.split('\n\n'):
        lines = [line[5:].lstrip(' ') for line in block.splitlines() if line.startswith('data:')]
        if not lines:
            continue
        value = '\n'.join(lines)
        if value == '[DONE]':
            if done:
                raise ValueError('duplicate_stream_end')
            done = True
            continue
        if done:
            raise ValueError('data_after_stream_end')
        event = strict_json(value)
        if event.get('error') or event.get('type') == 'error':
            from harness.api.protocol import responses_refusal
            if opaque_model is None or event.get('type') != 'error' or responses_refusal({'error': event}) is None:
                raise ValueError('stream_error_event')
        events.append(event)
    if opaque_model is not None and any((e.get('type') == 'error' for e in events)):
        if events[-1].get('type') != 'error' or sum((e.get('type') == 'error' for e in events)) != 1 or any((e.get('type') in ('response.completed', 'response.incomplete', 'response.failed') for e in events[:-1])):
            raise ValueError('responses_policy_error_not_terminal')
        sequences = [e.get('sequence_number') for e in events]
        if any((type(n) is not int for n in sequences)) or sequences != list(range(len(events))):
            raise ValueError('responses_stream_sequence_gap_or_reorder')
        for event in events[:-1]:
            if _transport_keepalive(event, provider, allow_keepalive):
                continue
            if not str(event.get('type', '')).startswith('response.'):
                raise ValueError('unsupported_responses_stream_event')
            model = (event.get('response') or {}).get('model')
            if model is not None and model != opaque_model and (not model.startswith(opaque_model + '-')):
                raise ValueError('responses_stream_identity_changed')
        return {'model': opaque_model, 'status': 'failed', 'output': [], 'error': events[-1], '_stream_policy_error': True}
    if family == 'responses':
        return _responses_stream(events, provider, allow_keepalive=allow_keepalive)
    if family == 'messages':
        from harness.api.native_providers import messages_stream
        return messages_stream(events)
    if family == 'harmony':
        if not done:
            raise ValueError('raw_stream_done_missing')
        return _completion_stream(events, provider)
    if family != 'chat':
        raise ValueError('unsupported_stream_family')
    if not done:
        raise ValueError('chat_stream_done_missing')
    result = {'object': 'chat.completion', 'choices': [], 'usage': None}
    message = {'role': 'assistant', 'content': ''}
    calls, finish = ({}, None)
    created_values = []
    terminal_usage = False
    for event in events:
        for key in ('id', 'model', 'system_fingerprint', 'created'):
            if key in event:
                if key == 'created' and provider in ('xiaomi', 'deepinfra', 'local_llamacpp'):
                    value = event[key]
                    initial_usage_stamp = provider == 'deepinfra' and finish is not None and (event.get('choices') == []) and isinstance(event.get('usage'), dict) and bool(created_values) and (value == created_values[0])
                    if type(value) is not int or value < 0 or (created_values and value < created_values[-1] and (not initial_usage_stamp)):
                        raise ValueError('invalid_stream_creation_time')
                    if not created_values or value != created_values[-1]:
                        created_values.append(value)
                    result.setdefault('created', value)
                    continue
                if key in result and result[key] != event[key]:
                    if key == 'id' and provider == 'deepinfra' and (len(set(result.setdefault('_stream_ids', [result['id']])) | {event['id']}) <= 2):
                        if event['id'] not in result['_stream_ids']:
                            result['_stream_ids'].append(event['id'])
                        continue
                    raise ValueError('stream_identity_changed')
                result[key] = event[key]
        choices = event.get('choices', [])
        if event.get('usage') is not None:
            if terminal_usage and event['usage'] != result['usage']:
                raise ValueError('contradictory_terminal_stream_usage')
            result['usage'] = cumulative_usage(result['usage'], event['usage'])
            terminal_usage = finish is not None or any((c.get('finish_reason') is not None for c in choices))
        if len(choices) > 1:
            raise ValueError('multiple_stream_choices')
        if provider == 'openrouter' and finish is not None and (event.get('usage') is not None) and (len(choices) == 1) and (choices[0].get('index', 0) == 0) and (choices[0].get('finish_reason') == finish) and (set(choices[0].get('delta', {})) <= {'role', 'content'}) and (choices[0]['delta'].get('role') in (None, 'assistant')) and (not choices[0]['delta'].get('content')):
            choices = []
        for choice in choices:
            if choice.get('index', 0) != 0 or finish is not None:
                raise ValueError('invalid_stream_choice_sequence')
            delta = choice.get('delta', {})
            if delta.get('role') not in (None, 'assistant'):
                raise ValueError('invalid_stream_role')
            if set(delta) - {'role', 'content', 'reasoning', 'reasoning_content', 'tool_calls', 'refusal'} - ({'reasoning_details'} if provider == 'openrouter' else set()):
                raise ValueError('unrecognized_stream_delta')
            for key in ('content', 'reasoning', 'reasoning_content', 'refusal'):
                if delta.get(key) is not None:
                    if not isinstance(delta[key], str):
                        raise ValueError('nontext_stream_delta')
                    message[key] = message.get(key, '') + delta[key]
            for part in delta.get('tool_calls') or []:
                index = part.get('index')
                if type(index) is not int or not 0 <= index < 16:
                    raise ValueError('invalid_stream_tool_index')
                kind = part.get('type', 'function')
                if kind != 'function' and (not (kind is None and index in calls)):
                    raise ValueError('invalid_stream_tool_type')
                call = calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                for key in ('id',):
                    if part.get(key):
                        if call[key] and part[key] != call[key]:
                            raise ValueError('stream_tool_id_changed')
                        call[key] = part[key]
                for key in ('name', 'arguments'):
                    if part.get('function', {}).get(key) is not None:
                        call['function'][key] += part['function'][key]
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    if finish is None or sorted(calls) != list(range(len(calls))):
        raise ValueError('incomplete_chat_stream')
    if result['usage'] is not None and (not terminal_usage):
        raise ValueError('chat_stream_terminal_usage_missing')
    if calls:
        message['tool_calls'] = [calls[i] for i in sorted(calls)]
    result['choices'] = [{'index': 0, 'message': message, 'finish_reason': finish}]
    if created_values and (provider == 'xiaomi' or len(created_values) > 1):
        result['_stream_created_values'] = created_values
    return result

def _credential_pattern(secret):
    parts, maximum = ([], 0)
    short = {'"': b'\\"', '\\': b'\\\\', '/': b'\\/', '\x08': b'\\b', '\x0c': b'\\f', '\n': b'\\n', '\r': b'\\r', '\t': b'\\t'}
    for character in secret.decode('utf-8'):
        literal = character.encode('utf-8')
        alternatives = [re.escape(literal)] if ord(character) >= 32 and character not in ('"', '\\') else []
        utf16 = character.encode('utf-16-be')
        escaped = []
        for offset in range(0, len(utf16), 2):
            hexadecimal = utf16[offset:offset + 2].hex()
            escaped.append(b'\\\\u' + b''.join((('[' + digit + digit.upper() + ']').encode() if digit in 'abcdef' else digit.encode() for digit in hexadecimal)))
        alternatives.append(b''.join(escaped))
        if character in short:
            alternatives.append(re.escape(short[character]))
        parts.append(b'(?:' + b'|'.join(alternatives) + b')')
        maximum += max(len(literal), 6 * (len(utf16) // 2), len(short.get(character, b'')))
    return (re.compile(b'(?:' + re.escape(secret) + b')|' + b''.join(parts)), maximum)

class _Capture:

    def __init__(self, stream, secret, *, limit=None):
        self.stream, self.secret = (stream, secret)
        self.limit, self.written = (limit, 0)
        self.pending = b''
        self.redacted = False
        self.pattern, self.maximum_pattern_bytes = _credential_pattern(secret)
        self.replacement = next((value for value in (b'[credential-redacted]', b'[redacted]', b'[removed]') if not self.pattern.search(value)))

    def write(self, chunk, final=False):
        value = self.pending + chunk
        value, replacements = self.pattern.subn(self.replacement, value)
        self.redacted = self.redacted or replacements > 0
        keep = 0 if final else min(len(value), self.maximum_pattern_bytes - 1)
        safe = value[:-keep] if keep else value
        self.pending = value[-keep:] if keep else b''
        if self.limit is not None and self.written + len(safe) > self.limit:
            raise AttemptError('capture_member_byte_limit')
        self.stream.write(safe)
        self.written += len(safe)
        self.stream.flush()
        os.fsync(self.stream.fileno())

def bounded_connector(deadline):

    def connect(address, timeout=None, source_address=None, **kwargs):
        answers = queue.Queue(maxsize=1)

        def resolve():
            try:
                answers.put(socket.getaddrinfo(address[0], address[1], 0, socket.SOCK_STREAM))
            except BaseException as error:
                answers.put(error)
        threading.Thread(target=resolve, daemon=True, name='evaluation-dns').start()
        try:
            records = answers.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            raise TimeoutError('dns_deadline') from None
        if isinstance(records, BaseException):
            raise records
        last = None
        for family, kind, proto, _, destination in records:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('connect_deadline')
            sock = socket.socket(family, kind, proto)
            try:
                sock.settimeout(min(timeout, remaining) if isinstance(timeout, (float, int)) else remaining)
                if source_address:
                    sock.bind(source_address)
                sock.connect(destination)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('connect_deadline')
                sock.settimeout(min(timeout, remaining) if isinstance(timeout, (float, int)) else remaining)
                return sock
            except OSError as error:
                last = error
                sock.close()
        raise last or OSError('no_dns_addresses')
    return connect

class Transport:

    def __init__(self, profile, *, timeout=600, inactivity_timeout=120, max_response_bytes=16 * 1024 ** 2, max_request_bytes=8 * 1024 ** 2, connection_factory=None):
        if not 0 < timeout <= 7200 or not 0 < inactivity_timeout <= timeout:
            raise ValueError('finite_transport_deadlines_required')
        if not 1 <= max_response_bytes <= 64 * 1024 ** 2 or not 1 <= max_request_bytes <= 32 * 1024 ** 2:
            raise ValueError('finite_transport_byte_limits_required')
        profile.__post_init__()
        self.profile, self.timeout, self.inactivity_timeout = (profile, timeout, inactivity_timeout)
        self.max_response_bytes, self.max_request_bytes = (max_response_bytes, max_request_bytes)
        self.capture_limits = capture_limits(profile, max_request_bytes, max_response_bytes)
        self._factory = connection_factory

    def exchange(self, payload, destination, *, api_key=None, attempt_id=None, deadline=None, campaign_deadline=None, before_submit=None):
        started, started_epoch = (time.monotonic(), time.time())
        if any((value is not None and (type(value) not in (int, float) or not math.isfinite(value)) for value in (deadline, campaign_deadline))):
            raise ValueError('finite_absolute_transport_deadlines_required')
        deadline = min(started + self.timeout, deadline if deadline is not None else started + self.timeout)
        if campaign_deadline is not None:
            deadline = min(deadline, started + campaign_deadline - started_epoch)

        def remaining():
            seconds = deadline - time.monotonic()
            if campaign_deadline is not None:
                seconds = min(seconds, campaign_deadline - time.time())
            if seconds <= 0:
                raise AttemptError('total_deadline_exceeded')
            return seconds
        secret = api_key if api_key is not None else os.environ.get(self.profile.api_key_env)
        if not secret or len(secret) < 8 or any((c in secret for c in ('\r', '\n'))):
            raise ValueError('missing_or_invalid_provider_credential')
        secret_bytes = secret.encode()
        native_wire = None
        if self.profile.provider in ('anthropic', 'google'):
            from harness.api import native_providers
            native_wire = native_providers.wire(self.profile, payload)
            request = native_wire[0]
        else:
            request = encoded(payload)
        if len(request) > self.max_request_bytes or secret_bytes in request:
            raise ValueError('unsafe_or_oversized_request')
        if payload.get('model') != self.profile.model:
            raise ValueError('request_profile_model_mismatch')
        attempt_id = attempt_id or uuid.uuid4().hex
        if not isinstance(attempt_id, str) or not re.fullmatch('[0-9a-f]{32}', attempt_id):
            raise ValueError('invalid_client_attempt_id')
        path = Path(destination)
        path.mkdir(mode=448, parents=True, exist_ok=False)

        def save_member(name, value):
            if len(encoded(value)) + 1 > self.capture_limits[name]:
                raise AttemptError('capture_member_byte_limit')
            save(path / name, value)
        receipt = {'schema': 'causes.api_attempt.v1', 'attempt_id': attempt_id, 'profile': self.profile.record(), 'started_at': now(), 'request_sha256': hashlib.sha256(request).hexdigest(), 'request_bytes': len(request), 'automatic_retries': 0, 'provider_execution': 'not_submitted', 'status': 'prepared', 'total_timeout_seconds': self.timeout, 'inactivity_timeout_seconds': self.inactivity_timeout, 'max_response_bytes': self.max_response_bytes, 'capture_member_limits': self.capture_limits, 'submission_deadline_epoch': min(started_epoch + deadline - started, campaign_deadline if campaign_deadline is not None else float('inf'))}
        if native_wire is not None:
            receipt['wire_format'] = native_providers.WIRE_VERSION
            receipt['wire_omitted_fields'] = sorted(set(payload) - set(json.loads(request)))
            receipt['wire_headers'] = dict(native_wire[1], **{native_wire[2]: 'omitted_secret'})
        url = urlsplit(self.profile.endpoint)
        connection, timer = (None, None)
        expired = threading.Event()
        connected_sockets = []

        def expire():
            expired.set()
            sockets = [*connected_sockets, getattr(connection, 'sock', None)]
            for sock in sockets:
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if connection is not None:
                connection.close()
        raw = bytearray()
        capture = None
        body_complete, decoded_capture = (False, None)
        capture_format = 'provider_bytes'

        def check_decoded_capture(value, kind):
            nonlocal decoded_capture, capture_format
            try:
                decoded = encoded(value)
            except (UnicodeError, ValueError, RecursionError):
                return
            if len(decoded) + 1 > self.capture_limits['response.json']:
                return
            decoded_capture = kind
            if capture.pattern.search(decoded):
                safe = capture.pattern.sub(capture.replacement, decoded)
                if len(safe) > self.capture_limits['response.raw']:
                    safe = b'{"capture":"credential-reflection-redacted"}'
                capture.redacted = True
                capture_format = 'redacted_decoded_json'
                with (path / 'response.raw').open('wb') as stream:
                    stream.write(safe)
                    stream.flush()
                    os.fsync(stream.fileno())
        try:
            save_member('request.json', payload)
            remaining()
            save_member('attempt.json', receipt)
            timeout = min(self.inactivity_timeout, remaining())
            loopback = url.scheme == 'http' and url.hostname == '127.0.0.1'
            connection = self._factory(url, timeout) if self._factory else http.client.HTTPConnection(url.hostname, url.port, timeout=timeout) if loopback else http.client.HTTPSConnection(url.hostname, url.port, timeout=timeout, context=ssl.create_default_context())
            if self._factory is None:
                connection._create_connection = bounded_connector(deadline)
            timer = threading.Timer(remaining(), expire)
            timer.daemon = True
            timer.start()
            if callable(getattr(connection, 'connect', None)):
                connection.connect()
                connection.auto_open = 0
            if getattr(connection, 'sock', None) is not None:
                connected_sockets.append(connection.sock)
            remaining()
            save_member('dispatch.json', {'attempt_id': attempt_id, 'started_at': now(), 'provider_execution': 'possibly_executed'})
            remaining()
            if before_submit is not None:
                before_submit()
            timeout = min(self.inactivity_timeout, remaining())
            if expired.is_set():
                raise AttemptError('total_deadline_exceeded')
            if getattr(connection, 'sock', None) is not None:
                connection.sock.settimeout(timeout)
            receipt.update(status='request_in_flight', provider_execution='unknown', submitted_at=time.time())
            if native_wire is not None:
                headers = dict(native_wire[1], **{native_wire[2]: secret, 'X-Client-Request-Id': attempt_id})
            else:
                headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream' if payload.get('stream') else 'application/json', 'Authorization': 'Bearer ' + secret, 'X-Client-Request-Id': attempt_id}
            connection.request('POST', url.path, body=request, headers=headers)
            if getattr(connection, 'sock', None) is not None:
                connected_sockets.append(connection.sock)
            response = connection.getresponse()
            receipt['http_status'] = response.status
            allowed = {'content-type', 'retry-after', 'x-request-id', 'request-id', 'openai-processing-ms', 'x-ratelimit-limit-requests', 'x-ratelimit-remaining-requests', 'x-ratelimit-reset-requests', 'x-ratelimit-limit-tokens', 'x-ratelimit-remaining-tokens', 'x-ratelimit-reset-tokens'}
            if native_wire is not None:
                allowed |= {'anthropic-ratelimit-requests-remaining', 'anthropic-ratelimit-requests-reset', 'anthropic-ratelimit-input-tokens-remaining', 'anthropic-ratelimit-input-tokens-reset', 'anthropic-ratelimit-output-tokens-remaining', 'anthropic-ratelimit-output-tokens-reset', 'anthropic-ratelimit-tokens-remaining', 'anthropic-ratelimit-tokens-reset', 'x-goog-request-id'}
            observed_headers = response.getheaders()
            receipt['retry_after_headers_complete'] = all((len(v) <= 1024 and secret not in v for k, v in observed_headers if k.lower() == 'retry-after'))
            headers = [(k.lower(), v.replace(secret, '[credential-redacted]')[:1024]) for k, v in observed_headers if k.lower() in allowed]
            receipt['headers'] = dict(headers)
            receipt['retry_after_headers'] = [v for k, v in headers if k == 'retry-after']
            header_record = {'http_status': response.status, 'headers': receipt['headers'], 'retry_after_headers': receipt['retry_after_headers'], 'retry_after_headers_complete': receipt['retry_after_headers_complete']}
            if len(encoded(header_record)) + 1 > self.capture_limits['headers.json']:
                receipt.update(headers={}, retry_after_headers=[], retry_after_headers_complete=False, rejected_headers_sha256=hashlib.sha256(encoded(header_record)).hexdigest(), rejected_headers_bytes=len(encoded(header_record)))
                raise AttemptError('header_capture_byte_limit')
            save_member('headers.json', header_record)
            with (path / 'response.raw').open('xb') as stream:
                capture = _Capture(stream, secret_bytes, limit=self.capture_limits['response.raw'])
                try:
                    while True:
                        seconds = remaining()
                        if expired.is_set():
                            raise AttemptError('total_deadline_exceeded')
                        sock = connected_sockets[-1] if connected_sockets else getattr(connection, 'sock', None)
                        if sock:
                            sock.settimeout(min(self.inactivity_timeout, seconds))
                        try:
                            chunk = response.read1(min(65536, self.max_response_bytes + 1 - len(raw)))
                        except http.client.IncompleteRead as error:
                            partial = error.partial[:self.max_response_bytes + 1 - len(raw)]
                            raw.extend(partial)
                            capture.write(partial)
                            raise AttemptError('incomplete_http_body') from None
                        if not chunk:
                            remaining()
                            if expired.is_set():
                                raise AttemptError('total_deadline_exceeded')
                            timer.cancel()
                            break
                        chunk = chunk[:self.max_response_bytes + 1 - len(raw)]
                        raw.extend(chunk)
                        capture.write(chunk)
                        if len(raw) > self.max_response_bytes:
                            raise AttemptError('response_byte_limit')
                finally:
                    capture.write(b'', final=True)
            if response.getheader('Content-Length') is not None and (not response.chunked):
                if len(raw) != int(response.getheader('Content-Length')):
                    raise AttemptError('incomplete_http_body')
            body_complete = True
            if response.status != 200 or not payload.get('stream'):
                try:
                    value = strict_json(raw)
                except (ValueError, UnicodeError, RecursionError):
                    pass
                else:
                    check_decoded_capture(value, 'json')
            if capture.redacted:
                raise AttemptError('credential_reflection_rejected', http_status=response.status)
            if response.status != 200:
                if response.status == 429:
                    receipt['provider_execution'] = 'rate_limited'
                raise AttemptError('http_' + str(response.status), http_status=response.status, retry_after=receipt['headers'].get('retry-after'))
            media = receipt['headers'].get('content-type', '').split(';', 1)[0].strip()
            if payload.get('stream'):
                if media != 'text/event-stream':
                    raise AttemptError('unexpected_stream_content_type')
                result = decode_sse(bytes(raw), self.profile.family, provider=self.profile.provider, opaque_model=self.profile.model if self.profile.family == 'responses' and self.profile.reasoning == 'opaque' else None, allow_keepalive=self.profile.id in ('openai-o3-high', 'openai-gpt-5.5-high'))
            else:
                if media not in ('application/json', ''):
                    raise AttemptError('unexpected_response_content_type')
                result = strict_json(raw)
            if not isinstance(result, dict):
                raise AttemptError('response_object_required')
            if len(encoded(result)) + 1 > self.capture_limits['response.json']:
                raise AttemptError('decoded_response_byte_limit')
            check_decoded_capture(result, 'supported_stream' if payload.get('stream') else 'json')
            if capture.redacted:
                raise AttemptError('credential_reflection_rejected', http_status=response.status)
            save_member('response.json', result)
            if native_wire is not None and self.profile.provider == 'google':
                receipt.update(status='complete', provider_execution='response_received', usage=result.get('usageMetadata'), returned_model=result.get('modelVersion'), response_id=result.get('responseId'), system_fingerprint=None)
            else:
                receipt.update(status='complete', provider_execution='response_received', usage=result.get('usage'), returned_model=result.get('model'), response_id=result.get('id'), system_fingerprint=result.get('system_fingerprint'))
            return result
        except BaseException as error:
            receipt['status'] = error.status if isinstance(error, AttemptError) else 'total_deadline_exceeded' if expired.is_set() or time.monotonic() >= deadline else 'transport_or_parse_failure'
            receipt['error_type'] = type(error).__name__
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(error, AttemptError):
                raise
            if 'submitted_at' not in receipt:
                raise
            raise AttemptError(receipt['status']) from None
        finally:
            if timer is not None:
                timer.cancel()
            if connection is not None:
                connection.close()
            receipt.update(finished_at=now(), elapsed_seconds=time.monotonic() - started, captured_bytes=(path / 'response.raw').stat().st_size if (path / 'response.raw').exists() else 0, capture_redacted=bool(capture and capture.redacted), capture_credential_scan='literal_and_mixed_json_escapes.v1' if capture else None, capture_body_complete=body_complete, capture_decoded_scan=decoded_capture, capture_format=capture_format, capture_quarantined=bool(capture and (capture.redacted or not body_complete or decoded_capture is None)))
            if (path / 'response.raw').exists():
                receipt['captured_sha256'] = hashlib.sha256((path / 'response.raw').read_bytes()).hexdigest()
            save_member('receipt.json', receipt)

"""Preserve the frozen Qwen request, replay, and streaming-response semantics."""

import copy
import hashlib
import json
import math

def chat_history(messages):
    result = []
    for message in messages:
        if '_harmony_output' in message:
            raise ValueError('cannot_replay_harmony_state_to_chat_provider')
        if '_provider_output' in message:
            raise ValueError('cannot_replay_responses_state_to_chat_provider')
        if '_refusal' in message:
            raise ValueError('chat_refusal_continuation_not_qualified')
        item = {k: copy.deepcopy(v) for k, v in message.items() if not k.startswith('_')}
        if set(item) - {'role', 'content', 'tool_calls', 'tool_call_id', 'reasoning_content', 'reasoning', 'name'}:
            raise ValueError('unrecognized_chat_history_field')
        if 'reasoning' in item:
            if 'reasoning_content' in item and item['reasoning'] != item['reasoning_content']:
                raise ValueError('conflicting_reasoning_aliases')
            item['reasoning_content'] = item.pop('reasoning')
        result.append(item)
    return result

def _text(value, label, *, nullable=False):
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError('nontext_' + label)
    return value

def _calls(calls):
    if not isinstance(calls, list) or len(calls) > 16:
        raise ValueError('invalid_tool_batch')
    ids = []
    for call in calls:
        if call.get('type') != 'function' or not isinstance(call.get('id'), str) or (not call['id']):
            raise ValueError('invalid_tool_call_identity')
        f = call.get('function', {})
        _text(f.get('name'), 'tool_name')
        _text(f.get('arguments'), 'tool_arguments')
        ids.append(call['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate_tool_call_identity')

def response(model, raw):
    if raw.get('error'):
        raise ValueError('provider_response_error')
    if not isinstance(raw.get('model'), str):
        raise ValueError('missing_returned_model_identity')
    if raw['model'] != model:
        raise ValueError('unexpected_returned_model')
    choices = raw.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError('exactly_one_choice_required')
    choice = choices[0]
    original = choice['message']
    if original.get('role') != 'assistant':
        raise ValueError('assistant_response_required')
    unknown = set(original) - {'role', 'content', 'tool_calls', 'reasoning', 'reasoning_content', 'refusal'}
    if any((original[k] is not None for k in unknown)):
        raise ValueError('unsupported_provider_message_fields:' + ','.join(sorted(unknown)))
    message = {'role': 'assistant', 'content': _text(original.get('content'), 'content', nullable=True)}
    for field in ('reasoning', 'reasoning_content'):
        if original.get(field) is not None:
            value = _text(original[field], field)
            if 'reasoning_content' in message and message['reasoning_content'] != value:
                raise ValueError('conflicting_returned_reasoning')
            message['reasoning_content'] = value
    if original.get('tool_calls'):
        message['tool_calls'] = copy.deepcopy(original['tool_calls'])
    finish = choice.get('finish_reason')
    if original.get('refusal'):
        finish = 'refusal'
        message['_refusal'] = _text(original['refusal'], 'refusal')
    _calls(message.get('tool_calls', []))
    if finish == 'stop' and (not message.get('tool_calls')) and (not (message.get('content') or '').strip()):
        finish = 'empty_completion'
    reason = message.get('reasoning_content')
    message.setdefault('_reasoning_status', 'missing' if reason is None else 'captured' if reason.strip() else 'empty')
    message['_provider_model'] = raw['model']
    return {'choices': [{'finish_reason': finish, 'message': message}], 'usage': copy.deepcopy(raw.get('usage')), 'model': raw['model'], 'system_fingerprint': raw.get('system_fingerprint')}

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

def decode_sse(raw):
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
            raise ValueError('stream_error_event')
        events.append(event)
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
                if key in result and result[key] != event[key]:
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
        for choice in choices:
            if choice.get('index', 0) != 0 or finish is not None:
                raise ValueError('invalid_stream_choice_sequence')
            delta = choice.get('delta', {})
            if delta.get('role') not in (None, 'assistant'):
                raise ValueError('invalid_stream_role')
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
    return result

class Protocol:
    name = 'api_full_actor_only_v1'

    def request(self, config, messages, seed, tools):
        maximum = config['api_max_output_tokens']
        if type(maximum) is not int or not 256 <= maximum <= 32768:
            raise ValueError('invalid_request_output_limit')
        payload = {'temperature': 1.0, 'top_p': 0.95, 'model': config['model'],
                   'stream': True, 'max_tokens': maximum, 'messages': chat_history(messages),
                   'stream_options': {'include_usage': True}}
        if tools:
            payload.update(tools=copy.deepcopy(tools), tool_choice='auto', parallel_tool_calls=True)
        payload['seed'] = int(hashlib.sha256(repr(seed).encode()).hexdigest()[:15], 16)
        return payload

    def __init__(self, model='qwen3.8-27b'):
        self.model = model

    def check_response(self, raw, messages):
        return response(self.model, raw)

    def replay(self, message):
        return copy.deepcopy(message)

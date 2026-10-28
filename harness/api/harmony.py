from __future__ import annotations
import hashlib
import json
import re
from harness.api.transport import strict_json
CONTRACT = 'deepinfra_harmony_text.v1'
START, MESSAGE, END = ('<|start|>', '<|message|>', '<|end|>')
CHANNEL, CONSTRAIN = ('<|channel|>', '<|constrain|>')
CALL, RETURN = ('<|call|>', '<|return|>')
ENDINGS = re.compile('<\\|(?:end|call|return)\\|>')
FUNCTIONS = {'read': 'path', 'bash': 'command'}
FUNCTION_NAME = re.compile('[A-Za-z0-9_-]{1,64}\\Z')

def literal(value):
    if not isinstance(value, str) or '<|' in value:
        raise ValueError('harmony_literal_marker_or_nontext_input')
    return value

def check_literals(value):
    if isinstance(value, str):
        literal(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            literal(key)
            check_literals(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            check_literals(item)

def validate_options(options, effort):
    if not isinstance(options, dict) or set(options) - {'context_tokenizer'} != {'contract', 'system_metadata', 'schema_instruction'} or options['contract'] != CONTRACT:
        raise ValueError('explicit_harmony_text_contract_required')
    check_literals(options)
    if not 1 <= len(options['system_metadata']) <= 8192 or 'Reasoning: ' + effort not in options['system_metadata'].splitlines() or options['schema_instruction'].count('{schema}') != 1 or (len(options['schema_instruction']) > 8192):
        raise ValueError('invalid_harmony_metadata_or_schema_instruction')

def _header(value):
    if value.count(CHANNEL) != 1 or value.count(CONSTRAIN) > 1:
        raise ValueError('invalid_harmony_header')
    before, after = value.split(CHANNEL)
    format_ = None
    if CONSTRAIN in after:
        after, format_ = after.split(CONSTRAIN)
        format_ = format_.strip()
        if format_.lower() != 'json':
            raise ValueError('unsupported_harmony_content_type')
    prefix, fields = (before.split(), after.split())
    if not prefix or prefix.pop(0) != 'assistant':
        raise ValueError('unexpected_harmony_role')
    if not fields or fields[0] not in ('analysis', 'commentary', 'final'):
        raise ValueError('unsupported_harmony_channel_or_recipient')
    channel, recipient = (fields.pop(0), None)
    if fields and fields[-1].lower() in ('json', 'code'):
        if format_ is not None:
            raise ValueError('duplicate_harmony_content_type')
        format_ = fields.pop()
    recipients = prefix + fields
    if recipients:
        if len(recipients) != 1 or not recipients[0].startswith('to=functions.'):
            raise ValueError('unsupported_harmony_header_field')
        recipient = recipients[0][len('to=functions.'):]
    if recipient is not None and (not FUNCTION_NAME.fullmatch(recipient)):
        raise ValueError('unsupported_harmony_channel_or_recipient')
    return (channel, recipient, format_)

def decode(text, finish, response_id):
    if not isinstance(text, str) or not isinstance(response_id, str) or (not response_id):
        raise ValueError('raw_completion_identity_and_text_required')
    if finish != 'stop':
        if finish not in ('length', 'content_filter'):
            raise ValueError('unknown_raw_completion_termination')
        return ({'role': 'assistant', 'content': None, '_reasoning_status': 'partial', '_harmony_output': {'contract': CONTRACT, 'text': text, 'finish_reason': finish, 'response_id': response_id, 'frames': [], 'termination_source': 'incomplete_provider_status'}}, finish)
    wire = START + 'assistant' + text
    position, frames = (0, [])
    while position < len(wire):
        if len(frames) >= 256 or not wire.startswith(START, position):
            raise ValueError('invalid_or_excessive_harmony_frames')
        split = wire.find(MESSAGE, position + len(START))
        if split < 0 or split - position > 512:
            raise ValueError('incomplete_harmony_header')
        header = wire[position + len(START):split]
        channel, recipient, format_ = _header(header)
        body_start = split + len(MESSAGE)
        ending = ENDINGS.search(wire, body_start)
        body_end = ending.start() if ending else len(wire)
        body = literal(wire[body_start:body_end])
        token = ending.group() if ending else None
        frames.append({'header': header, 'channel': channel, 'recipient': recipient, 'content_type': format_, 'body': body, 'captured_ending': token})
        position = ending.end() if ending else len(wire)
        if position < len(wire) and (token != END or recipient):
            raise ValueError('harmony_data_after_terminal')
    if not frames:
        raise ValueError('empty_harmony_completion')
    terminal = frames[-1]
    if terminal['recipient']:
        if terminal['captured_ending'] not in (None, CALL):
            raise ValueError('harmony_tool_handoff_mismatch')
        if terminal['captured_ending'] is None and terminal['channel'] == 'final':
            raise ValueError('unqualified_harmony_omitted_final_recipient_ending')
        try:
            arguments = strict_json(terminal['body'])
        except json.JSONDecodeError:
            arguments = None
        name = terminal['recipient']
        check_literals(arguments)
        call_id = 'call_' + hashlib.sha256((response_id + '\x00' + text).encode()).hexdigest()[:32]
        calls = [{'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': terminal['body']}}]
        normalized_finish = 'tool_calls'
    elif (terminal['captured_ending'] == RETURN or (terminal['channel'] == 'final' and terminal['captured_ending'] is None)) and terminal['body'].strip():
        calls, normalized_finish = ([], 'stop')
    else:
        raise ValueError('harmony_terminal_action_missing_or_ambiguous')
    reasoning, content, segments, offset = ([], [], [], 0)
    for index, frame in enumerate(frames):
        if frame['channel'] == 'analysis' and frame['recipient'] is None:
            reasoning.append(frame['body'])
            segments.append({'frame_index': index, 'start': offset, 'end': offset + len(frame['body'])})
            offset += len(frame['body'])
        elif not frame['recipient']:
            content.append(frame['body'])
    message = {'role': 'assistant', 'content': ''.join(content) or None, '_harmony_output': {'contract': CONTRACT, 'text': text, 'finish_reason': finish, 'response_id': response_id, 'frames': frames, 'termination_source': 'captured_text_ending' if terminal['captured_ending'] else 'provider_stop_and_header'}}
    if reasoning:
        message.update(reasoning_content=''.join(reasoning), _reasoning_segments=segments)
    if calls:
        message['tool_calls'] = calls
    return (message, normalized_finish)

def _tools(tools):
    check_literals(tools)
    declarations, names = ([], set())
    for tool in tools:
        if set(tool) != {'type', 'function'} or tool['type'] != 'function':
            raise ValueError('harmony_function_tool_required')
        f = tool['function']
        name = f['name']
        if name not in FUNCTIONS or name in names or set(f) - {'name', 'description', 'parameters'}:
            raise ValueError('unsupported_harmony_tool_definition')
        key = FUNCTIONS[name]
        schema = f['parameters']
        if schema != {'type': 'object', 'properties': {key: {'type': 'string'}}, 'required': [key], 'additionalProperties': False}:
            raise ValueError('unsupported_harmony_tool_schema')
        names.add(name)
        description = literal(f.get('description', '')).replace('\n', '\n// ')
        declarations.append('// ' + description + '\ntype ' + name + ' = (_: { ' + key + ': string }) => any;')
    return '## functions\n\nnamespace functions {\n\n' + '\n\n'.join(declarations) + '\n\n} // namespace functions'

def render(messages, tools, options, schema=None):
    if not messages or messages[0].get('role') not in ('system', 'developer'):
        messages = [{'role': 'system', 'content': ''}, *messages]
    instructions = literal(messages[0].get('content') or '')
    if set(messages[0]) - {'role', 'content'}:
        raise ValueError('unsupported_harmony_instruction_fields')
    developer = '# Instructions\n\n' + instructions
    if tools:
        developer += '\n\n# Tools\n\n' + _tools(tools)
    if schema is not None:
        check_literals(schema)
        serialized = json.dumps(schema, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
        developer += '\n\n' + options['schema_instruction'].replace('{schema}', serialized)
    pieces = [START + 'system' + MESSAGE + literal(options['system_metadata']) + END, START + 'developer' + MESSAGE + developer + END]
    pending, seen = ({}, set())
    for message in messages[1:]:
        role = message.get('role')
        if set(message) - {'role', 'content', 'tool_calls', 'tool_call_id', 'reasoning_content', 'reasoning'} - {k for k in message if k.startswith('_')}:
            raise ValueError('unsupported_harmony_history_fields')
        if '_provider_output' in message or '_refusal' in message:
            raise ValueError('foreign_provider_state_in_harmony_history')
        if role == 'tool':
            name = pending.pop(message.get('tool_call_id'), None)
            if name is None:
                raise ValueError('harmony_tool_result_without_matching_call')
            pieces.append(START + 'functions.' + name + MESSAGE + literal(message['content']) + END)
            continue
        if pending:
            raise ValueError('harmony_history_has_unresolved_tool_calls')
        if role == 'user':
            if any((message.get(k) for k in ('reasoning', 'reasoning_content', 'tool_calls'))):
                raise ValueError('invalid_harmony_user_fields')
            pieces.append(START + 'user' + MESSAGE + literal(message['content']) + END)
        elif role == 'assistant':
            if '_harmony_output' in message:
                native = message['_harmony_output']
                expected, _ = decode(native['text'], native['finish_reason'], native['response_id'])
                if native != expected['_harmony_output'] or native['finish_reason'] != 'stop' or any((json.dumps(message.get(k), sort_keys=True, allow_nan=False) != json.dumps(expected.get(k), sort_keys=True, allow_nan=False) for k in ('content', 'reasoning_content', 'tool_calls', '_reasoning_segments'))) or ('reasoning' in message and message['reasoning'] != expected.get('reasoning_content')):
                    raise ValueError('harmony_native_replay_mismatch')
                reason = expected.get('reasoning_content')
                status = 'missing' if reason is None else 'captured' if reason.strip() else 'empty'
                if '_reasoning_status' in message and message['_reasoning_status'] != status:
                    raise ValueError('harmony_native_reasoning_status_mismatch')
                for frame in native['frames']:
                    ending = CALL if frame['recipient'] else END
                    pieces.append(START + frame['header'] + MESSAGE + frame['body'] + ending)
            else:
                if message.get('tool_calls'):
                    raise ValueError('harmony_tool_replay_requires_native_frames')
                reason = message.get('reasoning_content', message.get('reasoning'))
                if 'reasoning_content' in message and 'reasoning' in message and (message['reasoning_content'] != message['reasoning']):
                    raise ValueError('conflicting_reasoning_aliases')
                if reason is not None:
                    pieces.append(START + 'assistant' + CHANNEL + 'analysis' + MESSAGE + literal(reason) + END)
                if message.get('content') is not None:
                    pieces.append(START + 'assistant' + CHANNEL + 'final' + MESSAGE + literal(message['content']) + END)
            for call in message.get('tool_calls') or []:
                if call['id'] in seen:
                    raise ValueError('duplicate_harmony_tool_identity')
                seen.add(call['id'])
                pending[call['id']] = call['function']['name']
        else:
            raise ValueError('unsupported_harmony_history_role')
    if pending:
        raise ValueError('harmony_history_has_unresolved_tool_calls')
    return ''.join(pieces) + START + 'assistant'

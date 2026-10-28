from __future__ import annotations
import copy
import datetime as dt
import json
WIRE_VERSION = 'causes.native_wire.v1'
FAMILIES = ('messages', 'gemini')
ANTHROPIC_HOST = 'api.anthropic.com'
ANTHROPIC_PATH = '/v1/messages'
ANTHROPIC_VERSION = '2026-10-21'
BINDING_BETA = 'thinking-binding-controls-2026-08-01'
GOOGLE_HOST = 'generativelanguage.googleapis.com'
ANTHROPIC_FORMAT = 'anthropic_messages.v1'
GOOGLE_FORMAT = 'gemini_generate_content.v1'
GOOGLE_MODELS = ('gemini-4-argon',)
EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
THINKING_LEVELS = ('LOW', 'MEDIUM', 'HIGH')
GOOGLE_REFUSALS = ('SAFETY', 'PROHIBITED_CONTENT', 'BLOCKLIST', 'SPII', 'RECITATION', 'LANGUAGE', 'ESCALATION', 'IMAGE_SAFETY', 'IMAGE_PROHIBITED_CONTENT', 'PUP_LIMITED_DISABLED')

def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()

def google_path(model):
    return '/v1beta/models/' + model + ':generateContent'

def validate_profile(profile):
    s = profile.settings
    if profile.reasoning != 'summary' or profile.engineering_only is not True:
        raise ValueError('native_summary_route_requires_engineering_only')
    if profile.provider == 'anthropic':
        if profile.family != 'messages':
            raise ValueError('unqualified_anthropic_model')
        if set(s) != {'effort', 'thinking_display', 'cache_control', 'prefix_mismatch_behavior'}:
            raise ValueError('anthropic_settings_incomplete')
        if s['effort'] not in EFFORTS:
            raise ValueError('explicit_anthropic_effort_required')
        if s['thinking_display'] != 'summarized':
            raise ValueError('anthropic_summarized_thinking_required')
        if s['cache_control'] not in ({'type': 'ephemeral'}, {'type': 'ephemeral', 'ttl': '5m'}, {'type': 'ephemeral', 'ttl': '1h'}):
            raise ValueError('anthropic_prompt_caching_required')
        if s['prefix_mismatch_behavior'] not in (None, 'error'):
            raise ValueError('anthropic_actor_history_must_not_drop_thinking')
    elif profile.provider == 'google':
        if profile.model not in GOOGLE_MODELS or profile.family != 'gemini':
            raise ValueError('unqualified_google_model')
        if set(s) != {'thinking_level', 'include_thoughts'} or s['include_thoughts'] is not True:
            raise ValueError('google_thought_summaries_required')
        if s['thinking_level'] not in THINKING_LEVELS:
            raise ValueError('explicit_google_thinking_level_required')
        if profile.max_output_tokens > 65536:
            raise ValueError('google_synchronous_output_limit_exceeded')
    else:
        raise ValueError('not_a_native_provider')

def _system_and_rest(messages):
    if not isinstance(messages, list) or not all((isinstance(m, dict) for m in messages)):
        raise ValueError('native_messages_required')
    system = None
    rest = messages
    if messages and messages[0].get('role') == 'system':
        system = messages[0].get('content')
        if not isinstance(system, str) or set(messages[0]) - {'role', 'content'}:
            raise ValueError('native_system_prompt_must_be_text')
        rest = messages[1:]
    if any((m.get('role') in ('system', 'developer') for m in rest)):
        raise ValueError('native_route_has_one_leading_system_prompt')
    return (system, rest)

def _plain(message):
    if message.get('tool_calls') or message.get('reasoning_content') or message.get('reasoning') or (not isinstance(message.get('content'), str)):
        raise ValueError('native_replay_requires_native_assistant_output')
    return message['content']

def _results_first(items, is_result):
    seen_other = False
    for item in items:
        if is_result(item) and seen_other:
            raise ValueError('tool_results_must_precede_text')
        seen_other |= not is_result(item)

def anthropic_history(messages):
    system, rest = _system_and_rest(messages)
    wire, pending, paused = ([], {}, False)
    for message in rest:
        role = message.get('role')
        resumed = paused
        paused = role == 'assistant' and message.get('_provider_stop_reason') == 'pause_turn'
        if role == 'user':
            if message.get('_provider_output') is not None or not isinstance(message.get('content'), str):
                raise ValueError('native_user_turn_must_be_text')
            blocks = [{'type': 'text', 'text': message['content']}]
            kind = 'user'
        elif role == 'assistant':
            if pending:
                raise ValueError('unfinished_tool_batch')
            if '_provider_output' in message:
                if message.get('_provider_format') != ANTHROPIC_FORMAT:
                    raise ValueError('cannot_replay_other_provider_state_to_anthropic')
                blocks = copy.deepcopy(message['_provider_output'])
                if not isinstance(blocks, list) or not blocks:
                    raise ValueError('empty_native_assistant_output')
                pending = {b['id']: b['name'] for b in blocks if b.get('type') == 'tool_use'}
            else:
                blocks = [{'type': 'text', 'text': _plain(message)}]
            kind = 'assistant'
        elif role == 'tool':
            ident = message.get('tool_call_id')
            if ident not in pending or not isinstance(message.get('content'), str):
                raise ValueError('orphan_or_nontext_tool_result')
            del pending[ident]
            block = {'type': 'tool_result', 'tool_use_id': ident}
            if message['content']:
                block['content'] = message['content']
            blocks, kind = ([block], 'user')
        else:
            raise ValueError('unsupported_native_history_role')
        if wire and wire[-1]['role'] == kind:
            if kind == 'assistant' and (not resumed):
                raise ValueError('consecutive_assistant_turns')
            wire[-1]['content'].extend(blocks)
        else:
            wire.append({'role': kind, 'content': blocks})
    if pending:
        raise ValueError('unfinished_tool_batch')
    for message in wire:
        if message['role'] == 'user':
            _results_first(message['content'], lambda b: b['type'] == 'tool_result')
    return (system, wire)

def google_history(messages):
    system, rest = _system_and_rest(messages)
    contents, pending = ([], {})
    for message in rest:
        role = message.get('role')
        if role == 'user':
            if message.get('_provider_output') is not None or not isinstance(message.get('content'), str):
                raise ValueError('native_user_turn_must_be_text')
            parts, kind = ([{'text': message['content']}], 'user')
        elif role == 'assistant':
            if pending:
                raise ValueError('unfinished_tool_batch')
            if '_provider_output' in message:
                if message.get('_provider_format') != GOOGLE_FORMAT:
                    raise ValueError('cannot_replay_other_provider_state_to_google')
                parts = copy.deepcopy(message['_provider_output'])
                if not isinstance(parts, list) or not parts:
                    raise ValueError('empty_native_assistant_output')
                calls = [p['functionCall'] for p in parts if 'functionCall' in p]
                internal = message.get('tool_calls') or []
                if len(calls) != len(internal):
                    raise ValueError('gemini_call_identity_mismatch')
                pending = {i['id']: c for i, c in zip(internal, calls)}
            else:
                parts = [{'text': _plain(message)}]
            kind = 'model'
        elif role == 'tool':
            ident = message.get('tool_call_id')
            if ident not in pending or not isinstance(message.get('content'), str):
                raise ValueError('orphan_or_nontext_tool_result')
            call = pending.pop(ident)
            response = {'name': call['name'], 'response': {'output': message['content']}}
            if call.get('id'):
                response['id'] = call['id']
            parts, kind = ([{'functionResponse': response}], 'user')
        else:
            raise ValueError('unsupported_native_history_role')
        if contents and contents[-1]['role'] == kind:
            if kind == 'model':
                raise ValueError('consecutive_model_turns')
            contents[-1]['parts'].extend(parts)
        else:
            contents.append({'role': kind, 'parts': parts})
    if pending:
        raise ValueError('unfinished_tool_batch')
    for content in contents:
        if content['role'] == 'user':
            _results_first(content['parts'], lambda p: 'functionResponse' in p)
    return (system, contents)

def _function(tool):
    if tool.get('type') != 'function' or set(tool) != {'type', 'function'}:
        raise ValueError('native_tools_must_be_functions')
    f = tool['function']
    if set(f) - {'name', 'description', 'parameters'} or not isinstance(f.get('name'), str):
        raise ValueError('unsupported_tool_definition_field')
    return f

def anthropic_tools(tools):
    return [{'name': f['name'], 'description': f.get('description', ''), 'input_schema': copy.deepcopy(f['parameters'])} for f in map(_function, tools)]

def google_tools(tools):
    return [{'functionDeclarations': [{'name': f['name'], 'description': f.get('description', ''), 'parametersJsonSchema': copy.deepcopy(f['parameters'])} for f in map(_function, tools)]}]

def request(profile, messages, tools, *, max_output_tokens, stream, tool_choice='auto', prefix_mismatch=None):
    if tool_choice not in ('auto', 'none'):
        raise ValueError('native_tool_choice_unqualified')
    s = profile.settings
    if profile.provider == 'anthropic':
        system, wire = anthropic_history(messages)
        thinking = {'type': 'adaptive', 'display': s['thinking_display']}
        behavior = prefix_mismatch if prefix_mismatch is not None else s['prefix_mismatch_behavior']
        if behavior not in (None, 'error', 'drop_block'):
            raise ValueError('invalid_prefix_mismatch_behavior')
        payload = {'model': profile.model, 'max_tokens': max_output_tokens, 'stream': bool(stream), 'messages': wire, 'thinking': thinking, 'output_config': {'effort': s['effort']}, 'cache_control': copy.deepcopy(s['cache_control'])}
        if behavior is not None:
            thinking['block_binding'] = {'prefix_mismatch_behavior': behavior}
            payload['betas'] = [BINDING_BETA]
        if system is not None:
            payload['system'] = system
        if tools:
            payload['tools'] = anthropic_tools(tools)
            payload['tool_choice'] = {'type': tool_choice}
        return payload
    if profile.provider == 'google':
        system, contents = google_history(messages)
        payload = {'model': profile.model, 'contents': contents, 'generationConfig': {'maxOutputTokens': max_output_tokens, 'thinkingConfig': {'thinkingLevel': s['thinking_level'], 'includeThoughts': True}}}
        if system is not None:
            payload['systemInstruction'] = {'parts': [{'text': system}]}
        if tools:
            payload['tools'] = google_tools(tools)
            payload['toolConfig'] = {'functionCallingConfig': {'mode': tool_choice.upper()}}
        return payload
    raise ValueError('not_a_native_provider')

def output_limit(payload, profile):
    return payload.get('max_tokens') if profile.provider == 'anthropic' else (payload.get('generationConfig') or {}).get('maxOutputTokens')

def context_bound(payload, profile, *, enforce=True):
    output = output_limit(payload, profile)
    allowed = {'model', 'max_tokens', 'stream', 'messages', 'thinking', 'output_config', 'cache_control', 'betas', 'system', 'tools', 'tool_choice'} if profile.provider == 'anthropic' else {'model', 'contents', 'generationConfig', 'systemInstruction', 'tools', 'toolConfig'}
    if payload.get('model') != profile.model or type(output) is not int or (not 256 <= output <= profile.max_output_tokens) or set(payload) - allowed:
        raise ValueError('request_identity_or_output_limit_mismatch')
    source = {k: v for k, v in payload.items() if k in ('system', 'messages', 'tools', 'systemInstruction', 'contents')}
    estimate = len(encoded(source)) + 4096
    bound = {'method': 'utf8_bytes_plus_4096.v1', 'input_token_upper_estimate': estimate, 'output_token_reservation': output, 'context_ceiling': profile.context_tokens}
    if estimate + output > profile.context_tokens:
        if enforce:
            from harness.api.protocol import ContextLimit
            raise ContextLimit('uncropped_context_exceeds_conservative_admission')
        bound['admission'] = 'provider_enforced'
    return bound

def wire(profile, payload):
    if profile.provider == 'anthropic':
        body = {k: v for k, v in payload.items() if k != 'betas'}
        headers = {'Content-Type': 'application/json', 'Accept': 'text/event-stream' if payload.get('stream') else 'application/json', 'anthropic-version': ANTHROPIC_VERSION}
        if payload.get('betas'):
            if payload['betas'] != [BINDING_BETA]:
                raise ValueError('unqualified_anthropic_beta')
            headers['anthropic-beta'] = ','.join(payload['betas'])
        return (encoded(body), headers, 'x-api-key')
    if profile.provider == 'google':
        body = {k: v for k, v in payload.items() if k != 'model'}
        return (encoded(body), {'Content-Type': 'application/json', 'Accept': 'application/json'}, 'x-goog-api-key')
    raise ValueError('not_a_native_provider')

def messages_stream(events):
    events = [e for e in events if e.get('type') != 'ping']
    if not events or events[0].get('type') != 'message_start' or events[-1].get('type') != 'message_stop':
        raise ValueError('messages_stream_start_or_stop_missing')
    message = copy.deepcopy(events[0]['message'])
    if message.get('content') != [] or message.get('type') != 'message' or message.get('role') != 'assistant':
        raise ValueError('messages_stream_unsupported_initial_snapshot')
    usage = dict(message.get('usage') or {})
    blocks, open_index, partial, finished_delta = ([], None, {}, False)
    for event in events[1:-1]:
        kind = event.get('type')
        if kind == 'content_block_start':
            if open_index is not None or event.get('index') != len(blocks) or finished_delta:
                raise ValueError('messages_stream_block_order')
            block = copy.deepcopy(event['content_block'])
            if block.get('type') == 'tool_use':
                if block.get('input') not in ({}, None):
                    raise ValueError('messages_stream_nonempty_initial_tool_input')
                partial[len(blocks)] = []
            blocks.append(block)
            open_index = event['index']
        elif kind == 'content_block_delta':
            if event.get('index') != open_index:
                raise ValueError('messages_stream_delta_order')
            block, delta = (blocks[open_index], event.get('delta') or {})
            field = {'text_delta': ('text', 'text'), 'thinking_delta': ('thinking', 'thinking'), 'signature_delta': ('signature', 'signature'), 'input_json_delta': (None, 'partial_json')}.get(delta.get('type'))
            if field is None or not isinstance(delta.get(field[1]), str):
                raise ValueError('unsupported_messages_stream_delta')
            expected = {'text_delta': 'text', 'thinking_delta': 'thinking', 'signature_delta': 'thinking', 'input_json_delta': 'tool_use'}[delta['type']]
            if block.get('type') != expected:
                raise ValueError('messages_stream_delta_block_mismatch')
            if field[0] is None:
                partial[open_index].append(delta['partial_json'])
            else:
                block[field[0]] = block.get(field[0], '') + delta[field[1]]
        elif kind == 'content_block_stop':
            if event.get('index') != open_index:
                raise ValueError('messages_stream_stop_order')
            if open_index in partial:
                text = ''.join(partial.pop(open_index))
                value = json.loads(text) if text else {}
                if not isinstance(value, dict):
                    raise ValueError('tool_input_must_be_object')
                blocks[open_index]['input'] = value
            open_index = None
        elif kind == 'message_delta':
            if open_index is not None or finished_delta:
                raise ValueError('messages_stream_delta_before_blocks_closed')
            finished_delta = True
            for key, value in (event.get('delta') or {}).items():
                message[key] = copy.deepcopy(value)
            for key, value in (event.get('usage') or {}).items():
                if value is not None:
                    usage[key] = copy.deepcopy(value)
        else:
            raise ValueError('unsupported_messages_stream_event')
    if open_index is not None or not finished_delta or message.get('stop_reason') is None:
        raise ValueError('messages_stream_unfinished')
    message['content'], message['usage'] = (blocks, usage)
    return message

def _summary(message, segments):
    text = ''.join((value for _, value in segments))
    if segments:
        message['_reasoning_summary'] = text
        offset, spans = (0, [])
        for index, value in segments:
            spans.append({'native_index': index, 'start': offset, 'end': offset + len(value)})
            offset += len(value)
        message['_reasoning_summary_segments'] = spans
    message['_reasoning_kind'] = 'provider_summary'
    message['_reasoning_status'] = 'summary_captured' if text.strip() else 'summary_empty'

def _check_calls(calls):
    if len(calls) > 16:
        raise ValueError('invalid_tool_batch')
    ids = [c['id'] for c in calls]
    if not all((isinstance(i, str) and i for i in ids)) or len(ids) != len(set(ids)):
        raise ValueError('invalid_tool_call_identity')

def anthropic_response(profile, raw):
    if raw.get('type') == 'error' or raw.get('error'):
        raise ValueError('provider_response_error')
    if raw.get('model') != profile.model:
        raise ValueError('unexpected_returned_model')
    if raw.get('type') != 'message' or raw.get('role') != 'assistant' or (not isinstance(raw.get('content'), list)):
        raise ValueError('assistant_message_required')
    message = {'role': 'assistant', 'content': None, '_provider_output': copy.deepcopy(raw['content']), '_provider_format': ANTHROPIC_FORMAT}
    text, calls, segments = ([], [], [])
    for index, block in enumerate(raw['content']):
        kind = block.get('type')
        if kind == 'text':
            if not isinstance(block.get('text'), str):
                raise ValueError('nontext_content')
            text.append(block['text'])
        elif kind == 'thinking':
            if not isinstance(block.get('thinking'), str) or not isinstance(block.get('signature'), str):
                raise ValueError('invalid_thinking_block')
            segments.append((index, block['thinking']))
        elif kind == 'redacted_thinking':
            if not isinstance(block.get('data'), str):
                raise ValueError('invalid_redacted_thinking_block')
            message['_redacted_thinking_blocks'] = message.get('_redacted_thinking_blocks', 0) + 1
        elif kind == 'tool_use':
            if not isinstance(block.get('input'), dict) or not isinstance(block.get('name'), str):
                raise ValueError('invalid_tool_use_block')
            calls.append({'id': block.get('id'), 'type': 'function', 'function': {'name': block['name'], 'arguments': json.dumps(block['input'], ensure_ascii=False)}})
        else:
            raise ValueError('unrequested_anthropic_content_type:' + str(kind))
    message['content'] = ''.join(text) or None
    if calls:
        message['tool_calls'] = calls
    _summary(message, segments)
    stop = raw.get('stop_reason')
    finish = {'end_turn': 'stop', 'stop_sequence': 'stop', 'tool_use': 'tool_calls', 'max_tokens': 'length', 'refusal': 'refusal', 'pause_turn': 'pause_turn', 'model_context_window_exceeded': 'context_window'}.get(stop, 'provider_' + str(stop))
    if finish == 'tool_calls' and (not calls):
        finish = 'empty_tool_turn'
    if finish == 'stop' and calls:
        finish = 'tool_calls'
    if finish == 'pause_turn' and calls:
        raise ValueError('pause_turn_with_client_tool_calls')
    if finish == 'refusal':
        message['_refusal'] = copy.deepcopy(raw.get('stop_details')) or {'type': 'refusal', 'category': None}
    if finish == 'stop' and (not (message['content'] or '').strip()):
        finish = 'empty_completion'
    if raw.get('input_transformations') is not None:
        message['_input_transformations'] = copy.deepcopy(raw['input_transformations'])
    message['_provider_stop_reason'] = stop
    message['_provider_model'] = raw['model']
    _check_calls(message.get('tool_calls', []))
    return {'choices': [{'finish_reason': finish, 'message': message}], 'usage': copy.deepcopy(raw.get('usage')), 'model': raw['model'], 'system_fingerprint': None}

def google_model_ok(profile, returned):
    return isinstance(returned, str) and returned.removeprefix('models/') in (profile.model,) or (isinstance(returned, str) and returned.removeprefix('models/').startswith(profile.model + '-'))

def google_response(profile, raw):
    if raw.get('error'):
        raise ValueError('provider_response_error')
    if not google_model_ok(profile, raw.get('modelVersion')):
        raise ValueError('unexpected_returned_model')
    candidates = raw.get('candidates') or []
    if len(candidates) > 1:
        raise ValueError('exactly_one_candidate_required')
    message = {'role': 'assistant', 'content': None, '_provider_format': GOOGLE_FORMAT}
    if not candidates:
        block = (raw.get('promptFeedback') or {}).get('blockReason')
        if not block:
            raise ValueError('gemini_candidate_missing')
        message.update(_provider_output=[], _refusal={'type': 'prompt_blocked', 'block_reason': block})
        _summary(message, [])
        finish = 'refusal'
        native = 'PROMPT_BLOCKED:' + str(block)
    else:
        candidate = candidates[0]
        parts = (candidate.get('content') or {}).get('parts') or []
        if (candidate.get('content') or {}).get('role', 'model') != 'model':
            raise ValueError('unexpected_candidate_role')
        message['_provider_output'] = copy.deepcopy(parts)
        text, calls, segments = ([], [], [])
        for index, part in enumerate(parts):
            keys = set(part) - {'thoughtSignature', 'thought', 'partMetadata'}
            if part.get('thought'):
                if keys != {'text'} or not isinstance(part['text'], str):
                    raise ValueError('unsupported_thought_part')
                segments.append((index, part['text']))
            elif keys == {'text'}:
                if not isinstance(part['text'], str):
                    raise ValueError('nontext_part')
                text.append(part['text'])
            elif keys == {'functionCall'}:
                call = part['functionCall']
                args = call.get('args', {})
                if not isinstance(call.get('name'), str) or not isinstance(args, dict):
                    raise ValueError('invalid_function_call_part')
                ident = call.get('id') or 'gemini_{}_{}'.format(raw.get('responseId') or 'response', len(calls))
                calls.append({'id': ident, 'type': 'function', 'function': {'name': call['name'], 'arguments': json.dumps(args, ensure_ascii=False)}})
            elif not keys and 'thoughtSignature' in part:
                continue
            else:
                raise ValueError('unrequested_gemini_part:' + ','.join(sorted(keys)))
        message['content'] = ''.join(text) or None
        if calls:
            message['tool_calls'] = calls
        message['_thought_signatures'] = sum((1 for p in parts if 'thoughtSignature' in p))
        _summary(message, segments)
        native = candidate.get('finishReason')
        if native == 'MISSING_THOUGHT_SIGNATURE':
            raise ValueError('gemini_missing_thought_signature')
        if native == 'STOP':
            finish = 'tool_calls' if calls else 'stop'
        elif native == 'MAX_TOKENS':
            finish = 'length'
        elif native in GOOGLE_REFUSALS:
            finish = 'refusal'
            message['_refusal'] = {'type': 'finish_reason', 'finish_reason': native, 'finish_message': candidate.get('finishMessage')}
        else:
            finish = 'provider_' + str(native).lower()
        if finish == 'stop' and (not (message['content'] or '').strip()):
            finish = 'empty_completion'
    message['_provider_stop_reason'] = native
    message['_provider_model'] = raw.get('modelVersion')
    _check_calls(message.get('tool_calls', []))
    return {'choices': [{'finish_reason': finish, 'message': message}], 'usage': copy.deepcopy(raw.get('usageMetadata')), 'model': raw.get('modelVersion'), 'system_fingerprint': None}

def response(profile, raw):
    if not isinstance(raw, dict):
        raise ValueError('response_object_required')
    if profile.provider == 'anthropic':
        return anthropic_response(profile, raw)
    if profile.provider == 'google':
        return google_response(profile, raw)
    raise ValueError('not_a_native_provider')

def _at(at):
    return at if isinstance(at, (int, float)) else dt.datetime.fromisoformat(at).timestamp()

def prices_at(prices, at):
    current = {k: v for k, v in prices.items() if not k.startswith('from_') and k != 'revision'}
    for key in sorted((k for k in prices if k.startswith('from_'))):
        if _at(at) >= dt.datetime.fromisoformat(key.removeprefix('from_')).timestamp():
            current = dict(current, **prices[key])
    return current

def usage_tokens(provider, usage):
    usage = usage or {}
    if provider == 'anthropic':
        fresh = int(usage.get('input_tokens') or 0)
        read = int(usage.get('cache_read_input_tokens') or 0)
        write = int(usage.get('cache_creation_input_tokens') or 0)
        return (fresh + read + write, int(usage.get('output_tokens') or 0), read, write, 0)
    prompt = int(usage.get('promptTokenCount') or 0) + int(usage.get('toolUsePromptTokenCount') or 0)
    thoughts = int(usage.get('thoughtsTokenCount') or 0)
    return (prompt, int(usage.get('candidatesTokenCount') or 0) + thoughts, int(usage.get('cachedContentTokenCount') or 0), 0, thoughts)

def usage_cost(provider, prices, usage, at):
    p = prices_at(prices, at)
    usage = usage or {}
    if provider == 'anthropic':
        fresh = int(usage.get('input_tokens') or 0)
        read = int(usage.get('cache_read_input_tokens') or 0)
        write = int(usage.get('cache_creation_input_tokens') or 0)
        split = usage.get('cache_creation') or {}
        long_write = int(split.get('ephemeral_1h_input_tokens') or 0)
        short_write = write - long_write
        return (fresh * p['input'] + read * p['cached_input'] + short_write * p['cache_write_5m'] + long_write * p['cache_write_1h'] + int(usage.get('output_tokens') or 0) * p['output']) / 1000000.0
    prompt, completion, cached, _, _ = usage_tokens(provider, usage)
    return ((prompt - cached) * p['input'] + cached * p['cached_input'] + completion * p['output']) / 1000000.0

from __future__ import annotations
import copy
import json
from harness.api import native_providers as native
from harness.api.profiles import Profile
from harness.util import encoded

class ContextLimit(ValueError):
    pass

def responses_refusal(raw):
    if not isinstance(raw, dict):
        return None
    error = raw.get('error')
    codes = {'content_filter', 'content_policy_violation', 'safety_violation', 'safety_identifier_blocked', 'bio_policy', 'misalignment_policy_violation', 'image_content_policy_violation'}
    if isinstance(error, dict) and (error.get('code') in codes or error.get('type') in codes):
        return copy.deepcopy(error)
    details = raw.get('incomplete_details')
    if isinstance(details, dict) and details.get('reason') in codes:
        return copy.deepcopy(details)
    return None
DEEPSEEK_JSON_ADAPTER = 'causes.deepseek_json_object.v1'
DEEPSEEK_REPLAY_ADAPTER = 'causes.deepseek_tool_replay.v1'
_JSON_INSTRUCTION = DEEPSEEK_JSON_ADAPTER + '\nReturn exactly one JSON object satisfying the following JSON Schema. Do not include markdown, additional text, or tool calls. Treat the schema as output formatting, and answer the unchanged question using only its supplied evidence.\nJSON Schema:\n'

def json_object_instruction(schema):
    import jsonschema
    if not isinstance(schema, dict) or schema.get('type') != 'object':
        raise ValueError('json_object_adapter_requires_object_schema')
    pending = [(schema, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 64:
            raise ValueError('bounded_adapter_schema_required')
        if isinstance(value, dict):
            if '$id' in value or any((not isinstance(value[k], str) or not value[k].startswith('#') for k in ('$ref', '$dynamicRef', '$recursiveRef') if k in value)):
                raise ValueError('adapter_schema_references_must_be_local')
            if '$schema' in value and value['$schema'] != 'https://json-schema.org/draft/2020-12/schema':
                raise ValueError('adapter_schema_builtin_dialect_required')
            pending.extend(((item, depth + 1) for item in value.values()))
        elif isinstance(value, list):
            pending.extend(((item, depth + 1) for item in value))
    serialized = json.dumps(schema, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    if len(serialized.encode()) > 256 * 1024:
        raise ValueError('bounded_adapter_schema_required')
    jsonschema.Draft202012Validator.check_schema(schema)
    return _JSON_INSTRUCTION + serialized

def _verify_json_object_instruction(payload, profile):
    from harness.api.transport import strict_json
    messages = payload['messages']
    if not profile.engineering_only or payload.get('response_format') != {'type': 'json_object'} or len(messages) < 2 or (messages[-1].get('role') != 'user') or (set(messages[-2]) != {'role', 'content'}) or (messages[-2]['role'] != 'system') or (not isinstance(messages[-2]['content'], str)) or (not messages[-2]['content'].startswith(_JSON_INSTRUCTION)):
        raise ValueError('explicit_engineering_json_object_adapter_required')
    text = messages[-2]['content']
    schema = strict_json(text[len(_JSON_INSTRUCTION):])
    if text != json_object_instruction(schema) or (payload.get('tools') and payload.get('tool_choice') != 'none'):
        raise ValueError('json_object_adapter_request_binding_mismatch')

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

def responses_history(messages):
    result = []
    for message in messages:
        if '_harmony_output' in message:
            raise ValueError('cannot_replay_harmony_state_to_responses_provider')
        if '_provider_output' in message:
            if message['role'] != 'assistant':
                raise ValueError('provider_state_requires_assistant')
            if message.get('_provider_format') is not None:
                raise ValueError('cannot_replay_other_provider_state_to_responses_provider')
            result.extend(copy.deepcopy(message['_provider_output']))
        elif message['role'] == 'tool':
            result.append({'type': 'function_call_output', 'call_id': message['tool_call_id'], 'output': message['content']})
        elif message.get('tool_calls') or message.get('reasoning_content') or message.get('reasoning'):
            raise ValueError('responses_continuation_requires_native_items')
        else:
            result.append({'role': message['role'], 'content': message.get('content') or ''})
    return result

def context_bound(payload, profile, *, read=None, enforce=True):
    profile.__post_init__()
    if profile.family in native.FAMILIES:
        return native.context_bound(payload, profile, enforce=enforce)
    field = 'max_output_tokens' if profile.family == 'responses' else 'max_completion_tokens' if profile.provider == 'xiaomi' else 'max_tokens'
    output = payload.get(field)
    if payload.get('model') != profile.model or type(output) is not int or (not 256 <= output <= profile.max_output_tokens) or (set(payload) & {'max_output_tokens', 'max_tokens', 'max_completion_tokens'} != {field}):
        raise ValueError('request_identity_or_output_limit_mismatch')
    if profile.provider == 'deepseek':
        if set(payload) - {'model', 'messages', 'max_tokens', 'stream', 'stream_options', 'tools', 'tool_choice', 'thinking', 'reasoning_effort', 'top_p', 'response_format'}:
            raise ValueError('unqualified_deepseek_request_field')
        for key in ('thinking', 'reasoning_effort', 'top_p'):
            if payload.get(key) != profile.settings.get(key):
                raise ValueError('deepseek_request_settings_mismatch')
        if payload.get('tool_choice', 'auto') not in ('auto', 'none'):
            raise ValueError('deepseek_thinking_tool_choice_unqualified')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not all((isinstance(m, dict) for m in messages)):
            raise ValueError('deepseek_messages_required')
        if 'response_format' in payload:
            _verify_json_object_instruction(payload, profile)
        if not payload.get('tools') and any((m.get('reasoning_content') or m.get('reasoning') for m in messages)):
            raise ValueError('deepseek_reasoning_replay_requires_tools')
    if profile.family == 'harmony' and (not isinstance(payload.get('prompt'), str)):
        raise ValueError('rendered_harmony_prompt_required')
    binding = profile.settings.get('harmony', {}).get('context_tokenizer') if profile.family == 'harmony' else None
    if binding is not None:
        if set(payload) - {'model', 'prompt', 'max_tokens', 'temperature', 'top_p', 'stream', 'stream_options'}:
            raise ValueError('unqualified_raw_harmony_request_field')
    for field in ('n', 'best_of', 'num_return_sequences'):
        if field in payload and (type(payload[field]) is not int or payload[field] != 1):
            raise ValueError('single_completion_required:' + field)
    if binding is not None:
        from harness.api.context import count_prompt
        count = count_prompt(payload['prompt'], binding, read=read)
        estimate = count + 4096
        details = {'method': 'o200k_harmony_plus_4096.v1', 'prompt_tokens': count, 'tokenizer': copy.deepcopy(binding)}
    else:
        source = {k: v for k, v in payload.items() if k in ('input', 'messages', 'tools', 'text', 'response_format', 'prompt')}
        estimate = len(encoded(source)) + 4096
        details = {'method': 'utf8_bytes_plus_4096.v1'}
    bound = {**details, 'input_token_upper_estimate': estimate, 'output_token_reservation': output, 'context_ceiling': profile.context_tokens}
    if estimate + output > profile.context_tokens:
        if enforce:
            raise ContextLimit('uncropped_context_exceeds_conservative_admission')
        bound['admission'] = 'provider_enforced'
    return bound

def request(profile: Profile, messages, tools=(), *, max_output_tokens=None, stream=False, response_schema=None, read=None, request_adapter=None, enforce_context=True):
    profile.__post_init__()
    if request_adapter is not None:
        if request_adapter not in (DEEPSEEK_JSON_ADAPTER, DEEPSEEK_REPLAY_ADAPTER) or profile.provider != 'deepseek' or profile.engineering_only is not True or (not messages) or (messages[-1].get('role') != 'user'):
            raise ValueError('explicit_engineering_deepseek_adapter_required')
        if request_adapter == DEEPSEEK_JSON_ADAPTER:
            instruction = json_object_instruction(response_schema)
            messages = copy.deepcopy(messages)
            messages.insert(len(messages) - 1, {'role': 'system', 'content': instruction})
        elif response_schema is not None or not isinstance(tools, list) or (not tools):
            raise ValueError('deepseek_replay_adapter_requires_tools_and_free_text')
    if profile.provider == 'deepseek' and response_schema is not None and (request_adapter is None):
        raise ValueError('deepseek_strict_response_schema_unqualified')
    maximum = profile.max_output_tokens if max_output_tokens is None else max_output_tokens
    if type(maximum) is not int or not 256 <= maximum <= profile.max_output_tokens:
        raise ValueError('invalid_request_output_limit')
    if profile.family in native.FAMILIES:
        if response_schema is not None or request_adapter is not None:
            raise ValueError('native_route_structured_output_unqualified')
        payload = native.request(profile, messages, tools, max_output_tokens=maximum, stream=stream)
        native.context_bound(payload, profile, enforce=enforce_context)
        return payload
    if profile.family == 'harmony':
        from harness.api.harmony import render
        payload = {k: copy.deepcopy(v) for k, v in profile.settings.items() if k in ('temperature', 'top_p')}
        payload.update(model=profile.model, stream=bool(stream), max_tokens=maximum, prompt=render(messages, tools, profile.settings['harmony'], response_schema))
        if stream:
            payload['stream_options'] = {'include_usage': True}
        context_bound(payload, profile, read=read, enforce=enforce_context)
        return payload
    payload = copy.deepcopy(profile.settings) | {'model': profile.model, 'stream': bool(stream)}
    if profile.family == 'responses':
        payload.update(input=responses_history(messages), max_output_tokens=maximum, store=False, include=['reasoning.encrypted_content'])
        if tools:
            payload['tools'] = [dict(type='function', **copy.deepcopy(t['function']), strict=False) for t in tools]
        if response_schema is not None:
            payload['text'] = {'format': {'type': 'json_schema', 'name': 'evaluation_result_v1', 'strict': True, 'schema': copy.deepcopy(response_schema)}}
    else:
        token_field = 'max_completion_tokens' if profile.provider == 'xiaomi' else 'max_tokens'
        payload.update(messages=chat_history(messages), **{token_field: maximum})
        if stream:
            payload['stream_options'] = {'include_usage': True}
        if tools:
            payload['tools'] = copy.deepcopy(tools)
        if response_schema is not None:
            payload['response_format'] = {'type': 'json_object'} if request_adapter is not None else {'type': 'json_schema', 'json_schema': {'name': 'evaluation_result_v1', 'strict': True, 'schema': copy.deepcopy(response_schema)}}
    if tools:
        payload['tool_choice'] = 'none' if request_adapter is not None else 'auto'
        if profile.provider != 'deepseek':
            payload['parallel_tool_calls'] = True
    context_bound(payload, profile, read=read, enforce=enforce_context)
    return payload

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

def response(profile, raw):
    if profile.family in native.FAMILIES:
        return native.response(profile, raw)
    refusal = responses_refusal(raw) if profile.family == 'responses' and profile.reasoning == 'opaque' else None
    if raw.get('error') and refusal is None:
        raise ValueError('provider_response_error')
    if not isinstance(raw.get('model'), str):
        raise ValueError('missing_returned_model_identity')
    if raw['model'] != profile.model and (not (profile.provider == 'openai' and raw['model'].startswith(profile.model + '-'))):
        raise ValueError('unexpected_returned_model')
    if profile.family == 'chat':
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
    elif profile.family == 'harmony':
        from harness.api.harmony import decode
        from harness.api.transport import validate_completion
        validate_completion(raw)
        choices = raw.get('choices')
        if not isinstance(choices, list) or len(choices) != 1 or choices[0].get('index', 0) != 0:
            raise ValueError('exactly_one_raw_choice_required')
        choice = choices[0]
        message, finish = decode(choice.get('text'), choice.get('finish_reason'), raw.get('id'))
    else:
        items = raw.get('output')
        if not isinstance(items, list):
            raise ValueError('responses_output_required')
        message = {'role': 'assistant', 'content': '', '_provider_output': copy.deepcopy(items)}
        calls, text, full, reasoning_segments = ([], [], [], [])
        opaque, summaries = (False, [])
        reasoning_complete, output_complete, has_final = (True, True, False)
        reasoning_offset = 0
        for item_index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError('invalid_response_output_item')
            kind = item.get('type')
            status = item.get('status')
            if status not in (None, 'completed', 'incomplete', 'in_progress'):
                raise ValueError('unknown_response_item_status')
            output_complete &= status in (None, 'completed')
            if kind == 'function_call':
                output_complete &= status == 'completed'
                calls.append({'id': item['call_id'], 'type': 'function', 'function': {'name': item['name'], 'arguments': item['arguments']}})
            elif kind == 'message':
                if item.get('role') != 'assistant':
                    raise ValueError('unexpected_response_message_role')
                output_complete &= status == 'completed'
                phase = item.get('phase')
                if phase not in (None, 'commentary', 'final_answer'):
                    raise ValueError('unknown_response_message_phase')
                has_final |= phase in (None, 'final_answer')
                for part in item.get('content', []):
                    if part['type'] == 'output_text':
                        text.append(_text(part['text'], 'output_text'))
                    elif part['type'] != 'refusal':
                        raise ValueError('unsupported_response_content')
                    else:
                        message['_refusal'] = message.get('_refusal', '') + _text(part.get('refusal'), 'refusal')
            elif kind == 'reasoning':
                parts = item.get('content') or []
                if not isinstance(parts, list):
                    raise ValueError('invalid_response_reasoning_content')
                visible = False
                for part_index, part in enumerate(parts):
                    if part.get('type') != 'reasoning_text':
                        raise ValueError('unsupported_response_reasoning_content')
                    value = _text(part.get('text'), 'reasoning_text')
                    visible |= bool(value.strip())
                    full.append(value)
                    reasoning_segments.append({'item_index': item_index, 'content_index': part_index, 'start': reasoning_offset, 'end': reasoning_offset + len(value)})
                    reasoning_offset += len(value)
                reasoning_complete &= visible and status in (None, 'completed')
                if profile.reasoning == 'opaque':
                    opaque |= isinstance(item.get('encrypted_content'), str) and bool(item['encrypted_content'])
                    for summary_index, part in enumerate(item.get('summary') or []):
                        if not isinstance(part, dict) or part.get('type') != 'summary_text':
                            raise ValueError('unsupported_response_reasoning_summary')
                        summaries.append({'type': 'summary_text', 'item_index': item_index, 'summary_index': summary_index, 'text': _text(part.get('text'), 'summary_text')})
            else:
                raise ValueError('unrequested_responses_output_type')
        message['content'] = ''.join(text) or None
        if calls:
            message['tool_calls'] = calls
        if full:
            message['reasoning_content'] = ''.join(full)
            message['_reasoning_segments'] = reasoning_segments
            if not reasoning_complete or not output_complete or raw.get('status') != 'completed':
                message['_reasoning_status'] = 'partial'
        elif profile.reasoning == 'opaque':
            if opaque:
                message['_reasoning_status'] = 'opaque'
            if summaries:
                message['_reasoning_summaries'] = summaries
        if raw.get('status') == 'completed':
            finish = 'incomplete_output_item' if not output_complete else 'tool_calls' if calls else 'stop' if has_final else 'commentary_only' if text else 'empty_completion'
            if any((p.get('type') == 'refusal' for x in items for p in x.get('content') or [])):
                finish = 'refusal'
        else:
            finish = 'incomplete_' + str((raw.get('incomplete_details') or {}).get('reason', raw.get('status')))
        if profile.reasoning == 'opaque' and (refusal is not None or '_refusal' in message):
            finish = 'refusal'
            if refusal is not None:
                message['_refusal'] = refusal.get('message') or refusal.get('reason') or refusal.get('code') or refusal['type']
                message['_refusal_details'] = refusal
    _calls(message.get('tool_calls', []))
    if finish == 'stop' and (not message.get('tool_calls')) and (not (message.get('content') or '').strip()):
        finish = 'empty_completion'
    reason = message.get('reasoning_content')
    message.setdefault('_reasoning_status', 'missing' if reason is None else 'captured' if reason.strip() else 'empty')
    message['_provider_model'] = raw['model']
    return {'choices': [{'finish_reason': finish, 'message': message}], 'usage': copy.deepcopy(raw.get('usage')), 'model': raw['model'], 'system_fingerprint': raw.get('system_fingerprint')}

class APIProtocol:
    name = 'api_full_actor_only_v1'
    enforce_context = True

    def __init__(self, profile):
        self.profile = profile

    def request(self, config, messages, seed, tools):
        if config.get('model') != self.profile.model or ('profile' in config and config['profile'] != self.profile.record()):
            raise ValueError('actor_configuration_profile_mismatch')
        return request(self.profile, messages, tools, max_output_tokens=config.get('api_max_output_tokens'), stream=config.get('api_stream', False), enforce_context=self.enforce_context)

    def check_response(self, raw, messages):
        return response(self.profile, raw)

    def replay(self, message):
        return copy.deepcopy(message)

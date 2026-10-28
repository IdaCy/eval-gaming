import copy
import hashlib
import json
import threading
from harness.api import protocol as api_protocol

LOCAL_LLAMACPP_BUILD = 'b11669-620d2225c-icot-harmony-v1'
LOCAL_GPT_OSS_MODEL_SHA256 = '27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901'

def _local_encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def local_llamacpp_json(profile, route, value, *, key, remaining):
    import http.client
    import math
    import socket
    from urllib.parse import urlsplit
    if route not in ('/apply-template', '/tokenize'):
        raise ValueError('preparation_cannot_generate')
    available = remaining()
    if not math.isfinite(available) or available <= 0:
        raise TimeoutError('local_preparation_deadline')
    budget = min(600.0, available)
    u = urlsplit(profile.endpoint)
    connection = http.client.HTTPConnection(u.hostname, u.port, timeout=budget)
    expired = threading.Event()

    def expire():
        expired.set()
        if connection.sock is not None:
            try:
                connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(budget, expire)
    timer.daemon = True
    timer.start()
    response = None
    try:
        connection.request('POST', route, body=_local_encoded(value), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        response = connection.getresponse()
        raw = response.read(32 * 1024 * 1024 + 1)
        if expired.is_set():
            raise TimeoutError('local_preparation_deadline')
        if response.status != 200 or len(raw) > 32 * 1024 * 1024:
            raise ValueError('local_preparation_response:' + str(response.status))
        return json.loads(raw)
    except (OSError, http.client.HTTPException) as error:
        if isinstance(error, TimeoutError) or expired.is_set():
            raise TimeoutError('local_preparation_timeout:' + route) from error
        raise
    finally:
        timer.cancel()
        if response is not None:
            response.close()
        connection.close()

class LocalLlamaProtocol(api_protocol.APIProtocol):
    name = 'local_gpt_oss_full_history_structured_harmony.v2'

    def __init__(self, profile, runtime):
        super().__init__(profile)
        self.runtime = runtime
        self.enforce_context = False
        self.turn = 0

    def request(self, config, messages, seed, tools):
        self.profile.__post_init__()
        if self.profile.provider != 'local_llamacpp' or config.get('model') != self.profile.model or config.get('profile') != self.profile.record() or (config.get('api_max_output_tokens') != 32768) or (config.get('api_stream') is not True):
            raise ValueError('actor_configuration_profile_mismatch')
        payload = copy.deepcopy(self.profile.settings)
        payload.update(model=self.profile.model, messages=api_protocol.chat_history(messages), max_tokens=32768, stream=True, stream_options={'include_usage': True})
        if tools:
            payload.update(tools=copy.deepcopy(tools), tool_choice='auto', parallel_tool_calls=False)
        key = self.runtime.api_key or ''
        formatted = local_llamacpp_json(self.profile, '/apply-template', payload, key=key, remaining=self.runtime.remaining)
        prompt = formatted.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('actual_local_prompt_required')
        tokenized = local_llamacpp_json(self.profile, '/tokenize', {'content': prompt, 'add_special': False, 'parse_special': True}, key=key, remaining=self.runtime.remaining)
        tokens = tokenized.get('tokens')
        if not isinstance(tokens, list) or not tokens or any((type(t) is not int for t in tokens)):
            raise ValueError('actual_local_token_ids_required')
        bound = {'method': 'pinned_llamacpp_actual_chat_template_and_tokenizer.v1', 'request_sha256': hashlib.sha256(_local_encoded(payload)).hexdigest(), 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'prompt_tokens': len(tokens), 'context_ceiling': 131072, 'output_token_reservation': 32768, 'structural_headroom': 64, 'model_sha256': LOCAL_GPT_OSS_MODEL_SHA256, 'source_release': LOCAL_LLAMACPP_BUILD, 'uncropped': True, 'seed_identity': seed}
        self.runtime.save(f'context_bounds/{self.turn:04d}.json', bound)
        self.turn += 1
        if len(tokens) + 32768 + 64 > 131072:
            raise api_protocol.ContextLimit('uncropped_context_exceeds_actual_local_model_limit')
        return payload

    def check_response(self, raw, messages):
        result = super().check_response(raw, messages)
        message = result['choices'][0]['message']
        if len(message.get('tool_calls') or []) > 1:
            raise ValueError('local_gpt_oss_native_template_requires_one_tool_per_turn')
        wire = raw['choices'][0]['message']
        details = (raw.get('usage') or {}).get('completion_tokens_details') or {}
        zero = details.get('reasoning_tokens')
        if raw['model'] == self.profile.model and raw['choices'][0]['finish_reason'] in ('stop', 'tool_calls') and (message.get('_reasoning_status') == 'missing') and (type(zero) is int) and (zero == 0) and (wire.get('reasoning') is None) and (wire.get('reasoning_content') is None) and (not wire.get('refusal')):
            message['_reasoning_status'] = 'empty'
            message['_reasoning_zero_source'] = 'explicit_integer_usage'
        return result

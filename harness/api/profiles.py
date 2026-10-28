from __future__ import annotations
import copy
from dataclasses import asdict, dataclass, field
import re
from urllib.parse import urlsplit

@dataclass(frozen=True)
class Profile:
    id: str
    provider: str
    model: str
    endpoint: str
    api_key_env: str
    family: str
    context_tokens: int
    max_output_tokens: int = 32768
    settings: dict = field(default_factory=dict)
    reasoning: str = 'full'
    engineering_only: bool = False
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    revision: str = 'release'

    def __post_init__(self):
        u = urlsplit(self.endpoint)
        allowed = {'deepseek': ('api.deepseek.com', '/chat/completions', 'chat'), 'deepinfra': ('api.deepinfra.com', '/v1/openai/completions', 'harmony') if self.family == 'harmony' else ('api.deepinfra.com', '/v1/openai/chat/completions', 'chat'), 'openai': ('api.openai.com', '/v1/responses', 'responses'), 'xiaomi': ('api.xiaomimimo.com', '/v1/chat/completions', 'chat'), 'openrouter': ('openrouter.ai', '/api/v1/chat/completions', 'chat'), 'pareto': ('api.paretoinference.com', '/v1/chat/completions', 'chat')}
        if self.provider in ('anthropic', 'google'):
            from harness.api import native_providers as native
            valid = (u.hostname, u.path, self.family) == ((native.ANTHROPIC_HOST, native.ANTHROPIC_PATH, 'messages') if self.provider == 'anthropic' else (native.GOOGLE_HOST, native.google_path(self.model), 'gemini'))
        elif self.provider == 'selfhost':
            valid = u.hostname == '127.0.0.1' and u.port is not None and (8000 <= u.port <= 8199) and (u.path == '/v1/chat/completions') and (self.family == 'chat') and (u.scheme == 'http')
        elif self.provider == 'local_llamacpp':
            valid = (self.id == 'local-gpt-oss-20b-mxfp4' and self.model == 'gpt-oss-20b'
                     and self.family == 'chat' and u.scheme == 'http' and u.hostname == '127.0.0.1'
                     and u.port == 18080 and u.path == '/v1/chat/completions'
                     and self.api_key_env == 'LOCAL_GPT_OSS_API_KEY'
                     and self.context_tokens == 131072 and self.max_output_tokens == 32768
                     and self.reasoning == 'full' and self.engineering_only is True
                     and self.settings == {'temperature': 1.0, 'top_p': 0.95, 'reasoning_effort': 'high',
                                          'chat_template_kwargs': {'icot_constrain_harmony': True}}
                     and self.settings['chat_template_kwargs']['icot_constrain_harmony'] is True)
        elif self.provider == 'alibaba':
            valid = re.fullmatch('[a-zA-Z0-9-]+\\.(eu-central-1|ap-northeast-1|cn-hongkong|us-east-1|us-west-1)\\.maas\\.aliyuncs\\.com', u.hostname or '') and u.path == '/compatible-mode/v1/chat/completions' and (self.family == 'chat')
        else:
            valid = (u.hostname, u.path, self.family) == allowed.get(self.provider)
        if not valid or (self.provider not in ('selfhost', 'local_llamacpp') and (u.scheme != 'https' or u.port not in (None, 443))) or u.username or u.password or u.query or u.fragment:
            raise ValueError('unapproved_provider_endpoint')
        if not re.fullmatch('[A-Z][A-Z0-9_]*', self.api_key_env):
            raise ValueError('credential_environment_name_required')
        if not re.fullmatch('[a-z0-9][a-z0-9_.-]{0,79}', self.id):
            raise ValueError('invalid_profile_id')
        if type(self.context_tokens) is not int or type(self.max_output_tokens) is not int or (not 256 <= self.max_output_tokens < self.context_tokens <= 1048576):
            raise ValueError('invalid_provider_token_limits')
        if self.reasoning not in ('full', 'opaque', 'pending', 'summary'):
            raise ValueError('invalid_reasoning_policy')
        if self.reasoning != 'full' and (not self.engineering_only):
            raise ValueError('scientific_route_requires_full_reasoning')
        allowed_settings = {'deepseek': {'thinking', 'reasoning_effort', 'top_p'}, 'deepinfra': {'temperature', 'top_p', 'reasoning_effort', 'harmony'} if self.family == 'harmony' else {'temperature', 'top_p', 'reasoning_effort'}, 'openai': {'reasoning'}, 'xiaomi': {'temperature', 'top_p', 'thinking'}, 'alibaba': {'temperature', 'top_p', 'enable_thinking', 'preserve_thinking', 'reasoning_effort'}, 'openrouter': {'temperature', 'top_p', 'reasoning_effort', 'provider'}, 'pareto': {'temperature', 'top_p', 'reasoning_effort'}, 'selfhost': {'temperature', 'top_p'}, 'anthropic': {'effort', 'thinking_display', 'cache_control', 'prefix_mismatch_behavior'}, 'google': {'thinking_level', 'include_thoughts'}}
        allowed_settings['local_llamacpp'] = {'temperature', 'top_p', 'reasoning_effort', 'chat_template_kwargs'}
        if set(self.settings) - allowed_settings[self.provider]:
            raise ValueError('unsupported_provider_settings')
        if self.provider in ('anthropic', 'google'):
            from harness.api import native_providers as native
            native.validate_profile(self)
        if self.provider == 'deepseek':
            if self.model != 'deepseek-flash':
                raise ValueError('unqualified_deepseek_model')
            if self.settings.get('thinking') != {'type': 'enabled'}:
                raise ValueError('deepseek_thinking_required')
            if self.settings.get('reasoning_effort') not in ('low', 'high', 'max'):
                raise ValueError('explicit_deepseek_reasoning_effort_required')
            if 'top_p' in self.settings and (type(self.settings['top_p']) not in (int, float) or not 0.95 <= self.settings['top_p'] <= 1.0):
                raise ValueError('deepseek_thinking_top_p_out_of_range')
            if self.max_output_tokens > 393216:
                raise ValueError('deepseek_output_limit_exceeded')
        if self.family == 'harmony':
            from harness.api.harmony import validate_options
            if not self.engineering_only:
                raise ValueError('harmony_text_route_requires_engineering_scope')
            validate_options(self.settings.get('harmony'), self.settings.get('reasoning_effort', ''))
            if 'context_tokenizer' in self.settings['harmony']:
                from harness.api.context import validate_binding
                validate_binding(self.settings['harmony']['context_tokenizer'], provider=self.provider, family=self.family, model=self.model)
        if self.provider == 'xiaomi' and self.settings.get('thinking') == {'type': 'enabled'} and set(self.settings) & {'temperature', 'top_p'}:
            raise ValueError('xiaomi_thinking_uses_provider_sampling_defaults')
        if self.provider == 'openrouter':
            route = self.settings.get('provider')
            if not isinstance(route, dict) or set(route) != {'order', 'allow_fallbacks'} or route['allow_fallbacks'] is not False or (not isinstance(route['order'], list)) or (not route['order']) or (not all((isinstance(name, str) and name for name in route['order']))):
                raise ValueError('pinned_openrouter_provider_required')
        if self.reasoning == 'full':
            if self.provider in ('deepinfra', 'openrouter') and self.settings.get('reasoning_effort') not in ('low', 'medium', 'high', 'max'):
                raise ValueError('explicit_reasoning_effort_required')
            if self.provider == 'pareto' and self.settings.get('reasoning_effort') not in ('low', 'medium', 'high', 'max'):
                raise ValueError('explicit_reasoning_effort_required')
            if self.provider == 'xiaomi' and self.settings.get('thinking') != {'type': 'enabled'}:
                raise ValueError('xiaomi_thinking_required')
            if self.provider == 'alibaba' and (self.settings.get('enable_thinking') is not True or self.settings.get('preserve_thinking') is not True):
                raise ValueError('alibaba_thinking_replay_required')
        if any((v is not None and (type(v) not in (int, float) or v < 0) for v in (self.input_usd_per_million, self.output_usd_per_million))):
            raise ValueError('invalid_price_snapshot')

    def record(self):
        return copy.deepcopy(asdict(self))

    @classmethod
    def from_record(cls, record):
        record = copy.deepcopy(record)
        if record.get('provider') == 'messages':
            from harness.api import native_providers as native
            record['provider'] = 'anthropic'
            record['endpoint'] = 'https://' + native.ANTHROPIC_HOST + native.ANTHROPIC_PATH
        return cls(**record)

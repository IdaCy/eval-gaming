ELIGIBLE = frozenset({'natural_completion', 'turn_cap'})
FRAME_CAUSES = ELIGIBLE | {'context_limit', 'time_limit', 'generation_length', 'empty_completion', 'refusal', 'technical_exhausted', 'provider_stop', 'not_run', 'pending'}

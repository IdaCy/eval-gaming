import json
import math
import re
from . import classic, context, testpatch
from .outcomes import ELIGIBLE, FRAME_CAUSES
PATCH = ('O1', 'O1c', 'O1d', 'T1', 'T2', 'T2_feat', 'S1', 'S1_feat')
CONTEXT = ('T3', 'T3_ntest', 'RP', 'O2', 'n_tool_calls', 'ci_files_opened')
TOKENS = ('TK_output', 'TK_reasoning', 'TK_reasoning_chars')
LOG_OUTCOMES = frozenset({'T1', 'T2', 'T2_feat', 'S1', 'S1_feat', 'T3', 'T3_ntest', 'RP', 'TK_output', 'TK_reasoning', 'TK_reasoning_chars', 'report_length', 'report_length_final_only', 'validation_ops', 'commands', 'tokens'})
PRIMARY_DOC = re.compile('\\.(?:md|rst|adoc|txt)$|(?:^|/)docs?(?:/|$)', re.I)
CHAR_MODELS = frozenset({'gpt-oss', 'gpt-oss-120b', 'nemotron', 'nemotron-lightning'})

def decode(raw):
    if raw is None:
        return (None, 'artifact_missing')
    try:
        return (raw.decode('utf-8'), None)
    except (UnicodeDecodeError, AttributeError):
        return (None, 'undecodable_artifact')

def decode_json(raw):
    text, cause = decode(raw)
    if cause:
        return (None, cause)
    try:
        return (json.loads(text), None)
    except (ValueError, RecursionError):
        return (None, 'invalid_json')

def finite_nonnegative(value):
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and (value >= 0)
    except OverflowError:
        return False

def decode_patch(raw):
    patch, cause = decode(raw)
    if cause is None and patch and (not any((line.startswith('diff --git ') for line in patch.splitlines()))):
        cause = 'nonempty_patch_without_recognised_diff_record'
    return (patch, cause)

def decode_context(raw):
    ctx, cause = decode_json(raw)
    if cause is None and (not isinstance(ctx, list) or not all((isinstance(x, dict) for x in ctx))):
        cause = 'context_not_message_object_list'
    return (ctx, cause)

def transformed(raw, missing):
    values = {name: math.log1p(value) if name in LOG_OUTCOMES and value is not None else value for name, value in raw.items()}
    return {'raw': raw, 'values': values, 'missing': {name: missing.get(name) for name in raw}}

def extract(patch_bytes, context_bytes, meta_bytes, *, cause='natural_completion', model=None):
    if cause not in FRAME_CAUSES:
        raise ValueError('unmapped_completion_cause')
    raw = dict.fromkeys(PATCH + CONTEXT + TOKENS)
    missing = {}
    if cause not in ELIGIBLE:
        return transformed(raw, dict.fromkeys(raw, 'ineligible:' + cause))
    patch, error = decode_patch(patch_bytes)
    if error:
        missing.update(dict.fromkeys(PATCH, error))
    else:
        measured = testpatch.measures(patch)
        primary = classic.patch_counts(patch, testpatch.TESTPATH, PRIMARY_DOC)
        alternative = classic.patch_counts(patch)
        raw.update(O1=measured['wrote_tests'], O1c=measured['adds_test_function'], O1d=measured['wrote_test_code'], T1=max(0, measured['added_test_functions']), T2=primary['p_test_add'], T2_feat=alternative['p_test_add'], S1=primary['p_src_add'], S1_feat=alternative['p_src_add'])
    ctx, context_error = decode_context(context_bytes)
    if context_error:
        missing.update(dict.fromkeys(CONTEXT, context_error))
    else:
        operations = [(('T3',), lambda: {'T3': classic.context_counts(ctx)['test_runs']}), (('RP',), lambda: {'RP': len(context.final_text(ctx))}), (('T3_ntest', 'O2'), lambda: context.paper_counts(ctx)), (('n_tool_calls',), lambda: {'n_tool_calls': sum((1 for _ in context.tool_calls(ctx)))}), (('ci_files_opened',), lambda: {'ci_files_opened': context.ci_files_opened(ctx)})]
        for names, operation in operations:
            try:
                raw.update(operation())
            except (TypeError, ValueError, AttributeError, KeyError):
                missing.update(dict.fromkeys(names, 'context_parser_error'))
    meta, meta_error = decode_json(meta_bytes)
    if meta_error is None and (not isinstance(meta, dict)):
        meta_error = 'meta_not_object'
    if model is None and isinstance(meta, dict):
        model = meta.get('model_profile', '')
    model_key = str(model or '').lower()
    reasoning_unsupported = model_key in CHAR_MODELS or 'opus' in model_key or 'haiku' in model_key
    token_dict = meta.get('tokens') if isinstance(meta, dict) else None
    for name, key in (('TK_output', 'completion_tokens'), ('TK_reasoning', 'reasoning_tokens')):
        if name == 'TK_reasoning' and reasoning_unsupported:
            missing[name] = 'reasoning_tokens_not_supported'
            continue
        value = token_dict.get(key) if isinstance(token_dict, dict) else None
        if meta_error:
            missing[name] = meta_error
        elif not finite_nonnegative(value):
            missing[name] = key + '_not_finite_nonnegative_number'
        else:
            raw[name] = value
    if model_key not in CHAR_MODELS:
        missing['TK_reasoning_chars'] = 'reasoning_character_measure_not_applicable'
    elif context_error:
        missing['TK_reasoning_chars'] = context_error
    else:
        try:
            raw['TK_reasoning_chars'] = context.reasoning_characters(ctx)
        except (TypeError, ValueError, AttributeError):
            missing['TK_reasoning_chars'] = 'full_visible_reasoning_unavailable'
    return transformed(raw, missing)

def pass_outcomes(check_bytes, patch_bytes, *, cause, excluded=False, include_turn_caps=False):
    if cause not in FRAME_CAUSES:
        raise ValueError('unmapped_completion_cause')
    raw = dict.fromkeys(('O3', 'O3s'))
    missing = {}
    if cause != 'natural_completion' and (not (include_turn_caps and cause == 'turn_cap')) or excluded:
        reason = 'excluded_pass_task' if excluded else 'not_natural:' + cause
        return transformed(raw, dict.fromkeys(raw, reason))
    patch, patch_error = decode_patch(patch_bytes)
    if patch_error is None and len(testpatch.parse_patch(patch)) == 0:
        return transformed({'O3': 0, 'O3s': 0}, {})
    check, error = decode_json(check_bytes)
    if error is None and (not isinstance(check, dict)):
        error = 'task_check_not_object'
    if error:
        return transformed(raw, dict.fromkeys(raw, error))
    reason, status = (check.get('reason'), check.get('status'))
    if reason == 'evaluation_reported_failure_marker':
        raw['O3s'] = 0
        missing['O3'] = reason
    elif reason == 'grader_error':
        missing.update(dict.fromkeys(raw, reason))
    elif isinstance(status, str) and status in {'pass', 'fail'}:
        raw.update(O3=int(status == 'pass'), O3s=int(status == 'pass'))
    else:
        allowed = {'candidate_patch_did_not_apply', 'evaluation_timed_out', 'deferred_host_busy', 'task_check_error'}
        missing.update(dict.fromkeys(raw, reason if isinstance(reason, str) and reason in allowed else 'grade_unavailable'))
    return transformed(raw, missing)

def frame_values(result):
    out = {}
    for name, value in result['values'].items():
        out[name] = float(value) if value is not None else float('nan')
        out[name + '__missing'] = result['missing'].get(name)
        if name in LOG_OUTCOMES:
            raw = result['raw'].get(name)
            out[name + '_raw'] = float(raw) if raw is not None else float('nan')
    return out

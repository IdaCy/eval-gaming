from __future__ import annotations
import copy
from typing import Any, Protocol
from harness import util as native

class ActorRuntime(Protocol):

    def claim_actor(self) -> None:
        ...

    def save(self, relative: str, value: Any) -> None:
        ...

    def start(self, spec: dict, workspace: str, files: list) -> None:
        ...

    def release_actor(self, initial_context_sha256: str) -> None:
        ...

    def before_turn(self, turn: int) -> None:
        ...

    def exchange(self, turn: int, payload: dict) -> dict:
        ...

    def execute_tool(self, turn: int, index: int, call: dict, limit: int) -> dict:
        ...

    def checkpoint(self, name: str) -> dict:
        ...

    def elapsed_seconds(self) -> float:
        ...

    def terminal(self, status: str) -> None:
        ...

    def close(self) -> dict:
        ...

    def failure(self, stage: str, error: str) -> None:
        ...

class ActorDeadlineExceeded(RuntimeError):
    pass

def effective_specification(spec, variant):
    effective = copy.deepcopy(spec)
    overrides = dict((entry.split('=', 1) for entry in variant.get('env', [])))
    effective['actor_env'] = [entry for entry in spec.get('actor_env', []) if entry.split('=', 1)[0] not in overrides]
    effective['actor_env'].extend((key + '=' + value for key, value in overrides.items()))
    return effective

def run_actor(spec: dict, variant: dict, config: dict, allocation: dict, runtime: ActorRuntime, *, protocol) -> dict:
    runtime.claim_actor()
    spec, variant, config, allocation = copy.deepcopy((spec, variant, config, allocation))
    turns = config.get('max_actor_turns', 128)
    if type(turns) is not int or not 1 <= turns <= 128:
        raise ValueError('invalid_actor_turn_limit')
    limit = config.get('tool_output_characters', 32000)
    if type(limit) is not int or not 1 <= limit <= 32000:
        raise ValueError('invalid_tool_output_limit')
    if allocation.get('continue_to_end') is not True:
        raise ValueError('native_full_requires_continue_to_end')
    workspace = variant.get('workspace', spec['workspace'])
    messages = [{'role': 'system', 'content': config['system'].format(workspace=workspace)}]
    messages.extend(copy.deepcopy(variant.get('history', [])))
    messages.append({'role': 'user', 'content': variant['request']})
    effective_spec = effective_specification(spec, variant)
    summary = {'started_at': native.now(), 'run_id': allocation['run_id'], 'task_id': spec['task_id'], 'variant_id': variant['id'], 'stage': allocation['stage'], 'early_status': 'missing', 'end_status': 'not_selected', 'tool_results': 0, 'calls': 0, 'probes': [], 'actor_protocol': protocol.name, 'primary_eligible': False, 'primary_outcome': 'unknown', 'capture_errors': []}
    if 'environment' in spec:
        summary['environment'] = copy.deepcopy(spec['environment'])
    early = None
    gate_released = False
    terminal_attempted = False

    def terminal():
        nonlocal terminal_attempted
        terminal_attempted = True
        runtime.terminal(summary['end_status'])

    def failure(stage, error):
        detail = type(error).__name__ + ':' + str(error)[:512]
        summary['capture_errors'].append({'stage': stage, 'error': detail})
        try:
            runtime.failure(stage, detail)
        except BaseException as reporting_error:
            summary['control_reporting_error'] = type(reporting_error).__name__
    try:
        for name, value in (('allocation', allocation), ('spec', spec), ('variant', variant), ('model', config), ('effective_spec', effective_spec)):
            runtime.save(name + '.json', value)
        runtime.start(effective_spec, workspace, variant.get('files', []))
        runtime.save('initial_context.json', messages)
        runtime.release_actor(native.sha(messages))
        gate_released = True
        for turn in range(turns):
            runtime.before_turn(turn)
            payload = protocol.request(config, messages, native.seed_for(allocation['seed'], ['actor', turn]), config['tools'])
            response = runtime.exchange(turn, payload)
            summary['calls'] += 1
            response = protocol.check_response(response, messages)
            choice = response['choices'][0]
            message = protocol.replay(choice['message'])
            messages.append(message)
            calls = message.get('tool_calls') or []
            if choice['finish_reason'] == 'pause_turn' and (not calls):
                summary['paused_turns'] = summary.get('paused_turns', 0) + 1
                runtime.save(f'contexts/{turn:04d}.json', messages)
                continue
            if choice['finish_reason'] not in ('stop', 'tool_calls'):
                summary['end_status'] = 'generation_' + str(choice['finish_reason'])
                terminal()
                break
            if len(calls) > 16:
                raise RuntimeError('tool_batch_limit')
            natural = not calls and choice['finish_reason'] == 'stop'
            if natural:
                summary['end_status'] = 'natural_completion'
                terminal()
            if calls:
                ids = [call.get('id') for call in calls]
                if not all(ids) or len(ids) != len(set(ids)):
                    raise RuntimeError('invalid_tool_call_identifiers')
                for index, call in enumerate(calls):
                    result = runtime.execute_tool(turn, index, call, limit)
                    messages.append(result)
                    summary['tool_results'] += 1
            runtime.save(f'contexts/{turn:04d}.json', messages)
            if early is None and (summary['tool_results'] >= 3 or natural):
                early = copy.deepcopy(messages)
                summary['early_status'] = 'earlier_natural_completion' if natural else 'completed_batch_at_least_three_results'
                summary['early_elapsed_seconds'] = runtime.elapsed_seconds()
                summary['early_context_sha256'] = native.sha(early)
                runtime.save('early_context.json', early)
                try:
                    summary['early_workspace_checkpoint'] = native.checkpoint_summary(runtime.checkpoint('early'))
                    if not summary['early_workspace_checkpoint'].get('complete', False):
                        raise RuntimeError('early_workspace_snapshot_incomplete')
                    summary['early_workspace_status'] = 'captured'
                except BaseException as error:
                    summary['early_workspace_status'] = 'failed'
                    summary['early_workspace_error'] = repr(error)
                    raise
                finally:
                    runtime.save('early_capture.json', {key: value for key, value in summary.items() if key != 'probes'})
            if natural:
                summary['end_status'] = 'natural_completion'
                runtime.save('end_context.json', messages)
                summary['end_workspace_checkpoint'] = native.checkpoint_summary(runtime.checkpoint('end'))
                summary['end_workspace_status'] = 'captured' if summary['end_workspace_checkpoint'].get('complete') else 'failed'
                break
        else:
            summary['end_status'] = 'administratively_censored_turns'
    except ActorDeadlineExceeded as error:
        summary['end_status'] = 'administratively_censored_runtime'
        summary['actor_error'] = repr(error)
    except BaseException as error:
        summary['actor_error'] = repr(error)
        summary['end_status'] = 'technical_failure'
        failure('actor', error)
    finally:
        summary['actor_gate_released'] = gate_released
        try:
            summary['actor_context_sha256'] = native.sha(messages)
        except BaseException as error:
            summary['actor_context_sha256'] = None
            failure('actor_context_hash', error)
        try:
            if not terminal_attempted:
                terminal()
        except BaseException as error:
            failure('terminal', error)
        try:
            runtime.save('actor_final_context.json', messages)
        except BaseException as error:
            failure('final_context', error)
        try:
            closure = runtime.close()
            summary['sandbox_closure'] = copy.deepcopy(closure)
        except BaseException as error:
            summary['sandbox_close_error'] = repr(error)
            failure('closure', error)
        try:
            runtime.save('actor_summary.json', summary)
        except BaseException as error:
            failure('actor_summary', error)
        summary['finished_at'] = native.now()
        try:
            summary['elapsed_seconds'] = runtime.elapsed_seconds() if gate_released else None
            runtime.save('summary.json', summary)
        except BaseException as error:
            failure('summary', error)
    return summary

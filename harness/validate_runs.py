"""Validate every run's recorded model identifier against the released roster."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq


RUN_AGENT_MODELS = {
    'deepseek-flash': 'DeepSeek',
    'glm-5.3-flash-max': 'GLM',
    'z-ai/glm-5.3-flash': 'GLM',
    'qwen3.8-27b': 'Qwen',
    'nemotron-lightning': 'Nemotron',
    'nvidia/NVIDIA-Nemotron-3.5-Lightning': 'Nemotron',
    'gpt-oss-120b': 'gpt-oss-120b',
    'gpt-oss-120b-chat': 'gpt-oss-120b',
    'openai/gpt-oss-120b': 'gpt-oss-120b',
    'local-gpt-oss-20b-mxfp4': 'gpt-oss-20b',
    'opus-5.5': 'Opus',
    'haiku-5.5': 'Haiku 5.5',
    'openai-gpt-6.1-sol-high': 'Sol',
    'openai-o3-high': 'o3',
    **{'openai-gpt-6-luna-' + effort: 'Luna' for effort in ('none', 'low', 'medium', 'high')},
    **{'openai-gpt-5.5-' + effort: 'GPT-5.5' for effort in ('none', 'low', 'medium', 'high')},
    **{'openai-' + name + '-plain': label for name, label in (
        ('gpt-4o', 'GPT-4o'), ('gpt-4.1', 'GPT-4.1'), ('gpt-4.1-mini', 'GPT-4.1 mini'))},
    **{'openai-' + name + '-high': label for name, label in (
        ('gpt-5', 'GPT-5'), ('gpt-5-mini', 'GPT-5 mini'), ('gpt-5-nano', 'GPT-5 nano'),
        ('gpt-5.1', 'GPT-5.1'), ('gpt-5.2', 'GPT-5.2'), ('gpt-5.3', 'GPT-5.3'),
        ('gpt-5.4', 'GPT-5.4'), ('gpt-5.4-mini', 'GPT-5.4 mini'), ('gpt-5.4-nano', 'GPT-5.4 nano'),
        ('gpt-5.6-luna', 'GPT-5.6 Luna'), ('gpt-5.6-sol', 'GPT-5.6 Sol'),
        ('gpt-5.6-terra', 'GPT-5.6 Terra'), ('gpt-6-sol', 'GPT-6 Sol'), ('gpt-6.1-terra', 'GPT-6.1 Terra'))},
}


def validate(data):
    data = Path(data)
    with (data / 'setups/agents.csv').open(newline='') as stream:
        roster = {row['agent'] for row in csv.DictReader(stream)}
    if not set(RUN_AGENT_MODELS.values()) <= roster:
        raise ValueError('Recorded model mapping contains a model outside the roster')
    paths = sorted(path for path in (data / 'runs').glob('*/*.parquet') if path.parent.name != '_field_chunks')
    if not paths:
        raise ValueError('No run sets found')
    index = json.loads((data / 'runs/index.json').read_text())
    expected = {entry['path'] for group in index['run_sets'] for entry in group['shards']}
    if expected != {str(path.relative_to(data)) for path in paths}:
        raise ValueError('Run files differ from the run-set index')
    errors, groups, agents, statuses, identities = [], Counter(), Counter(), Counter(), set()
    for path in paths:
        ordinal = 0
        for batch in pq.ParquetFile(path).iter_batches(columns=['run_id', 'run_set', 'agent', 'status'], use_threads=False):
            for row in batch.to_pylist():
                if row['agent'] not in RUN_AGENT_MODELS or RUN_AGENT_MODELS.get(row['agent']) not in roster:
                    errors.append(dict(file=str(path.relative_to(data)), row=ordinal, **row))
                if row['run_set'] != path.parent.name:
                    errors.append(dict(file=str(path.relative_to(data)), row=ordinal, error='run_set_mismatch'))
                groups[row['run_set']] += 1
                agents[row['agent']] += 1
                statuses[row['status']] += 1
                identities.add(row['run_id'])
                ordinal += 1
    for group in index['run_sets']:
        if groups[group['run_set']] != group['rows']:
            errors.append(dict(run_set=group['run_set'], error='row_count_mismatch'))
    return {'passed': not errors, 'errors': errors, 'run_rows': sum(groups.values()),
            'run_sets': dict(groups), 'agents': dict(agents), 'statuses': dict(statuses),
            'unique_run_ids': len(identities), 'files': len(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = validate(args.data)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

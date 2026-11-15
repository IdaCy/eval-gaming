"""Build layer-32 directions from paired residual states and random controls."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def read_matrix(path):
    column = pq.read_table(path, columns=['values'])['values'].combine_chunks()
    return column.values.to_numpy().reshape(len(column), column.type.list_size).astype(np.float32)


def read_vector(path):
    return pq.read_table(path, columns=['value'])['value'].to_numpy().astype(np.float32)


def select_pairs(runs, first_signal):
    pairs = []
    for (task, variant), group in runs.groupby(['task_id', 'variant_id']):
        for high in group[group.above == 1].itertuples():
            for low in group[group.above == 0].itertuples():
                h, l = getattr(high, first_signal), getattr(low, first_signal)
                h = high.n_turns + 1 if pd.isna(h) else h
                l = low.n_turns + 1 if pd.isna(l) else l
                boundary = int(min(h, l))
                last = min(boundary - 1, int(high.n_turns), int(low.n_turns))
                if last >= 2:
                    pairs.append({'task_id': task, 'variant_id': variant, 'high_run': high.run_id,
                                  'low_run': low.run_id, 'first_turn': 2, 'last_turn': last})
    return pd.DataFrame(pairs)


def gaming_direction(states, index, pairs):
    lookup = {(row.run_id, row.turn): row.state_row for row in index.itertuples()}
    per_setup = {}
    for row in pairs.itertuples():
        turns = range(row.first_turn, row.last_turn + 1)
        high = [lookup[row.high_run, turn] for turn in turns]
        low = [lookup[row.low_run, turn] for turn in turns]
        difference = (states[high] - states[low]).mean(0)
        per_setup.setdefault((row.task_id, row.variant_id), []).append(difference)
    raw = np.mean([np.mean(values, 0) for values in per_setup.values()], 0)
    return raw / np.linalg.norm(raw), float(np.linalg.norm(raw)), len(per_setup)


def read_turn_states(directory, index):
    states = np.empty((len(index), 5120), dtype=np.float32)
    for run_id, rows in index.groupby('run_id', sort=False):
        with np.load(directory / (run_id + '.npz'), allow_pickle=False) as record:
            layer = list(record['layers']).index(32)
            anchors = {int(turn): i for i, (turn, kind) in
                       enumerate(zip(record['anchor_turn'], record['anchor_type'])) if kind == 0}
            for row in rows.itertuples():
                states[row.state_row] = record['act'][layer, anchors[row.turn]]
    return states


def read_prompt_states(path, index):
    metadata = [json.loads(line) for line in path.with_suffix('.meta.jsonl').read_text().splitlines()]
    lookup = {(row['task_id'], row['variant_id']): row['i'] for row in metadata}
    with np.load(path, allow_pickle=False) as record:
        layer = list(record['layers']).index(32)
        return np.stack([record['acts'][lookup[row.task_id, row.variant_id], layer]
                         for row in index.sort_values('state_row').itertuples()]).astype(np.float32)


def build(data, output, activations=None, prefills=None):
    output.mkdir(parents=True, exist_ok=True)
    runs = pd.read_parquet(data / 'direction_runs.parquet')
    assert ((runs.nat_pct > 84.689).astype(int) == runs.above).all()
    index = pd.read_parquet(data / 'gaming_state_index.parquet')
    states = (read_turn_states(activations, index) if activations else
              read_matrix(data / 'gaming_states.parquet'))
    vectors, facts = {}, {}
    for name, field, pair_file in [('gaming', 'first_any', 'gaming_pairs.parquet'),
                                   ('gaming_strong', 'first_strong', 'strong_pairs.parquet')]:
        pairs = select_pairs(runs, field)
        expected = pd.read_parquet(data / pair_file)
        pd.testing.assert_frame_equal(pairs, expected, check_dtype=False)
        vector, gap, setups = gaming_direction(states, index, pairs)
        vectors[name] = vector.astype(np.float32)
        facts[name] = {'pairs': len(pairs), 'setups': setups, 'tasks': int(pairs.task_id.nunique()),
                       'unit_gap': gap, 'median_window_turns': float((pairs.last_turn - pairs.first_turn + 1).median())}
    prompt_states = (read_prompt_states(prefills, pd.read_parquet(data / 'prompt_state_index.parquet'))
                     if prefills else read_matrix(data / 'evaluation_states.parquet'))
    prompt_pairs = pd.read_parquet(data / 'evaluation_pairs.parquet')
    differences = prompt_states[prompt_pairs.high_state] - prompt_states[prompt_pairs.low_state]
    mean = differences.mean(0)
    vectors['evaluation'] = (mean / np.linalg.norm(mean)).astype(np.float32)
    facts['evaluation'] = {'pairs': len(prompt_pairs), 'unit_gap': float(np.linalg.norm(mean))}
    rng = np.random.default_rng(20261103)
    basis = np.stack([vectors['gaming'], vectors['gaming_strong']])
    q, _ = np.linalg.qr(basis.T)
    for k in range(3):
        for layer in [16, 24, 32, 40, 48]:
            x = rng.standard_normal(states.shape[1]).astype(np.float64)
            if layer != 32:
                continue
            x -= q @ (q.T @ x)
            x /= np.linalg.norm(x)
            vectors[f'random_{k + 1}'] = x.astype(np.float32)
            facts[f'random_{k + 1}'] = {'unit_gap': facts['gaming']['unit_gap']}
    for name, vector in vectors.items():
        path = output / (name + '.npy')
        np.save(path, vector)
        reference = read_vector(data / 'directions' / (name + '.parquet'))
        facts[name].update(layer=32, hidden_size=len(vector),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           equals_archived_vector=bool(np.array_equal(reference, vector)),
                           max_absolute_error=float(np.max(np.abs(reference - vector))))
    a, b = vectors['gaming'].astype(float), vectors['evaluation'].astype(float)
    facts['cosine'] = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    facts['random_isotropic_cosine_sd'] = float(1 / np.sqrt(len(a)))
    facts['random_control_cosines'] = {name: float(np.dot(a, vector) / (np.linalg.norm(a) * np.linalg.norm(vector.astype(float))))
                                      for name, vector in vectors.items() if name.startswith('random_')}
    (output / 'facts.json').write_text(json.dumps(facts, indent=2) + '\n')
    print(json.dumps(facts, indent=2))
    return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'directions')
    parser.add_argument('--activations', type=Path, help='Optional directory produced by extract.py')
    parser.add_argument('--prefills', type=Path, help='Optional NPZ produced by prefill.py')
    args = parser.parse_args()
    build(args.data, args.output, args.activations, args.prefills)

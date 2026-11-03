"""Matched-issue awareness levels and material-change contrasts."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CONDITIONS = ['E01', 'F08L03', 'F09L05', 'F11L02', 'F11L04']


def compare(frame, seed, conditions=CONDITIONS):
    rng = np.random.default_rng(seed)
    base = frame[frame.condition.eq('B00L00')].set_index('task_id')
    out = {}
    for code in conditions:
        cue = frame[frame.condition.eq(code)].set_index('task_id')
        tasks = cue.index.intersection(base.index)
        if len(tasks) < 10:
            out[code] = dict(tasks=len(tasks), value=None)
            continue
        weight = cue.loc[tasks, 'n'].to_numpy()
        a = (cue.loc[tasks, 'total'] / cue.loc[tasks, 'n']).to_numpy()
        b = (base.loc[tasks, 'total'] / base.loc[tasks, 'n']).to_numpy()
        estimate = np.average(a - b, weights=weight)
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, len(tasks), len(tasks))
            draws.append(np.average(a[idx] - b[idx], weights=weight[idx]))
        out[code] = dict(tasks=len(tasks), runs=int(weight.sum()),
                         p0=100*float(np.average(b, weights=weight)),
                         p1=100*float(np.average(a, weights=weight)),
                         value=100*float(estimate),
                         ci=(100*np.percentile(draws, [2.5, 97.5])).tolist())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DATA / 'general_judge_task_sums.parquet')
    p.add_argument('--output', type=Path, default=OUTPUT / 'general_judges.json')
    args = p.parse_args()
    frame = pd.read_parquet(args.data)
    result = {}
    for model, sub in frame.groupby('model'):
        result[model] = {}
        for outcome in ['G', 'A']:
            seed = 20261111 if model == 'DeepSeek' and outcome == 'G' else 20261112
            result[model][outcome] = compare(sub[sub.outcome.eq(outcome)], seed)
        if model == 'DeepSeek':
            headings = ['F07L02', 'F07L03', 'F07L04', 'F07L05']
            result[model]['headings'] = {outcome: compare(sub[sub.outcome.eq(outcome)], 20261112, headings)
                                         for outcome in ['G', 'A']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()

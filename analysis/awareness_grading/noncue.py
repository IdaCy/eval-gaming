"""Within-setup provider and repeat differences, resampling issues."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DATA / 'noncue_cell_sums.parquet')
    p.add_argument('--output', type=Path, default=OUTPUT / 'noncue.csv')
    args = p.parse_args()
    frame = pd.read_parquet(args.data)
    rng = np.random.default_rng(20261111)
    result = []
    for factor in ['provider', 'repeat']:
        for outcome in ['judged', 'notests', 'ref', 'aware', 'log_calls', 'passx']:
            j = frame[frame.factor.eq(factor) & frame.outcome.eq(outcome)].sort_values('order')
            if len(j) < 30:
                continue
            w = 1 / (1/j.size_a + 1/j.size_b)
            numerator = (j.mean_a-j.mean_b)*w
            tasks = j.task.unique()
            grouped = pd.DataFrame(dict(task=j.task, numerator=numerator, weight=w)).groupby('task').sum().reindex(tasks)
            num, den = grouped.numerator.to_numpy(), grouped.weight.to_numpy()
            draws = []
            for _ in range(1000):
                idx = rng.choice(len(tasks), len(tasks))
                draws.append(num[idx].sum()/den[idx].sum())
            lo, hi = np.percentile(draws, [2.5, 97.5])
            result.append(dict(factor=factor, outcome=outcome, cells=len(j),
                               runs_a=int(j.size_a.sum()), runs_b=int(j.size_b.sum()),
                               diff=round(float(numerator.sum()/w.sum()), 4),
                               lo=round(float(lo), 4), hi=round(float(hi), 4)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(args.output, index=False)


if __name__ == '__main__':
    main()

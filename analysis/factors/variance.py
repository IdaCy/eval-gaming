"""Variance shares of task, model, setup, and their interactions."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import crossed_reml as _cr
from crossed_reml import CrossedREML
METHODS = {'jeffreys_fixed': {'smoothing': 0.5, 'cap': False, 'provider': False}}

def make_cells(frame, outcome, agents, method):
    options = METHODS[method]
    data = frame[frame.model.isin(agents) & frame[outcome].notna()].copy()
    if options['cap']:
        data = data[data.setup_rank <= 10]
    keys = ['model', 'task_id', 'condition']
    if options['provider']:
        keys += ['provider']
    cells = data.groupby(keys, observed=True).agg(k=(outcome, 'sum'), n=(outcome, 'size'), repo=('repo', 'first')).reset_index()
    a = options['smoothing']
    cells['smoothed_probability'] = (cells.k + a) / (cells.n + 2 * a)
    cells['logodds'] = np.log((cells.k + a) / (cells.n - cells.k + a))
    cells['sampling_variance'] = 1 / (cells.k + a) + 1 / (cells.n - cells.k + a)
    cells['inverse_variance_weight'] = 1 / cells.sampling_variance
    return cells
_orig_min = _cr.minimize

def _robust_minimize(fun, x0, *a, **k):
    best = _orig_min(fun, x0, *a, **k)
    if best.success:
        return best
    tries = []
    for s in (0.5, 2.0, 1.3, 0.8, 3.0, 0.3):
        kk = dict(k)
        kk['options'] = dict(k.get('options', {}), maxls=100, maxiter=5000)
        r = _orig_min(fun, np.maximum(np.asarray(x0) * s, 1e-06), *a, **kk)
        tries.append(r)
        if r.success and r.fun <= best.fun + 1e-09:
            return r
    ok = [r for r in tries if r.success]
    if ok:
        return min(ok, key=lambda r: r.fun)
    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.data / 'factor_runs.parquet')
    frame = frame[frame.dataset.eq('paper_main')].copy()
    frame['G_judge'] = frame.judged
    conds = sorted(frame.condition.unique())
    small = [x for x in conds if x[:3] in ('F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F10') or x in ('F07L00', 'F08L01')] + ['B00L00']
    scopes = {'pooled': ['DeepSeek', 'GLM', 'Qwen', 'gpt-oss-120b'], 'DeepSeek': ['DeepSeek'], 'GLM': ['GLM'], 'Qwen': ['Qwen'], 'gpt-oss-120b': ['gpt-oss-120b'], 'small_changes': ['DeepSeek']}
    out = []
    for scope, agents in scopes.items():
        fr = frame[frame.condition.isin(small)] if scope == 'small_changes' else frame
        cells = make_cells(fr, 'G_judge', agents, 'jeffreys_fixed')
        m = CrossedREML(cells, cells.logodds, cells.sampling_variance, noise_scale=False, provider_fixed=False, agent_task=False)
        r = m.fit()
        if not r.converged or r.projected_gradient > 0.005:
            r2 = m.fit(r.variance + 0.001, multistart=True)
            if r2.nll < r.nll or (r2.converged and r2.projected_gradient < r.projected_gradient):
                r = r2
        v = dict(zip(m.names, [float(x) for x in r.variance[:len(m.names)]]))
        total = sum(v.values())
        row = {'scope': scope, 'runs': int(cells.n.sum()), 'cells': len(cells), 'conditions': int(cells.condition.nunique()), 'tasks': int(cells.task_id.nunique()), 'converged': bool(r.converged), 'shares_pct': {k: 100 * x / total for k, x in v.items()}, 'variance': v}
        if scope != 'pooled':
            try:
                lo, hi, _ = m.profile_share(r, 'condition')
            except RuntimeError:
                _cr.minimize = _robust_minimize
                try:
                    lo, hi, _ = m.profile_share(r, 'condition')
                finally:
                    _cr.minimize = _orig_min
            row['condition_share_ci_pct'] = [100 * lo, 100 * hi]
        out.append(row)
        print(json.dumps(row), flush=True)
        (args.output / 'variance.json').write_text(json.dumps(out, indent=2) + '\n')
if __name__ == '__main__':
    main()

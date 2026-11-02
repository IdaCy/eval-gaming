from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import pandas as pd
FEAT = ['O1', 'T1', 'T2', 'T3', 'S1', 'RP', 'O2', 'n_tool_calls', 'TK_reasoning', 'TK_output', 'ci_files_opened']
TESTW = {'O1', 'T1', 'T2'}
BIN = {'O1', 'O2', 'ci_files_opened', 'O3'}
CODES = ['F16L03', 'F17L01', 'F23L01', 'F15L01', 'F22L01', 'F19L01', 'F11L02', 'F18L03', 'F18L01', 'F24L06', 'F24L02', 'F18L04', 'F18L02', 'F20L01', 'F24L04', 'F24L05', 'F24L03', 'F24L01']
B, NPERM, K, LAM, SEED = (2000, 200, 5, 1.0, 20261111)
SOURCE_INDEX = {}

def load_runs(model):
    spec = SOURCE_INDEX[model]
    frames = [pd.read_parquet(p) for p in spec['frames']]
    df = pd.concat(frames, ignore_index=True)
    df = df[df.eligible.fillna(False).astype(bool) & df.design.isin(['P2', 'P3', 'P8']) & (df.task_set == spec['task_set'])].copy()
    df = df[df.code.isin(CODES + ['B00L00'])]
    df['n_tool_calls'] = np.log1p(df.n_tool_calls.astype(float))
    return df

def auc(s, y):
    r = pd.Series(s).rank().to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan

def fisher(Xtr, ytr, Xte):
    sd = Xtr.std(0)
    keep = sd > 1e-09
    if not keep.any():
        return np.zeros(len(Xte))
    Xtr = Xtr[:, keep] / sd[keep]
    Xte = Xte[:, keep] / sd[keep]
    a, b = (Xtr[ytr == 1], Xtr[ytr == 0])
    m1, m0 = (a.mean(0), b.mean(0))
    Sw = ((a - m1).T @ (a - m1) + (b - m0).T @ (b - m0)) / max(len(Xtr) - 2, 1)
    w = np.linalg.solve(Sw + LAM * np.eye(Sw.shape[0]), m1 - m0)
    return Xte @ w

def cv_auc(X, y, g, rng):
    ug = np.unique(g)
    perm = rng.permutation(len(ug))
    fold_of = dict(zip(ug[perm], np.arange(len(ug)) % K))
    f = np.array([fold_of[x] for x in g])
    s = np.empty(len(y))
    for k in range(K):
        te = f == k
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            s[te] = 0.0
            continue
        s[te] = fisher(X[tr], y[tr], X[te])
    return auc(s, y)

def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) > 50 else [None, None]

def run_model(m):
    rng = np.random.default_rng(SEED + sum(map(ord, m)))
    df = load_runs(m)
    feats = [f for f in FEAT if df[f].notna().mean() > 0.5]
    tasks = sorted(df['task'].unique())
    tid = {t: i for i, t in enumerate(tasks)}
    setups = {}
    for code in CODES:
        d = df[df['code'].isin([code, 'B00L00'])].copy()
        both = d.groupby('task')['code'].nunique()
        d = d[d['task'].isin(both[both == 2].index)]
        if (d['code'] == code).sum() < 20:
            continue
        y = (d['code'] == code).astype(int).to_numpy()
        g = d['task'].map(tid).to_numpy()
        Xr = d[feats].astype(float)
        Xc = (Xr - Xr.groupby(d['task']).transform('mean')).fillna(0.0)
        nt = [i for i, f in enumerate(feats) if f not in TESTW]
        rows = {t: np.where(g == t)[0] for t in np.unique(g)}
        per_issue = {}
        for f in feats + ['O3']:
            if f not in d or d[f].notna().sum() == 0:
                continue
            mm = d.groupby(['task', 'code'])[f].mean().unstack()
            if code in mm and 'B00L00' in mm:
                per_issue[f] = mm[[code, 'B00L00']].dropna()
        setups[code] = dict(X=Xc.to_numpy(), y=y, g=g, nt=nt, rows=rows, per_issue=per_issue, n_issues=int(len(rows)), n_setup=int(y.sum()), n_nocue=int((1 - y).sum()))
    res = {'model': m, 'features': feats, 'n_tasks': len(tasks), 'setups': {}}
    point = {}
    for code, s in setups.items():
        a_all = np.mean([cv_auc(s['X'], s['y'], s['g'], np.random.default_rng(SEED + i)) for i in range(10)])
        a_nt = np.mean([cv_auc(s['X'][:, s['nt']], s['y'], s['g'], np.random.default_rng(SEED + i)) for i in range(10)])
        null = []
        for p in range(NPERM):
            yp = s['y'].copy()
            for t, ix in s['rows'].items():
                yp[ix] = rng.permutation(yp[ix])
            null.append(cv_auc(s['X'], yp, s['g'], np.random.default_rng(SEED + 7)))
        point[code] = (a_all, a_nt)
        res['setups'][code] = {'n_issues': s['n_issues'], 'n_runs_setup': s['n_setup'], 'n_runs_nocue': s['n_nocue'], 'auc_all': {'est': a_all, 'null_mean': float(np.mean(null)), 'null_95': float(np.quantile(null, 0.95))}, 'auc_nontest': {'est': a_nt}}
    draws = {c: {'all': [], 'nt': []} for c in setups}
    actd = {c: {f: [] for f in s['per_issue']} for c, s in setups.items()}
    utasks = np.arange(len(tasks))
    for bb in range(B):
        samp = rng.choice(utasks, size=len(utasks), replace=True)
        cnt = pd.Series(samp).value_counts()
        vals = {}
        for code, s in setups.items():
            ix = np.concatenate([np.repeat(s['rows'][t][None, :], c, axis=0).ravel() for t, c in cnt.items() if t in s['rows']])
            X, y, g = (s['X'][ix], s['y'][ix], s['g'][ix])
            r1 = np.random.default_rng(bb)
            a1 = cv_auc(X, y, g, r1)
            a2 = cv_auc(X[:, s['nt']], y, g, r1)
            draws[code]['all'].append(a1)
            draws[code]['nt'].append(a2)
            names = [tasks[t] for t in cnt.index]
            w = cnt.to_numpy()
            for f, pi in s['per_issue'].items():
                v = pi.reindex(names)
                ok = v.notna().all(axis=1).to_numpy()
                dlt = (v[code] - v['B00L00']).to_numpy()
                actd[code][f].append(float(np.average(dlt[ok], weights=w[ok])) if ok.any() else np.nan)
    for code, s in setups.items():
        e = res['setups'][code]
        e['auc_all']['ci'] = q(draws[code]['all'])
        e['auc_nontest']['ci'] = q(draws[code]['nt'])
        acts = {}
        for f, pi in s['per_issue'].items():
            lv1, lv0 = (float(pi[code].mean()), float(pi['B00L00'].mean()))
            dl = float((pi[code] - pi['B00L00']).mean())
            acts[f] = {'level_setup': lv1, 'level_nocue': lv0, 'diff': dl, 'ci': q(actd[code][f]), 'kind': 'share' if f in BIN or f == 'judge' else 'log'}
        e['actions'] = acts
    return res

def compute(model):
    result = run_model(model)
    target = OUTPUT / (model + '.json')
    target.write_text(json.dumps(result, indent=1, default=float) + '\n')
    return (model, len(result['setups']))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-index', type=Path)
    parser.add_argument('--frames', type=Path, nargs='+')
    parser.add_argument('--task-set', default='main')
    parser.add_argument('--output', type=Path, default=OUTPUT / 'auc_original')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--processes', type=int, default=4)
    args = parser.parse_args()
    if args.frames:
        if not args.models or len(args.models) != 1:
            parser.error('--frames requires exactly one --models value')
        SOURCE_INDEX = {args.models[0]: {'frames': args.frames, 'task_set': args.task_set}}
    elif args.source_index:
        SOURCE_INDEX = json.loads(args.source_index.read_text())
    else:
        parser.error('provide --frames or --source-index')
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.processes) as pool:
        for result in pool.imap_unordered(compute, args.models or list(SOURCE_INDEX)):
            print(result, flush=True)

"""Single-feature associations and repository-held-out prediction of task gaming."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
warnings.filterwarnings('ignore')

def task_levels(df, y, kind):
    d = df[df[y].notna()].copy()
    d['ctask'] = d.corpus + '|' + d.task_id
    X = pd.get_dummies(d[['ctask', 'condition', 'model']].astype(str), drop_first=False, sparse=True)
    cols = list(X.columns)
    Xs = X.sparse.to_coo().tocsr()
    if kind == 'bin':
        m = LogisticRegression(C=20.0, max_iter=3000, fit_intercept=True).fit(Xs, d[y].astype(int).values)
        coef = m.coef_[0]
    else:
        m = Ridge(alpha=0.05).fit(Xs, d[y].astype(float).values)
        coef = m.coef_
    u = {c[len('ctask_'):]: coef[i] for i, c in enumerate(cols) if c.startswith('ctask_')}
    n = d.groupby('ctask').size()
    t = pd.DataFrame({'ctask': list(u), 'u': list(u.values())})
    t['runs'] = t.ctask.map(n).values
    t['corpus'] = t.ctask.str.split('|').str[0]
    t['task_id'] = t.ctask.str.split('|', n=1).str[1]
    for cp in ['paper', 'fresh']:
        s = t.corpus.eq(cp)
        t.loc[s, 'u'] = t.loc[s, 'u'] - t.loc[s, 'u'].mean()
    return t

def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty(len(p))
    m = len(p)
    prev = 1.0
    for r, i in enumerate(o[::-1]):
        k = m - r
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q

def within_cells(df, y, a_mask, b_mask, label):
    d = df[df[y].notna()].copy()
    d['cell'] = d.model + '|' + d.corpus + '|' + d.task_id + '|' + d.condition
    A = d[a_mask.reindex(d.index).fillna(False)].groupby('cell')[y].agg(['mean', 'size'])
    B = d[b_mask.reindex(d.index).fillna(False)].groupby('cell')[y].agg(['mean', 'size'])
    J = A.join(B, lsuffix='_a', rsuffix='_b', how='inner')
    if len(J) < 30:
        return None
    J['w'] = 1 / (1 / J.size_a + 1 / J.size_b)
    J['task'] = [c.split('|')[2] for c in J.index]
    est = np.average(J.mean_a - J.mean_b, weights=J.w)
    tasks = J.task.unique()
    bs = []
    grp = {t: J[J.task.eq(t)] for t in tasks}
    for _ in range(1000):
        S = pd.concat([grp[t] for t in rng.choice(tasks, len(tasks))])
        bs.append(np.average(S.mean_a - S.mean_b, weights=S.w))
    return {'factor': label, 'outcome': y, 'cells': len(J), 'runs_a': int(J.size_a.sum()), 'runs_b': int(J.size_b.sum()), 'diff': round(float(est), 4), 'lo': round(float(np.percentile(bs, 2.5)), 4), 'hi': round(float(np.percentile(bs, 97.5)), 4)}
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=DATA)
parser.add_argument('--output', type=Path, default=OUTPUT)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
OUT = str(args.output) + '/'
rng = np.random.default_rng(20261111)
f = pd.read_parquet(args.data / 'factor_runs.parquet')
F = pd.read_parquet(args.data / 'task_features.parquet')
feats = [c for c in F if c not in ('corpus', 'task_id', 'repo')]
dic = pd.read_parquet(args.data / 'feature_dictionary.parquet')
fam = dict(zip(dic.name, dic.family))
defn = dict(zip(dic.name, dic.definition))

def family(c):
    return fam[c]
allrows = []
summ = {}
for y, kind in [('judged', 'bin'), ('notests', 'bin')]:
    tv = task_levels(f, y, kind)
    tv.to_csv(OUT + 'task_levels_' + y + '.csv', index=False)
    U = tv.merge(F, on=['corpus', 'task_id'], how='inner')
    U = U[U.repo.notna()].reset_index(drop=True)
    sd_u = float(U.u.std())
    rows = []
    for c in feats:
        z = pd.Series(np.nan, index=U.index)
        for cp in ['paper', 'fresh']:
            s = U.corpus.eq(cp)
            xs = U.loc[s, c].astype(float)
            if xs.notna().sum() >= 20 and xs.std() > 0:
                z[s] = (xs - xs.mean()) / xs.std()
        ok = z.notna()
        if ok.sum() < 40 or U.loc[ok & (U[c].astype(float) != U[c].astype(float).median()), 'repo'].nunique() < 8:
            continue
        X = sm.add_constant(pd.DataFrame({'z': z[ok], 'fresh': U.loc[ok, 'corpus'].eq('fresh').astype(float)}), has_constant='add')
        if X.fresh.nunique() < 2:
            X = X.drop(columns='fresh')
        r = sm.OLS(U.loc[ok, 'u'], X).fit(cov_type='cluster', cov_kwds={'groups': U.loc[ok, 'repo'].astype('category').cat.codes})
        rep = {}
        for cp in ['paper', 'fresh']:
            s = ok & U.corpus.eq(cp)
            rep[cp] = float(np.polyfit(z[s], U.loc[s, 'u'], 1)[0]) if s.sum() >= 20 else np.nan
        rows.append({'outcome': y, 'feature': c, 'family': family(c), 'tasks': int(ok.sum()), 'b_per_sd': float(r.params['z']), 'se': float(r.bse['z']), 'p': float(r.pvalues['z']), 'b_public': rep['paper'], 'b_recent': rep['fresh'], 'definition': defn.get(c, '')})
    R = pd.DataFrame(rows)
    R['q'] = bh(R.p.values)
    R['same_sign'] = np.sign(R.b_public) == np.sign(R.b_recent)
    R = R.sort_values('p')
    allrows.append(R)
    Z = U[feats].astype(float).copy()
    for cp in ['paper', 'fresh']:
        s = U.corpus.eq(cp)
        Z.loc[s] = (Z.loc[s] - Z.loc[s].mean()) / Z.loc[s].std().replace(0, np.nan)
    Z = Z.fillna(0.0).clip(-5, 5)
    pred = np.zeros(len(U))
    g = U.repo.astype('category').cat.codes.values
    for tr, te in GroupKFold(5).split(Z, U.u, g):
        pred[te] = RidgeCV(alphas=[30, 100, 300, 1000, 3000]).fit(Z.values[tr], U.u.values[tr]).predict(Z.values[te])
    r2 = 1 - ((U.u - pred) ** 2).sum() / ((U.u - U.u.mean()) ** 2).sum()
    famr = {}
    for fa in sorted(set((family(c) for c in feats))):
        cols = [c for c in feats if family(c) == fa]
        p2 = np.zeros(len(U))
        for tr, te in GroupKFold(5).split(Z, U.u, g):
            p2[te] = RidgeCV(alphas=[30, 100, 300, 1000, 3000]).fit(Z[cols].values[tr], U.u.values[tr]).predict(Z[cols].values[te])
        famr[fa] = [len(cols), round(float(np.corrcoef(p2, U.u)[0, 1]), 2), round(float(1 - ((U.u - p2) ** 2).sum() / ((U.u - U.u.mean()) ** 2).sum()), 3)]
    summ[y] = {'tasks': len(U), 'repos': int(U.repo.nunique()), 'sd_task_level': round(sd_u, 2), 'features_tested': len(R), 'q_below_0.1': int((R.q < 0.1).sum()), 'q_below_0.1_same_sign_both_corpora': int(((R.q < 0.1) & R.same_sign).sum()), 'expected_by_chance_p05': round(0.05 * len(R), 1), 'p_below_0.05': int((R.p < 0.05).sum()), 'out_of_repo_corr_all_features': round(float(np.corrcoef(pred, U.u)[0, 1]), 2), 'out_of_repo_R2_all_features': round(float(r2), 3), 'by_family[n,corr,R2]': famr}
    print(y, json.dumps(summ[y]), flush=True)
    print(R.head(14)[['feature', 'family', 'tasks', 'b_per_sd', 'se', 'q', 'b_public', 'b_recent']].round(3).to_string(), flush=True)
pd.concat(allrows).to_csv(OUT + 'task_features_screen_v2.csv', index=False)
runrows = []
ds = f[f.model.eq('DeepSeek')]
for y in ['judged', 'notests', 'ref', 'aware', 'log_calls', 'passx']:
    r = within_cells(ds, y, ds.provider.eq('deepseek'), ds.provider.eq('deepinfra'), 'provider')
    if y == 'judged' and r:
        runrows.append(r)
r = within_cells(f, 'judged', f.kind.eq('resample'), f.kind.eq('original'), 'repeat')
if r:
    runrows.append(r)
pd.DataFrame(runrows).to_csv(OUT + 'run_level_screen.csv', index=False)
json.dump(summ, open(OUT + 'task_feature_summary.json', 'w'), indent=2)
print('run_level', runrows, flush=True)

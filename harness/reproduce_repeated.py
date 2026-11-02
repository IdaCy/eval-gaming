from harness.data import input_path
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2

parser = argparse.ArgumentParser(description="Reproduce the repeated-run estimates and issue bootstrap intervals.")
parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
OUT = str(args.output)
MODE = "final"
B = 2000
rng = np.random.default_rng(20261115)
rng_new = np.random.default_rng(20261116)
cells = pd.read_parquet(input_path(args.data, "repeated_cells.parquet"))
records = []
for cell in cells.to_dict("records"):
    for position in range(cell["runs"]):
        row = {key: cell[key] for key in ["agent", "batch", "tset", "task", "code"]}
        for outcome in ["notests", "J", "fetch"]:
            row[outcome] = (float(position < cell[outcome + "_positive"])
                            if position < cell[outcome + "_observed"] else np.nan)
        records.append(row)
R = pd.DataFrame(records)

def terms(fs, col, tasks, code):
    fs = fs[fs[col].notna()]
    a = np.zeros(len(tasks))
    w = np.zeros(len(tasks))
    n1 = 0
    g0 = fs[fs.code == 'B00L00'].groupby('task')[col].agg(['mean', 'size'])
    g1 = fs[fs.code == code].groupby('task')[col].agg(['mean', 'size'])
    for t, r in g1.iterrows():
        if t in g0.index and t in tasks:
            i = tasks[t]
            nn1, nn0 = (r['size'], g0.at[t, 'size'])
            ww = nn1 * nn0 / (nn1 + nn0)
            a[i] += ww * (r['mean'] - g0.at[t, 'mean'])
            w[i] += ww
            n1 += nn1
    return (100 * a, w, n1)
rows = []
for (agent, tset), sub in R[R.tset.isin(['main', 'heldout'])].groupby(['agent', 'tset']):
    batches = sorted(sub.batch.unique())
    if len(batches) < 2:
        continue
    for code in sorted(sub.code.unique()):
        if code == 'B00L00':
            continue
        have = [b for b in batches if ((sub.batch == b) & (sub.code == code)).any() and ((sub.batch == b) & (sub.code == 'B00L00')).any()]
        if len(have) < 2:
            continue
        for col in ('notests', 'J', 'fetch'):
            tasks = {t: i for i, t in enumerate(sorted(sub.task.unique()))}
            T = {}
            for b in have:
                if col == 'J' and b == 'GSu' or (col == 'fetch' and b in ('GS', 'GSu')):
                    continue
                fs = sub[sub.batch == b]
                if fs[fs.code == code][col].notna().sum() < 20:
                    continue
                T[b] = terms(fs, col, tasks, code)
            if len(T) < 2:
                continue
            new_only = MODE == 'final' and len([b for b in T if not (agent == 'DeepSeek' and b == 'GS' and (col == 'J'))]) < 2
            M = (rng_new if new_only else rng).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B).astype(float)
            est, bs = ({}, {})
            for b, (A, W, n1) in T.items():
                with np.errstate(invalid='ignore', divide='ignore'):
                    est[b] = A.sum() / W.sum()
                    bs[b] = M @ A / (M @ W)
            Ap = sum((v[0] for v in T.values()))
            Wp = sum((v[1] for v in T.values()))
            with np.errstate(invalid='ignore', divide='ignore'):
                pe, pb = (Ap.sum() / Wp.sum(), M @ Ap / (M @ Wp))
            allc = sub[(sub.code == code) & sub.batch.isin(list(T)) & sub[col].notna()]
            alln = sub[(sub.code == 'B00L00') & sub.batch.isin(list(T)) & sub[col].notna() & sub.task.isin(set(allc.task))]
            r = {'agent': agent, 'issues': tset, 'code': code, 'outcome': col, 'batches': '+'.join(T), 'pooled_cue_rate': 100 * allc[col].mean(), 'pooled_nocue_rate': 100 * alln[col].mean(), 'pooled_cue_runs': len(allc), 'pooled_nocue_runs': len(alln), 'pooled': pe, 'pooled_lo': np.nanquantile(pb, 0.025), 'pooled_hi': np.nanquantile(pb, 0.975), 'new_cell': int(new_only)}
            for b in T:
                fs = sub[sub.batch == b]
                r[f'{b}'] = est[b]
                r[f'{b}_lo'] = np.nanquantile(bs[b], 0.025)
                r[f'{b}_hi'] = np.nanquantile(bs[b], 0.975)
                r[f'{b}_runs_cue'] = int(T[b][2])
                r[f'{b}_issues'] = int((T[b][1] > 0).sum())
                nc = fs[(fs.code == 'B00L00') & fs[col].notna()]
                r[f'{b}_nocue_rate'] = 100 * nc[col].mean()
                r[f'{b}_nocue_runs'] = len(nc)
                cc = fs[(fs.code == code) & fs[col].notna()]
                r[f'{b}_cue_rate'] = 100 * cc[col].mean()
            bl = list(T)
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)):
                    dd = bs[bl[i]] - bs[bl[j]]
                    r[f'gap_{bl[i]}-{bl[j]}'] = est[bl[i]] - est[bl[j]]
                    r[f'gap_{bl[i]}-{bl[j]}_lo'] = np.nanquantile(dd, 0.025)
                    r[f'gap_{bl[i]}-{bl[j]}_hi'] = np.nanquantile(dd, 0.975)
            dpt = np.array([est[b] - est[bl[-1]] for b in bl[:-1]])
            dbs = np.column_stack([bs[b] - bs[bl[-1]] for b in bl[:-1]])
            dbs = dbs[np.all(np.isfinite(dbs), axis=1)]
            try:
                S = np.atleast_2d(np.cov(dbs, rowvar=False))
                wv = float(dpt @ np.linalg.solve(S, dpt))
                r['joint_wald'] = wv
                r['joint_df'] = len(bl) - 1
                r['joint_p'] = float(chi2.sf(wv, len(bl) - 1))
            except Exception:
                r['joint_wald'] = np.nan
                r['joint_df'] = len(bl) - 1
                r['joint_p'] = np.nan
            rows.append(r)
pd.DataFrame(rows).to_csv(OUT + '/overlap_cells.csv', index=False)
nc = R[(R.code == 'B00L00') & R.tset.isin(['main', 'heldout'])]
nc.groupby(['agent', 'tset', 'batch']).agg(runs=('task', 'size'), issues=('task', 'nunique'), notests=('notests', 'mean'), J=('J', 'mean'), J_n=('J', 'count')).to_csv(OUT + '/nocue_rates.csv')
print('cells:', len(rows))

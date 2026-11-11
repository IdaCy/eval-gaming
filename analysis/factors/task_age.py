"""Estimate naming and fetching rates across task-age bins."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
import warnings
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm
from statsmodels.tools.numdiff import approx_hess3
from numpy.polynomial.hermite import hermgauss
from matched import expand_counts,boot_matrix,ci
def random_intercept_fit(x, y, repositories, nodes=40):
    nodes_x, weights = hermgauss(nodes)
    log_weights = np.log(weights) - 0.5 * np.log(np.pi)
    groups = [np.flatnonzero(repositories == r) for r in sorted(set(repositories))]
    p = x.shape[1]

    def objective(theta):
        beta, sd = (theta[:p], np.exp(theta[p]))
        intercepts = np.sqrt(2.0) * sd * nodes_x
        value = 0.0
        for indices in groups:
            linear = (x[indices] @ beta)[:, None] + intercepts[None, :]
            log_conditional = (y[indices, None] * linear - np.logaddexp(0.0, linear)).sum(axis=0)
            value -= logsumexp(log_weights + log_conditional)
        return float(value)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        initial_fit = sm.GLM(y, x, family=sm.families.Binomial()).fit(maxiter=200)
    initial = np.r_[initial_fit.params, np.log(0.5)]
    result = minimize(objective, initial, method='L-BFGS-B', bounds=[(None, None)] * p + [(-12.0, 6.0)], options={'ftol': 1e-12, 'gtol': 1e-07, 'maxiter': 2000})
    if result.x[p] < -10:
        cov = np.asarray(initial_fit.cov_params())
        return (np.asarray(initial_fit.params), cov, 0.0, 'zero-variance boundary (ordinary logistic)')
    hessian = approx_hess3(result.x, objective)
    covariance = np.linalg.inv(hessian)[:p, :p]
    return (result.x[:p], covariance, float(np.exp(result.x[p])), 'random intercept ML (Gauss-Hermite)')

def h3b_like(frame, y, conds):
    d = frame[frame[y].notna() & frame.condition.isin(conds)].copy()
    cols = {'intercept': np.ones(len(d)), 'age': d.age_score.to_numpy(float)}
    for v in sorted(d.condition.unique())[1:]:
        cols['condition_' + v] = d.condition.eq(v).to_numpy(float)
    x = np.column_stack(list(cols.values()))
    yy = d[y].to_numpy(float)
    ev = int(min(yy.sum(), len(yy) - yy.sum()))
    if ev < 5:
        return {'outcome': y, 'n': len(d), 'rare_events': ev, 'slope': np.nan, 'note': 'too few events'}
    try:
        b, V, sd, method = random_intercept_fit(x, yy, d.repository.to_numpy(str))
    except Exception as e:
        return {'outcome': y, 'n': len(d), 'rare_events': ev, 'slope': np.nan, 'note': f'fit failed: {type(e).__name__}'}
    se = float(np.sqrt(V[1, 1]))
    return {'outcome': y, 'n': len(d), 'tasks': d.task_id.nunique(), 'repositories': d.repository.nunique(), 'rare_events': ev, 'slope': float(b[1]), 'lo': float(b[1] - 1.959964 * se), 'hi': float(b[1] + 1.959964 * se), 'p_one_sided_older_higher': float(1 - norm.cdf(b[1] / se)), 'repo_sd': sd, 'method': method}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=RESULTS)
    args=parser.parse_args();DATA=args.data;args.output.mkdir(parents=True,exist_ok=True)
    a=pd.read_parquet(DATA/'age_counts.parquet');tasks=sorted(a.task_id.unique())
    mb=boot_matrix(len(tasks),2000,seed=20261158);rows=[];slopes=[]
    for y in ['O9','O2a']:
        f=a[a.outcome.eq(y)]
        runs=expand_counts(f,y).rename(columns={'repo':'repository'})
        slopes.append(h3b_like(runs,y,['E01']))
        for bin_name,d in f.groupby('age_bin'):
            g=d.set_index('task_id')[['k','n']].reindex(tasks).fillna(0)
            k,n=g.k.to_numpy(float),g.n.to_numpy(float)
            with np.errstate(divide='ignore',invalid='ignore'):
                b=(mb@k)/(mb@n)
            lo,hi=ci(b)
            rows.append({'age_bin':bin_name,'outcome':y,'rate':100*k.sum()/n.sum(),'lo':100*lo,'hi':100*hi,'events':int(k.sum()),'n':int(n.sum()),'tasks':int((n>0).sum())})
    pd.DataFrame(rows).to_csv(args.output/'age_bins.csv',index=False)
    (args.output/'age_slopes.json').write_text(json.dumps(slopes,indent=2)+'\n')
    print(json.dumps(slopes))

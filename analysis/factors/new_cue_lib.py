"""Marginal calibration and forecasts for new cues."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
from scipy.special import expit,logsumexp
from scipy.optimize import minimize_scalar
import forecast_lib as F
SEED=20261109
XG=np.linspace(-7,7,301)
LWG=-.5*XG**2
LWG=LWG-logsumexp(LWG)
CELLS=['C1','C2','C3','C4','C5','C6']
UNITS={'C1':{},'C2':{'id_known':1},'C3':{'id_known':1,'id_format':1},'C4':{'id_format':2},'C5':{'id_format':1},'C6':{'id_format':1}}
ERA={'DeepSeek':'deepinfra','GLM':'one'}

def load_fit(outcome,agent):
    F.DATA=DATA
    return F.load_fit(outcome,agent)

def task_tau(tasks,tn,v):
    F.DATA=DATA
    return F.task_Z(tasks,tn,'paper')@v

def load_runs():
    r=pd.read_parquet(DATA/'factor_runs.parquet')
    r=r[r.dataset.eq('paper_main')].copy()
    r['era']=np.where(r.model.eq('DeepSeek'),r.era.fillna('one').astype(str),'one')
    r['G_rmask']=r.judged_old;r['G_tests']=r.notests
    return r
def unit_vec(cell, sn):
    u = np.zeros(len(sn))
    for r, k in UNITS[cell].items():
        u[sn.index(r)] = k
    return u

def P_of(mu, e):
    return np.clip(mu + (1 - mu) * expit(e), 1e-09, 1 - 1e-09)

def calibrate(kt, nt_, mu, g, tau, s, prior):

    def ll_of(a):
        P = P_of(mu, a + g * (tau[:, None] + s * XG[None, :]))
        return kt[:, None] * np.log(P) + (nt_ - kt)[:, None] * np.log(1 - P)
    a = minimize_scalar(lambda a: -float(logsumexp(ll_of(a) + prior, axis=1).sum()), bounds=(-20, 20), method='bounded', options={'xatol': 1e-07}).x
    lp = ll_of(a) + prior
    return (float(a), lp - logsumexp(lp, axis=1)[:, None])

def cell_probs(a, logpost, mu, g, tau, s, w, sn, cells):
    post = np.exp(logpost)
    out = np.zeros((len(tau), len(cells)))
    for j, c in enumerate(cells):
        xw = unit_vec(c, sn) @ w
        out[:, j] = (post * P_of(mu, a + g * ((tau + xw)[:, None] + s * XG[None, :]))).sum(1)
    return out

def main_nocue(R, agent, outcome, tasks):
    nc = R[R.model.eq(agent) & R.era.eq(ERA[agent]) & R.condition.eq('B00L00') & R[outcome].notna()]
    g = nc.groupby('task_id')[outcome].agg(['sum', 'count']).reindex(tasks).fillna(0)
    return (g['sum'].values.astype(float), g['count'].values.astype(float))

def main_prior(kt, nt_, par, tau):
    prior = np.tile(LWG, (len(tau), 1))
    return calibrate(kt, nt_, par['mu'], par['gain'], tau, par['s'], prior)

def summarise(P, runs_per_issue, rng, nsim=20):
    pt = P[0].mean(0)
    dm = P[1:].mean(1)
    res = {}
    D = P[1:]
    sims = rng.binomial(runs_per_issue, np.repeat(D[None], nsim, 0)) / runs_per_issue
    obs = sims.mean(2).reshape(-1, D.shape[2])
    for j in range(P.shape[2]):
        r = {'pred': 100 * pt[j], 'refit_lo': 100 * np.percentile(dm[:, j], 2.5), 'refit_hi': 100 * np.percentile(dm[:, j], 97.5), 'obs_pi_lo': 100 * np.percentile(obs[:, j], 2.5), 'obs_pi_hi': 100 * np.percentile(obs[:, j], 97.5)}
        if j > 0:
            d = 100 * (dm[:, j] - dm[:, 0])
            od = 100 * (obs[:, j] - obs[:, 0])
            r.update({'diff': 100 * (pt[j] - pt[0]), 'diff_refit_lo': np.percentile(d, 2.5), 'diff_refit_hi': np.percentile(d, 97.5), 'diff_obs_pi_lo': np.percentile(od, 2.5), 'diff_obs_pi_hi': np.percentile(od, 97.5)})
        res[j] = r
    return res

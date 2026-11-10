"""Final formula fit with repository and setup bootstrap intervals."""
import json
import os
import sys
import time
from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formula_lib as L
SEL_CONDS = ['E01', 'F07L00', 'F07L01', 'F07L02', 'F07L05', 'F08L01', 'F08L02', 'F08L03', 'F08L04', 'F08L05', 'F09L01', 'F09L03', 'F09L04', 'F09L05', 'F11L01', 'F11L02', 'F11L03', 'F11L04', 'F11L05', 'F01L05', 'F02L05', 'F03L01', 'F03L05', 'F04L05', 'F05L05', 'F06L05', 'F10L05']

def final_penalties(y, version):
    ch = {}
    for d in ('new_repositories', 'new_conditions'):
        meta = json.load(open(os.path.join(L.OUT, 'cv', f'cv_{y}_{d}.json')))
        ch[d] = [(f[version]['lz'], f[version]['lx']) for f in meta['folds']]

    def mode(vals):
        c = Counter(vals)
        top = max(c.values())
        return max((k for k, v in c.items() if v == top))
    best = (mode([a for a, b in ch['new_repositories']]), mode([b for a, b in ch['new_conditions']]))
    return (best, {d: [list(x) for x in v] for d, v in ch.items()})

def unit_weights(f, nt, lz, lx):
    w = f.weights()
    return np.concatenate([w[:nt] / np.sqrt(lz), w[nt:] / np.sqrt(lx)])

def ds_rates(par, I, version, task_T, setup_vals, cdat=None):
    mu, a, v, w = (par['mu_ds'], par['alpha_ds'], par['v'], par['w'])
    base = a + task_T @ v
    out = {'no cue': float(np.mean(mu + (1 - mu) * expit(base)))}
    if version in ('category', 'split'):
        for j, r in enumerate(I.names[version]):
            out[f'+1 {r}'] = float(np.mean(mu + (1 - mu) * expit(base + w[j])))
    if cdat is not None:
        for c, (T_, X_, ai_, n_) in cdat.items():
            e0 = par['alpha_all'][ai_] + T_ @ v
            sp = X_ @ w
            out[f'cond {c}'] = float(np.sum(n_ * (mu + (1 - mu) * expit(e0 + sp))) / n_.sum())
            out[f'cond0 {c}'] = float(np.sum(n_ * (mu + (1 - mu) * expit(e0))) / n_.sum())
            out[f'shift {c}'] = float(np.sum(n_ * sp) / n_.sum())
    return out

def task_spread(ctx, muv, eta, gainv, tidx, ntask):
    x = np.linspace(-7, 7, 301)
    lw = -0.5 * x ** 2
    lw = lw - logsumexp(lw)
    k, n = (ctx.k, ctx.n)

    def negll(par):
        s, sh = (np.exp(par[0]), par[1:])
        tot = 0.0
        T = np.zeros((ntask, len(x)))
        for lo in range(0, ctx.N, 2000):
            sl = slice(lo, min(lo + 2000, ctx.N))
            e = (eta[sl] + sh[ctx.ai[sl]])[:, None] + gainv[sl][:, None] * s * x[None, :]
            P = np.clip(muv[sl][:, None] + (1 - muv[sl][:, None]) * expit(e), 1e-12, 1 - 1e-12)
            ll = k[sl][:, None] * np.log(P) + (n[sl] - k[sl])[:, None] * np.log(1 - P)
            np.add.at(T, tidx[sl], ll)
        return -float(np.sum(logsumexp(T + lw[None, :], axis=1)))
    best = None
    for s0 in (0.5, 1.5):
        r = minimize(negll, np.r_[np.log(s0), np.zeros(ctx.A)], method='L-BFGS-B', bounds=[(np.log(0.01), np.log(8.0))] + [(-6, 6)] * ctx.A, options={'maxiter': 500})
        if best is None or r.fun < best.fun:
            best = r
    return (float(np.exp(best.x[0])), list(map(float, best.x[1:])), float(-best.fun), bool(best.success))

def main():
    y = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    boot_versions = ['category']
    od = os.path.join(L.OUT, 'final')
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    R = L.load_runs()
    rt = L.room_table(R, y)
    agents = rt[rt.enters].model.tolist()
    S = L.build_setups(R, y, agents)
    ctx, I = (L.Ctx(S), L.Inputs(S))
    nt = len(I.tnames)
    tasks = sorted(S.task_id.unique())
    Bt, mean, sd = L.task_constants()
    task_T = ((Bt - mean) / sd).reindex(tasks).values
    tidx = pd.Series(np.arange(len(tasks)), index=tasks)[S.task_id].values
    ds_main = ctx.groups.index(('DeepSeek', 'deepinfra'))
    w1 = np.ones(ctx.N)
    mu_all, fb = L.agent_mu(ctx, w1)
    muv = mu_all[ctx.mi]
    out = {'outcome': y, 'agents': agents, 'room': rt.to_dict('records'), 'models': ctx.models, 'groups': [list(g) for g in ctx.groups], 'setups': int(ctx.N), 'runs': int(ctx.n.sum()), 'events': int(ctx.k.sum()), 'repositories': int(S.repo.nunique()), 'conditions': int(S.condition.nunique()), 'tasks': len(tasks), 'mu': dict(zip(ctx.models, map(float, mu_all))), 'mu_clean_runs': {m: {'runs': float(ctx.cn[ctx.mi == i].sum()), 'events': float(ctx.ck[ctx.mi == i].sum())} for i, m in enumerate(ctx.models)}, 'task_constants': I.tconst, 'versions': {}}
    obs = S[S.model == 'DeepSeek']
    base_obs = obs[obs.condition == 'B00L00'].groupby('task_id')[['k', 'n']].sum()
    obs_rates = {}
    for c in SEL_CONDS:
        g = obs[obs.condition == c].groupby('task_id')[['k', 'n']].sum()
        if not len(g):
            continue
        b = base_obs.reindex(g.index)
        okb = b.n > 0
        wts = g.n[okb]
        obs_rates[c] = {'obs': float(g.k.sum() / g.n.sum()), 'runs': int(g.n.sum()), 'tasks': int(len(g)), 'matched_nocue': float(np.sum(wts * b.k[okb] / b.n[okb]) / np.sum(wts))}
    out['deepseek_observed'] = {'nocue_rate': float(base_obs.k.sum() / base_obs.n.sum()), 'conditions': obs_rates}
    for v in L.VERSIONS:
        (lz, lx), ch = final_penalties(y, v)
        Xs = I.setup_block(v, None)
        Xd = L.design(I.T, Xs, lz, lx, ctx.N)
        allidx = np.arange(ctx.N)
        f = L.fit(ctx, allidx, muv, Xd)
        w = unit_weights(f, nt, lz, lx)
        P = L.predict(f, ctx, allidx, muv, Xd)
        gains = f.gains()
        rec = {'penalty_task': lz, 'penalty_setup': lx, 'penalty_choices': ch, 'converged': bool(f.res.success), 'alpha': {f'{g[0]}|{g[1]}': float(a) for g, a in zip(ctx.groups, f.p['alpha'])}, 'gain': dict(zip(ctx.models, map(float, gains))), 'task_weights_per_sd': dict(zip(I.tnames, map(float, w[:nt]))), 'setup_names': I.names[v], 'setup_weights': dict(zip(I.names[v], map(float, w[nt:]))), 'loglik': float(L.ll_vec(ctx.k, ctx.n, P).sum())}
        dense = ctx.n >= 20
        lo, hi = (binom.ppf(0.025, ctx.n, P), binom.ppf(0.975, ctx.n, P))
        inside = (ctx.k >= lo) & (ctx.k <= hi)
        rec['in_sample_per_setup'] = {'setups_n20': int(dense.sum()), 'coverage': float(inside[dense].mean()) if dense.any() else None, 'mae_pp': float(100 * np.mean(np.abs(ctx.k[dense] / ctx.n[dense] - P[dense]))) if dense.any() else None, 'by_agent': {m: {'setups': int((dense & (ctx.mi == i)).sum()), 'coverage': float(inside[dense & (ctx.mi == i)].mean()) if (dense & (ctx.mi == i)).any() else None, 'mae_pp': float(100 * np.mean(np.abs(ctx.k[dense & (ctx.mi == i)] / ctx.n[dense & (ctx.mi == i)] - P[dense & (ctx.mi == i)]))) if (dense & (ctx.mi == i)).any() else None} for i, m in enumerate(ctx.models)}}
        eta = f.p['alpha'][ctx.ai] + gains[ctx.mi] * (Xd @ f.weights())
        s_task, shifts, llm, okm = task_spread(ctx, muv, eta, gains[ctx.mi], tidx, len(tasks))
        a_ds = f.p['alpha'][ds_main] + shifts[ds_main]
        mu_ds = mu_all[ctx.gi]
        z = np.array([-1.96, 0, 1.96])
        rec['task_spread'] = {'sd_logodds_deepseek_scale': s_task, 'alpha_shifts': shifts, 'marginal_loglik': llm, 'converged': okm, 'deepseek_nocue_average_task_rate_range95': list(map(float, mu_ds + (1 - mu_ds) * expit(a_ds + z * s_task)))}
        setup_vals = None
        cdat = {}
        for c in SEL_CONDS:
            ii = np.where((S.model.values == 'DeepSeek') & (S.condition.values == c))[0]
            if len(ii):
                cdat[c] = (I.T[ii], Xs[ii], ctx.ai[ii], ctx.n[ii])
        par = {'mu_ds': mu_ds, 'alpha_ds': f.p['alpha'][ds_main], 'v': w[:nt], 'w': w[nt:], 'alpha_all': f.p['alpha']}
        rates = ds_rates(par, I, v, task_T, setup_vals, cdat)
        rec['deepseek_rates_point'] = rates
        out['versions'][v] = rec
        print(f'{y} {v}: penalties {lz:g}/{lx:g}, loglik {rec['loglik']:.1f}, gains {np.round(gains, 3)}, task sd {s_task:.2f} ({time.time() - t0:.0f}s)', flush=True)
        print('   setup weights:', {k_: round(x_, 3) for k_, x_ in rec['setup_weights'].items()}, flush=True)
        if v not in boot_versions:
            continue
        ru, ri = np.unique(S.repo.values, return_inverse=True)
        cu, ci = np.unique(S.condition.values, return_inverse=True)
        bres = {}
        for kind in ('two_way', 'repo'):
            rng = np.random.default_rng(20261110 if kind == 'two_way' else 20261111)
            A, G, W, RT, MU = ([], [], [], [], [])
            theta0 = f.theta.copy()
            for b in range(B):
                wr = np.bincount(rng.integers(0, len(ru), len(ru)), minlength=len(ru)).astype(float)
                wc = np.bincount(rng.integers(0, len(cu), len(cu)), minlength=len(cu)).astype(float) if kind == 'two_way' else np.ones(len(cu))
                mult = wr[ri] * wc[ci]
                idx = np.where(mult > 0)[0]
                mu_b, _ = L.agent_mu(ctx, mult)
                muv_b = mu_b[ctx.mi]
                Xs_b = Xs
                Xd_b = L.design(I.T, Xs_b, lz, lx, ctx.N)
                fb_ = L.fit(ctx, idx, muv_b, Xd_b, mult=mult[idx], theta0=theta0)
                wb = unit_weights(fb_, nt, lz, lx)
                A.append(fb_.p['alpha'].copy())
                G.append(fb_.gains().copy())
                W.append(wb)
                MU.append(mu_b)
                cdat_b = cdat
                RT.append(ds_rates({'mu_ds': mu_b[ctx.gi], 'alpha_ds': fb_.p['alpha'][ds_main], 'v': wb[:nt], 'w': wb[nt:], 'alpha_all': fb_.p['alpha']}, I, v, task_T, setup_vals, cdat_b))
            A, G, W, MU = map(np.array, (A, G, W, MU))
            RTd = pd.DataFrame(RT)
            np.savez_compressed(os.path.join(od, f'final_boot_{y}_{v}_{kind}.npz'), alpha=A, gain=G, w=W, mu=MU, rates=RTd.values, rate_names=np.array(RTd.columns))
            q = lambda arr: (np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0))
            wlo, whi = q(W)
            glo, ghi = q(G)
            rlo, rhi = q(RTd.values)
            bres[kind] = {'replicates': B, 'weights_lo': dict(zip(I.tnames + I.names[v], map(float, wlo))), 'weights_hi': dict(zip(I.tnames + I.names[v], map(float, whi))), 'share_weight_positive': dict(zip(I.tnames + I.names[v], map(float, (W > 0).mean(0)))), 'gain_lo': dict(zip(ctx.models, map(float, glo))), 'gain_hi': dict(zip(ctx.models, map(float, ghi))), 'rates_lo': dict(zip(RTd.columns, map(float, rlo))), 'rates_hi': dict(zip(RTd.columns, map(float, rhi))), 'mu_lo': dict(zip(ctx.models, map(float, q(MU)[0]))), 'mu_hi': dict(zip(ctx.models, map(float, q(MU)[1])))}
            print(f'   bootstrap {kind}: {B} replicates ({time.time() - t0:.0f}s)', flush=True)
        rec['bootstrap'] = bres
        tab = pd.DataFrame({'input': I.tnames + I.names[v], 'kind': ['task (per SD)'] * nt + ['setup role (per unit)'] * len(I.names[v]), 'weight': w})
        for kind in bres:
            tab[f'lo_{kind}'] = tab.input.map(bres[kind]['weights_lo'])
            tab[f'hi_{kind}'] = tab.input.map(bres[kind]['weights_hi'])
        tab.to_csv(os.path.join(od, f'final_weights_{y}_{v}.csv'), index=False)
        rt_ = pd.DataFrame({'quantity': list(rates.keys()), 'point': list(rates.values())})
        for kind in bres:
            rt_[f'lo_{kind}'] = rt_.quantity.map(bres[kind]['rates_lo'])
            rt_[f'hi_{kind}'] = rt_.quantity.map(bres[kind]['rates_hi'])
        rt_.to_csv(os.path.join(od, f'final_rates_{y}_{v}.csv'), index=False)
        json.dump(out, open(os.path.join(od, f'final_{y}.json'), 'w'), indent=1, default=float)
    out['fit_log'] = dict(L.FIT_LOG)
    json.dump(out, open(os.path.join(od, f'final_{y}.json'), 'w'), indent=1, default=float)
    print('done', f'{time.time() - t0:.0f}s', L.FIT_LOG)

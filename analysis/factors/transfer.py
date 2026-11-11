"""Forecast outcomes on new tasks with no-cue calibration."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
from scipy.special import expit,logsumexp
from scipy.optimize import minimize_scalar
import formula_lib as L
import matched as WP3
import forecast_lib as F
from matched import expand_counts
from types import SimpleNamespace
AGENTS=['DeepSeek','Qwen','GLM']
COVER=['E01','F07L05','F11L01','F11L02','F11L04','F15L01','F16L03']
P1_BITS={'B00L00':{},'E01':{'sys_eval_statement':1},'F07L05':{'headings6':1,'bench_suffix_heading':1,'bench_word_headings':5},'F11L01':{'ps_heading':1},'F11L02':{'ps_heading':1,'bname_request':1},'F11L04':{'ps_heading':1,'bname_request':1,'bname_file':1,'bname_path':1,'ws_benchmarks':1},'F16L03':{'ps_heading':1,'sys_eval_statement':1},'F15L01':{'ps_heading':2}}
W=SimpleNamespace(P1_BITS=P1_BITS)
NBOOT,SEED=2000,20261107
CODING='known'
XG=np.linspace(-7,7,301)
LWG=-.5*XG**2
LWG=LWG-logsumexp(LWG)
RECORDS=[]

def emit(tag,obj):
    if tag=='T_block':
        RECORDS.append(obj)
        print(obj['agent'],obj['block'],obj['real_minus_invented'],flush=True)

def load_fit(outcome,fit,agent):
    return F.load_fit(outcome,agent)

def task_Z(tasks,tn):
    return F.task_Z(tasks,tn,'fresh')

def load_prereg(agent):
    f=pd.read_parquet(DATA/'transfer_counts.parquet')
    f=f[f.model.eq(agent)&f.outcome.eq(OUTCOME)]
    return expand_counts(f,OUTCOME),None
def cell_units(conds, sn):
    out = {}
    Rm = L.role_matrix(L.ROLE_OF, L.ROLES)
    for c in conds:
        vec = np.zeros(len(L.BIT_NAMES))
        for b, val in W.P1_BITS[c].items():
            vec[L.BIT_NAMES.index(b)] = val
        u = pd.Series(vec @ Rm, index=L.ROLES)
        extra = [r for r in u.index if u[r] != 0 and r not in sn]
        assert not extra, (c, extra)
        out[c] = u.reindex(sn).values
    return out

def calibrate(kt, nt_, mu, g, tau, s, mode):

    def P_of(e):
        return np.clip(mu + (1 - mu) * expit(e), 1e-09, 1 - 1e-09)
    if mode == 'task':

        def negll(a):
            P = P_of(a + g * (tau[:, None] + s * XG[None, :]))
            ll = kt[:, None] * np.log(P) + (nt_ - kt)[:, None] * np.log(1 - P)
            return -float(logsumexp(ll + LWG[None, :], axis=1).sum())
    else:

        def negll(a):
            P = P_of(a + g * tau)
            return -float((kt * np.log(P) + (nt_ - kt) * np.log(1 - P)).sum())
    a = minimize_scalar(negll, bounds=(-20, 20), method='bounded', options={'xatol': 1e-07}).x
    if mode == 'task':
        P = P_of(a + g * (tau[:, None] + s * XG[None, :]))
        ll = kt[:, None] * np.log(P) + (nt_ - kt)[:, None] * np.log(1 - P) + LWG[None, :]
        post = np.exp(ll - logsumexp(ll, axis=1)[:, None])
    else:
        post = None
    return (float(a), post)

def run_probs(a, post_t, tau_t, xw, mu, g, s, mode):
    if mode == 'task':
        e = a + g * ((tau_t + xw)[:, None] + s * XG[None, :])
        return (post_t * (mu + (1 - mu) * expit(e))).sum(1)
    return mu + (1 - mu) * expit(a + g * (tau_t + xw))

def cell_ll(k, n, p):
    p = np.clip(p, 1e-09, 1 - 1e-09)
    return k * np.log(p) + (n - k) * np.log(1 - p)

def predict_block(runs, y, par, units, mode):
    tasks = TASK_UNIVERSE[(runs.model.iloc[0], runs['set'].iloc[0], y)]
    ti = {t: i for i, t in enumerate(tasks)}
    Z = task_Z(tasks, par['tn'])
    tau = Z @ par['v']
    nc = runs[runs.condition.eq('B00L00')].groupby('task_id')[y].agg(['sum', 'count']).reindex(tasks)
    has = nc['count'].fillna(0).values > 0
    kt, nt_ = (nc['sum'].fillna(0).values[has], nc['count'].fillna(0).values[has])
    a, post = calibrate(kt, nt_, par['mu'], par['gain'], tau[has], par['s'], mode)
    POST = np.tile(np.exp(LWG), (len(tasks), 1))
    if post is not None:
        POST[has] = post
    cue = runs[runs.condition.isin(units)].copy()
    idx = cue.task_id.map(ti).values
    xw = np.array([units[c] @ par['w'] for c in cue.condition])
    cue['p'] = run_probs(a, POST[idx] if mode == 'task' else None, tau[idx], xw, par['mu'], par['gain'], par['s'], mode)
    cue['p0'] = run_probs(a, POST[idx] if mode == 'task' else None, tau[idx], np.zeros(len(cue)), par['mu'], par['gain'], par['s'], mode)
    return (cue, a, tasks)

def analyse(outcome, fit, mode):
    res = []
    for agent in AGENTS:
        fr, fsha = load_prereg(agent)
        try:
            par, draws = load_fit(outcome, fit, agent)
        except (FileNotFoundError, ValueError) as e:
            emit('T_skip', {'agent': agent, 'outcome': outcome, 'fit': fit, 'reason': str(e)})
            continue
        for block in ('main', 'heldout'):
            runs = fr[fr.set.eq(block) & fr[outcome].notna()].copy()
            conds = [c for c in COVER if c in set(runs.condition)]
            if not conds or not runs.condition.eq('B00L00').any():
                continue
            units = cell_units(conds, par['sn'])
            cue, a, tasks = predict_block(runs, outcome, par, units, mode)
            Mb = WP3.boot_matrix(len(tasks), NBOOT, seed=SEED)
            ti = {t: i for i, t in enumerate(tasks)}
            nc = runs[runs.condition.eq('B00L00')]
            k0t = nc.groupby('task_id')[outcome].sum().reindex(tasks).fillna(0).values
            n0t = nc.groupby('task_id')[outcome].count().reindex(tasks).fillna(0).values
            floor = k0t.sum() / n0t.sum()
            cells, Lf, Lp, Lc = ([], 0.0, 0.0, 0.0)
            agg = {}
            for c in conds:
                cc = cue[cue.condition.eq(c)]
                kt = cc.groupby('task_id')[outcome].sum().reindex(tasks).fillna(0).values
                ntt = cc.groupby('task_id')[outcome].count().reindex(tasks).fillna(0).values
                pt = cc.groupby('task_id').p.sum().reindex(tasks).fillna(0).values
                p0t = cc.groupby('task_id').p0.sum().reindex(tasks).fillna(0).values
                k, n = (kt.sum(), ntt.sum())
                pred, pred0, obs = (pt.sum() / n, p0t.sum() / n, k / n)
                ob = Mb @ kt / np.where(Mb @ ntt > 0, Mb @ ntt, np.nan)
                m = WP3.Matched(kt, ntt, k0t, n0t, Mb, weights='r08')
                sm = m.summary()
                cells.append({'condition': c, 'runs': int(n), 'tasks': int((ntt > 0).sum()), 'obs': obs, 'obs_lo': float(np.nanpercentile(ob, 2.5)), 'obs_hi': float(np.nanpercentile(ob, 97.5)), 'pred': pred, 'pred_nocue_same_runs': pred0, 'pred_diff_pp': 100 * (pred - pred0), 'obs_diff_pp': 100 * m.rd, 'obs_diff_lo': sm['rd_lo'], 'obs_diff_hi': sm['rd_hi'], 'obs_nocue_matched': m.p0})
                Lf += cell_ll(k, n, floor)
                Lp += cell_ll(k, n, pred)
                Lc += cell_ll(k, n, obs)
                agg[c] = (kt, ntt, pt)
            frac = (Lp - Lf) / (Lc - Lf) if Lc > Lf else np.nan
            fb = []
            for b in range(NBOOT):
                mb = Mb[b]
                fl = mb @ k0t / max(mb @ n0t, 1e-09)
                lf = lp = lc = 0.0
                for c, (kt, ntt, pt) in agg.items():
                    k, n = (mb @ kt, mb @ ntt)
                    if n <= 0:
                        continue
                    lf += cell_ll(k, n, fl)
                    lp += cell_ll(k, n, mb @ pt / n)
                    lc += cell_ll(k, n, k / n)
                fb.append((lp - lf) / (lc - lf) if lc > lf else np.nan)
            ri = None
            if 'F11L02' in agg and 'F15L01' in agg:
                k1, n1, p1 = agg['F11L02']
                k2, n2, p2 = agg['F15L01']
                mm = WP3.Matched(k1, n1, k2, n2, Mb, weights='r08')
                s2 = mm.summary()
                ri = {'obs_pp': 100 * mm.rd, 'obs_lo': s2['rd_lo'], 'obs_hi': s2['rd_hi'], 'pred_pp': 100 * (p1.sum() / n1.sum() - p2.sum() / n2.sum())}
            rec = {'coding': CODING, 'outcome': outcome, 'fit': fit, 'calibration': mode, 'agent': agent, 'block': block, 'tasks': len(tasks), 'alpha_calibrated': a, 'nocue_obs': floor, 'nocue_runs': int(n0t.sum()), 'cells': cells, 'real_minus_invented': ri, 'frac_gap_closed': frac, 'frac_lo': float(np.nanpercentile(fb, 2.5)), 'frac_hi': float(np.nanpercentile(fb, 97.5))}
            if fit == 'pooled' and mode == 'task':
                pr = {c: [] for c in conds}
                pdiff = {c: [] for c in conds}
                ri_b = []
                for d in range(len(draws['gain'])):
                    pard = dict(par, mu=float(draws['mu'][d]), gain=float(draws['gain'][d]), v=draws['v'][d], w=draws['w'][d])
                    cue_d, _, _ = predict_block(runs, outcome, pard, units, mode)
                    for c in conds:
                        cc = cue_d[cue_d.condition.eq(c)]
                        pr[c].append(cc.p.mean())
                        pdiff[c].append(100 * (cc.p.mean() - cc.p0.mean()))
                    if ri is not None:
                        ri_b.append(100 * (cue_d[cue_d.condition.eq('F11L02')].p.mean() - cue_d[cue_d.condition.eq('F15L01')].p.mean()))
                for cell in rec['cells']:
                    c = cell['condition']
                    cell['pred_lo'], cell['pred_hi'] = np.percentile(pr[c], [2.5, 97.5]).tolist()
                    cell['pred_diff_lo'], cell['pred_diff_hi'] = np.percentile(pdiff[c], [2.5, 97.5]).tolist()
                if ri is not None:
                    ri['pred_lo'], ri['pred_hi'] = np.percentile(ri_b, [2.5, 97.5]).tolist()
                rec['refit_draws'] = len(draws['gain'])
            emit('T_block', rec)
    emit('done', {'outcome': outcome, 'fit': fit, 'calibration': mode, 'coding': CODING})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--outcome',choices=['G_rmask','G_tests'],default='G_rmask')
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=RESULTS/'transfer')
    args=parser.parse_args();DATA=args.data;F.DATA=DATA;L.DATA=DATA
    OUTCOME=args.outcome
    if OUTCOME=='G_tests':AGENTS=['DeepSeek']
    universe=pd.read_parquet(DATA/'transfer_tasks.parquet')
    TASK_UNIVERSE={key:sorted(g.task_id.unique()) for key,g in universe.groupby(['model','set','outcome'])}
    L.known_coding()
    analyse(OUTCOME,'pooled','task')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/f'{OUTCOME}.json').write_text(json.dumps(RECORDS,indent=2,default=lambda x:x.item())+'\n')

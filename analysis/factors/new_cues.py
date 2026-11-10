"""Evaluate forecasts and no-cue recalibration for new setups."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
import new_cue_lib as N
from matched import expand_counts
YCOL={'G_rmask':'y_rmask','G_tests':'y_tests'}
NB=2000
def cell_ll(k, n, p):
    p = np.clip(p, 1e-09, 1 - 1e-09)
    return k * np.log(p) + (n - k) * np.log(1 - p)

def run(new, R, agent, outcome, prior_mode):
    par, draws = N.load_fit(outcome, agent)
    X = new[new.agent.eq(agent) & new[YCOL[outcome]].notna()]
    tasks = sorted(X.task_id.unique())
    cells = [c for c in N.CELLS if c in set(X.cell)]
    K = X.pivot_table(index='task_id', columns='cell', values=YCOL[outcome], aggfunc='sum').reindex(index=tasks, columns=cells).fillna(0).values
    M = X.pivot_table(index='task_id', columns='cell', values=YCOL[outcome], aggfunc='count').reindex(index=tasks, columns=cells).fillna(0).values
    kmain, nmain = N.main_nocue(R, agent, outcome, tasks)
    P = []
    for d in range(-1, len(draws['gain'])):
        p = par if d < 0 else dict(par, mu=float(draws['mu'][d]), gain=float(draws['gain'][d]), v=draws['v'][d], w=draws['w'][d])
        tau = N.task_tau(tasks, par['tn'], p['v'])
        if prior_mode == 'main':
            _, prior = N.main_prior(kmain, nmain, p, tau)
        else:
            prior = np.tile(N.LWG, (len(tasks), 1))
        a, lp = N.calibrate(K[:, 0], M[:, 0], p['mu'], p['gain'], tau, p['s'], prior)
        P.append(N.cell_probs(a, lp, p['mu'], p['gain'], tau, p['s'], p['w'], par['sn'], cells))
    P = np.array(P)
    rng = np.random.default_rng(N.SEED)
    runs_per = int(round(M.sum() / (len(tasks) * len(cells))))
    summ = N.summarise(P, max(runs_per, 1), rng)
    Mb = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=NB).astype(float)
    out = {'agent': agent, 'outcome': outcome, 'prior': prior_mode, 'issues': len(tasks), 'runs': int(M.sum()), 'cells': {}}
    Lf = Lp = Lc = 0.0
    k0, n0 = (K[:, 0], M[:, 0])
    for j, c in enumerate(cells):
        k, n = (K[:, j], M[:, j])
        obs = k.sum() / n.sum()
        ob = Mb @ k / (Mb @ n)
        rec = dict(summ[j])
        rec.update({'obs': 100 * obs, 'obs_lo': 100 * np.percentile(ob, 2.5), 'obs_hi': 100 * np.percentile(ob, 97.5), 'runs': int(n.sum()), 'obs_inside_pi': bool(rec['obs_pi_lo'] <= 100 * obs <= rec['obs_pi_hi'])})
        if j > 0:
            od = 100 * (obs - k0.sum() / n0.sum())
            odb = 100 * (Mb @ k / (Mb @ n) - Mb @ k0 / (Mb @ n0))
            rec.update({'obs_diff': od, 'obs_diff_lo': np.percentile(odb, 2.5), 'obs_diff_hi': np.percentile(odb, 97.5), 'diff_inside_pi': bool(rec['diff_obs_pi_lo'] <= od <= rec['diff_obs_pi_hi'])})
            pred = (P[0, :, j] * n).sum() / n.sum()
            Lf += cell_ll(k.sum(), n.sum(), k0.sum() / n0.sum())
            Lp += cell_ll(k.sum(), n.sum(), pred)
            Lc += cell_ll(k.sum(), n.sum(), obs)
        out['cells'][c] = {kk: round(float(v), 2) if not isinstance(v, bool) else v for kk, v in rec.items()}
    out['share_gap_closed'] = (Lp - Lf) / (Lc - Lf) if Lc > Lf else None
    fb = []
    for b in range(NB):
        mb = Mb[b]
        f0 = mb @ k0 / (mb @ n0)
        lf = lp = lc = 0.0
        for j in range(1, len(cells)):
            k, n = (mb @ K[:, j], mb @ M[:, j])
            lf += cell_ll(k, n, f0)
            lp += cell_ll(k, n, mb @ (P[0, :, j] * M[:, j]) / n)
            lc += cell_ll(k, n, k / n)
        fb.append((lp - lf) / (lc - lf) if lc > lf else np.nan)
    out['share_lo'], out['share_hi'] = (float(np.nanpercentile(fb, 2.5)), float(np.nanpercentile(fb, 97.5)))
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=RESULTS/'new_cues.json')
    args=parser.parse_args();DATA=args.data;N.DATA=DATA
    f=pd.read_parquet(DATA/'new_cue_counts.parquet')
    new=expand_counts(f[f.outcome.eq('y_rmask')],'y_rmask')
    R=N.load_runs();records=[]
    for agent in sorted(new.agent.unique()):
        par,_=N.load_fit('G_rmask',agent)
        tasks=sorted(new[new.agent.eq(agent)].task_id.unique())
        k,n=N.main_nocue(R,agent,'G_rmask',tasks)
        tau=N.task_tau(tasks,par['tn'],par['v'])
        alpha,prior=N.main_prior(k,n,par,tau)
        cells=[c for c in N.CELLS if c in set(new[new.agent.eq(agent)].cell)]
        pred=N.cell_probs(alpha,prior,par['mu'],par['gain'],tau,par['s'],par['w'],par['sn'],cells).mean(0)
        record=run(new,R,agent,'G_rmask','main')
        record['before_run_predictions']={c:float(100*p) for c,p in zip(cells,pred)}
        record['before_run_differences']={c:float(100*(p-pred[0])) for c,p in zip(cells,pred)}
        records.append(record)
        print(agent,record['share_gap_closed'],record['before_run_differences'],flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(records,indent=2,default=lambda x:x.item())+'\n')

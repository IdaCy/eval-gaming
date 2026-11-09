"""Reproduce task wording, recall, date, and task-group comparisons."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
from scipy.stats import spearmanr
from matched import Matched,boot_matrix,task_counts,count_arrays,ci
B=2000
def spear(x, y):
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).correlation)

def corr(label, part, xcol, ycol):
    s = part[part[xcol].notna() & part[ycol].notna()]
    n = len(s)
    x, yv = (s[xcol].to_numpy(float), s[ycol].to_numpy(float))
    r2 = np.random.default_rng(20261109 + 1)
    idx = r2.integers(0, n, size=(B, n))
    pt = spear(x, yv)
    dr = np.array([spear(x[i], yv[i]) for i in idx])
    ok = np.isfinite(dr)
    return {'analysis': label, 'x': xcol, 'y': ycol, 'issues': n, 'rho': pt, 'lo': float(np.percentile(dr[ok], 2.5)), 'hi': float(np.percentile(dr[ok], 97.5)), 'valid_draws': int(ok.sum())}

def compare(f,tasks,cue,control,matrix,weights='mh',condition='condition'):
    return Matched(*count_arrays(f[f[condition].eq(cue)],tasks),*count_arrays(f[f[condition].eq(control)],tasks),matrix,weights=weights)

def calculate():
    results={}
    f=pd.read_parquet(DATA/'reword_counts.parquet')
    tasks=f.sort_values('order').task_id.drop_duplicates().tolist()
    rng=np.random.default_rng(20261105);M=np.zeros((B,len(tasks)))
    for b in range(B):np.add.at(M[b],rng.integers(0,len(tasks),len(tasks)),1)
    results['whole_issue_rewording']=compare(f,tasks,'P','O',M,condition='arm').summary()
    f=pd.read_parquet(DATA/'factor_runs.parquet')
    main=f[f.dataset.eq('paper_main')&f.model.eq('DeepSeek')].copy()
    tasks=sorted(main.task_id.unique());M=boot_matrix(len(tasks),B,20261107)
    conditions=sorted(c for c in main.condition.unique() if c[:3] in ['F01','F02','F03','F04','F06'] and c!='F03L00')
    effects=[];draws=[]
    for condition in conditions:
        m=Matched(*task_counts(main[main.condition.eq(condition)],'notests',tasks),*task_counts(main[main.condition.eq('B00L00')],'notests',tasks),M,weights='r08')
        effects.append(m.rd);draws.append(m.rd_b)
    lo,hi=ci(np.column_stack(draws).mean(1))
    results['small_rewordings']={'conditions':len(conditions),'estimate':100*float(np.mean(effects)),'lo':100*lo,'hi':100*hi}
    d=pd.read_parquet(DATA/'date_counts.parquet');tasks=sorted(d.task_id.unique())
    m=compare(d,tasks,'present_date','original_date',boot_matrix(len(tasks),B,20261105),condition='date_variant')
    se=float(np.std(m.rd_b,ddof=1))
    results['commit_date']={**m.summary(),'normal_lo':100*(m.rd-1.96*se),'normal_hi':100*(m.rd+1.96*se)}
    r=pd.read_parquet(DATA/'recall_scores.parquet').sort_values('order')
    levels=main[main.condition.eq('B00L00')].groupby('task_id').judged_old.mean()
    r['judged']=r.task_id.map(levels)
    results['recall']=corr('recall versus judged gaming',r,'q2_recall','judged')
    f=f[f.dataset.eq('swefresh')]
    ids=set(f.loc[pd.to_datetime(f.t_issue_created,utc=True)>pd.Timestamp('2026-10-17T00:00:00Z'),'task_id'])
    assert len(ids)==41
    d=pd.read_parquet(DATA/'transfer_counts.parquet')
    d=d[d.model.eq('DeepSeek')&d.outcome.eq('G_tests')&d.task_id.isin(ids)]
    tasks=sorted(d.task_id.unique());M=boot_matrix(len(tasks),B,20261188)
    labels=d[['task_id','set']].drop_duplicates().set_index('task_id')['set'].reindex(tasks)
    for label,seed in [('heldout',20261188),('main',20261189)]:
        idx=np.flatnonzero(labels.eq(label).to_numpy());M[:,idx]=boot_matrix(len(idx),B,seed)
    results['recent_issues']=compare(d,tasks,'F11L02','B00L00',M).summary()
    d=pd.read_parquet(DATA/'task_group_counts.parquet');groups=[]
    for (group,outcome),g in d.groupby(['group','outcome']):
        groups.append({'group':group,'outcome':outcome,'rate':100*float((g.k/g.n).mean()),'tasks':len(g),'runs':int(g.n.sum()),'events':int(g.k.sum())})
    results['task_groups']=groups
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=RESULTS/'additional_factors.json')
    args=parser.parse_args();DATA=args.data
    result=calculate();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

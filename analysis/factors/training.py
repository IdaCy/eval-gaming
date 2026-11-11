"""Checkpoint test omission and belief frequencies during training."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
from matched import expand_counts
def matched(g, cue, col='omit'):
    c = g[g.arm.isin(cue)].groupby('task')[col].agg(['mean', 'count'])
    n = g[g.arm == 'B00L00'].groupby('task')[col].mean()
    both = c.index.intersection(n.index)
    rc, rn, wc = (np.full(len(TASKS), np.nan) for _ in range(3))
    for t in both:
        rc[ti[t]], rn[ti[t]], wc[ti[t]] = (c.loc[t, 'mean'], n.loc[t], c.loc[t, 'count'])
    return (rc, rn, wc)

def eff_from(rc, rn, wc):
    ok = ~np.isnan(rc)
    wt = np.where(ok, wc, 0.0)
    d = np.where(ok, rc - rn, 0.0)
    pc, pn = (np.where(ok, rc, 0.0), np.where(ok, rn, 0.0))
    pt = (wt * d).sum() / wt.sum()
    bw = W_ * wt
    bs = (bw * d).sum(1) / bw.sum(1)
    clip = lambda p: np.clip(p, 0.01, 0.99)
    lo_pt = np.log(clip((wt * pc).sum() / wt.sum()) / (1 - clip((wt * pc).sum() / wt.sum()))) - np.log(clip((wt * pn).sum() / wt.sum()) / (1 - clip((wt * pn).sum() / wt.sum())))
    bc, bn = ((bw * pc).sum(1) / bw.sum(1), (bw * pn).sum(1) / bw.sum(1))
    blo = np.log(clip(bc) / (1 - clip(bc))) - np.log(clip(bn) / (1 - clip(bn)))
    return (pt, bs, lo_pt, blo, int(ok.sum()), (wt * pc).sum() / wt.sum(), (wt * pn).sum() / wt.sum())

def calculate():
    global TASKS,ti,W_
    f=pd.read_parquet(DATA/'checkpoint_counts.parquet')
    TASKS=sorted(f.task_id.unique());ti={t:i for i,t in enumerate(TASKS)}
    rng=np.random.default_rng(1108);idx=rng.integers(0,len(TASKS),size=(2000,len(TASKS)))
    W_=np.zeros((2000,len(TASKS)))
    for b in range(2000):np.add.at(W_[b],idx[b],1)
    runs=expand_counts(f,'omit').rename(columns={'task_id':'task'})
    rows=[];boot={}
    for cue in ['F11L03','F09L05']:
        for (lineage,ck),g in runs.groupby(['lineage','ck']):
            pt,bs,lp,lb,nt,pc,pn=eff_from(*matched(g,[cue]))
            rows.append({'cue':cue,'lineage':lineage,'checkpoint':ck,'effect':100*pt,'lo':100*np.percentile(bs,2.5),'hi':100*np.percentile(bs,97.5),'tasks':nt})
            boot[(cue,lineage,ck)]=bs
    checkpoint=pd.DataFrame(rows);pooled=[]
    for cue in ['F11L03','F09L05']:
        for lineage in ['tmax9b','tmax4b','tmax27b','all']:
            d=checkpoint[checkpoint.cue.eq(cue)&(checkpoint.lineage.eq(lineage) if lineage!='all' else True)]
            b=np.mean([boot[(r.cue,r.lineage,r.checkpoint)] for r in d.itertuples()],axis=0)
            pooled.append({'cue':cue,'lineage':lineage,'checkpoints':len(d),'effect':float(d.effect.mean()),'lo':float(100*np.percentile(b,2.5)),'hi':float(100*np.percentile(b,97.5))})
    b=pd.read_parquet(DATA/'checkpoint_beliefs.parquet');b=b[b.arm.eq('F11L03')].groupby('outcome')[['k','n']].sum().reset_index();b['rate']=b.k/b.n
    t=pd.read_parquet(DATA/'training_beliefs.parquet');t['rate']=100*t.k/t.n
    a=t[t.outcome.eq('bp_grader_or_checker')];e=t[t.outcome.eq('b_evaluation_of_the_ai')]
    records={'checkpoint_effects':pooled,'hidden_test_beliefs':b.to_dict('records'),'training_start':float(a[a.bin.eq(0)].rate.iloc[0]),'training_200_239':float(100*a[a.bin.isin([200,220])].k.sum()/a[a.bin.isin([200,220])].n.sum()),'training_peak':float(a.rate.max()),'strict_evaluation_peak':float(e.rate.max())}
    return checkpoint,t,records

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--output',type=Path,default=RESULTS);args=parser.parse_args();DATA=args.data
    c,t,result=calculate();args.output.mkdir(parents=True,exist_ok=True)
    c.to_csv(args.output/'checkpoint_effects.csv',index=False);t.to_csv(args.output/'training_beliefs.csv',index=False)
    (args.output/'training.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

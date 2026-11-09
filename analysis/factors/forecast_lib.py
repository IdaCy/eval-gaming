"""Load validation coefficients and standardized task ratings."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
RESULTS = OUTPUT
import formula_lib as L

def load_fit(outcome,agent):
    f=pd.read_parquet(DATA/'validation_fit_draws.parquet')
    f=f[f.outcome.eq(outcome)&f.model.eq(agent)].sort_values('draw')
    p=f[f.draw.eq(-1)].iloc[0]
    tn=list(L.RATINGS)
    sn=[c for c in f.columns if c not in ['model','outcome','draw','mu','gain','task_sd']+tn and pd.notna(p[c])]
    d=f[f.draw.ge(0)]
    par={'mu':float(p.mu),'gain':float(p.gain),'v':p[tn].to_numpy(float),'w':p[sn].to_numpy(float),'s':float(p.task_sd),'tn':tn,'sn':sn}
    draws={'mu':d.mu.to_numpy(float),'gain':d.gain.to_numpy(float),'v':d[tn].to_numpy(float),'w':d[sn].to_numpy(float)}
    return par,draws

def task_Z(tasks,tn,corpus):
    L.DATA=DATA
    _,mean,sd=L.task_constants()
    b=pd.read_parquet(DATA/'task_ratings.parquet')
    b=b[b.corpus.eq(corpus)].set_index('task_id')[tn].astype(float)
    z=((b-mean[tn])/sd[tn]).reindex(tasks)
    assert not z.isna().any().any()
    return z.to_numpy(float)

"""Apply the frozen belief-shift rule and score the forward-test outcomes."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from answer_parser import parse

CELLS=["C1","C2","C3","C4","C5","C6"]
ORDER=["C3","C2","C4","C5","C6"]
NAMES={"C3":"Benchmark name","C2":"Benchmark name in system prompt",
       "C4":"Invented benchmark name as the first line","C5":"Invented benchmark name in system prompt",
       "C6":"Tags around the request"}


def matrices(data,tasks):
    sums=data.pivot(index="task",columns="cell",values="sum").reindex(index=tasks,columns=CELLS).fillna(0).to_numpy(float)
    counts=data.pivot(index="task",columns="cell",values="count").reindex(index=tasks,columns=CELLS).fillna(0).to_numpy(float)
    with np.errstate(invalid="ignore",divide="ignore"):
        means=np.where(counts>0,sums/counts,np.nan)
    return sums,counts,means


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i])
    result=[0.0]*len(values)
    i=0
    while i<len(order):
        j=i
        while j+1<len(order) and values[order[j+1]]==values[order[i]]:
            j+=1
        for k in range(i,j+1):result[order[k]]=(i+j)/2
        i=j+1
    return result


def rank_corr(a,b):
    return float(np.corrcoef(np.array(ranks(a),float),np.array(ranks(b),float))[0,1])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=pd.read_csv(args.data/"forward_task_order.csv").sort_values("order").task.tolist()
    draws=np.random.default_rng(20261110).integers(0,len(tasks),size=(2000,len(tasks)))
    answers=pd.read_csv(args.data/"forward_belief_answers.csv")
    parsed=[parse(text) for text in answers.answer]
    if any(status!="ok" for _,_,status in parsed):
        raise ValueError("invalid probability answer")
    answers["sum"]=[p*n for (p,_,_),n in zip(parsed,answers["count"])]
    belief_sums=answers.groupby(["task","cell"])[["sum","count"]].sum().reset_index()
    _,_,belief=matrices(belief_sums,tasks)
    prediction={}
    for cell in ORDER:
        j=CELLS.index(cell)
        delta=100*(belief[:,j]-belief[:,0])
        boot=np.nanmean(delta[draws],axis=1)
        shift,lo,hi=float(np.nanmean(delta)),float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))
        prediction[cell]=dict(shift=shift,lower=lo,upper=hi,flagged=shift<=-5 and hi<0)
    (args.output/"forward_prediction.json").write_text(json.dumps(prediction,indent=2)+"\n")
    testdata=pd.read_csv(args.data/"forward_tests_sums.csv")
    judgedata=pd.read_csv(args.data/"forward_judged_sums.csv")
    states=pd.read_csv(args.data/"forward_judge_rng.csv",dtype={"state":str,"inc":str})
    summaries={}
    for snapshot in ["earlier","paper"]:
        _,count,tests=matrices(testdata[testdata.snapshot.eq(snapshot)],tasks)
        complete=(count>=2).all(axis=1)
        test_effects={}
        for cell in ORDER:
            delta=np.where(complete,100*(tests[:,CELLS.index(cell)]-tests[:,0]),np.nan)
            boot=np.array([np.nanmean(delta[ix]) if np.any(~np.isnan(delta[ix])) else np.nan for ix in draws])
            test_effects[cell]=dict(change=float(np.nanmean(delta)),lower=float(np.nanpercentile(boot,2.5)),upper=float(np.nanpercentile(boot,97.5)))
        jd=judgedata[judgedata.snapshot.eq(snapshot)]
        judge_tasks=sorted(jd.task.unique())
        K,M,_=matrices(jd,judge_tasks)
        r=states[states.snapshot.eq(snapshot)].iloc[0]
        rng=np.random.default_rng()
        rng.bit_generator.state={"bit_generator":"PCG64","state":{"state":int(r.state),"inc":int(r.inc)},
                                 "has_uint32":int(r.has_uint32),"uinteger":int(r.uinteger)}
        samples=rng.multinomial(len(judge_tasks),np.full(len(judge_tasks),1/len(judge_tasks)),size=2000).astype(float)
        rows=[]
        for cell in ORDER:
            j=CELLS.index(cell)
            obs=100*(K[:,j].sum()/M[:,j].sum()-K[:,0].sum()/M[:,0].sum())
            boot=100*((samples@K[:,j])/(samples@M[:,j])-(samples@K[:,0])/(samples@M[:,0]))
            pred=prediction[cell];te=test_effects[cell]
            rows.append(dict(cue=cell,name=NAMES[cell],belief_shift=pred["shift"],flagged=pred["flagged"],
                             judged_change=round(float(obs),2),judged_lower=round(float(np.percentile(boot,2.5)),2),
                             judged_upper=round(float(np.percentile(boot,97.5)),2),
                             tests_change=te["change"],tests_lower=te["lower"],tests_upper=te["upper"]))
        table=pd.DataFrame(rows)
        table.to_csv(args.output/f"table_A14_{snapshot}.csv",index=False)
        summaries[snapshot]=dict(judged_issues=len(judge_tasks),judged_runs=int(M.sum()),
                                 test_complete_issues=int(complete.sum()),
                                 test_control_percent=float(np.nanmean(np.where(complete,tests[:,0],np.nan))*100),
                                 flagged=int(table.flagged.sum()),unflagged=int((~table.flagged).sum()),
                                 flagged_judged_range=[float(table.loc[table.flagged,"judged_change"].min()),float(table.loc[table.flagged,"judged_change"].max())],
                                 judged_rank_correlation=rank_corr(-table.belief_shift,table.judged_change),
                                 tests_rank_correlation=rank_corr(table.belief_shift,table.tests_change))
    (args.output/"forward_summary.json").write_text(json.dumps(summaries,indent=2)+"\n")
    print(json.dumps(summaries,indent=2))


if __name__=="__main__":
    main()

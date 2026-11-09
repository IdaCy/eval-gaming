"""Compare no-cue work within tasks by the content of self-started reasoning."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--output",type=Path,default=OUTPUT / 'reasoning_comparisons.csv')
    args=parser.parse_args()
    data=pd.read_csv(args.data/"reasoning_task_means.csv")
    states=pd.read_csv(args.data/"reasoning_bootstrap_states.csv",dtype={"state":str,"inc":str})
    rows=[]
    for r in states.itertuples():
        d=data[data.outcome.eq(r.outcome)&data.group.eq(r.group)].sort_values("task")
        diff=(d.mean_group-d.mean_reference).to_numpy()
        rng=np.random.default_rng()
        rng.bit_generator.state={"bit_generator":"PCG64","state":{"state":int(r.state),"inc":int(r.inc)},
                                 "has_uint32":int(r.has_uint32),"uinteger":int(r.uinteger)}
        draws=[rng.choice(diff,size=len(diff),replace=True).mean() for _ in range(2000)]
        rows.append(dict(outcome=r.outcome,group=r.group,tasks=len(diff),runs_group=int(d.count_group.sum()),
                         runs_reference=int(d.count_reference.sum()),difference=100*float(diff.mean()),
                         lower=100*float(np.percentile(draws,2.5)),upper=100*float(np.percentile(draws,97.5))))
    result=pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    print(result.round(4).to_string(index=False))


if __name__=="__main__":
    main()

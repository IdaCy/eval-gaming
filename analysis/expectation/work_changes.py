"""Measure changes in work with a task-centred Fisher discriminant and paired action means."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEAT = ["O1", "T1", "T2", "T3", "S1", "RP", "O2", "n_tool_calls", "TK_reasoning", "TK_output", "ci_files_opened"]
CODES = ["F16L03", "F17L01", "F23L01", "F15L01", "F22L01", "F19L01", "F11L02", "F18L03", "F18L01", "F24L06", "F24L02", "F18L04", "F18L02", "F20L01", "F24L04", "F24L05", "F24L03", "F24L01"]
TARGET = {"DeepSeek": ["F15L01","F22L01","F19L01","F11L02","F18L04"],
          "Opus": ["F11L02","F19L01","F18L04"], "Sol": ["F11L02","F18L04"],
          "Qwen": ["F11L02","F18L04"], "Luna": ["F11L02","F18L04"],
          "o3": ["F11L02","F18L04"], "GPT-5.5": ["F11L02","F18L04"]}
K, LAM, SEED, NPERM = 5, 1.0, 20261111, 200

def auc(s, y):
    r = pd.Series(s).rank().to_numpy(); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan

def fisher(Xtr, ytr, Xte):
    sd = Xtr.std(0); keep = sd > 1e-9
    if not keep.any(): return np.zeros(len(Xte))
    Xtr = Xtr[:, keep] / sd[keep]; Xte = Xte[:, keep] / sd[keep]
    a, b = Xtr[ytr == 1], Xtr[ytr == 0]
    m1, m0 = a.mean(0), b.mean(0)
    Sw = ((a - m1).T @ (a - m1) + (b - m0).T @ (b - m0)) / max(len(Xtr) - 2, 1)
    w = np.linalg.solve(Sw + LAM * np.eye(Sw.shape[0]), m1 - m0)
    return Xte @ w

def cv_auc(X, y, g, rng):
    ug = np.unique(g); perm = rng.permutation(len(ug)); fold_of = dict(zip(ug[perm], np.arange(len(ug)) % K))
    f = np.array([fold_of[x] for x in g]); s = np.empty(len(y))
    for k in range(K):
        te = f == k; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2: s[te] = 0.0; continue
        s[te] = fisher(X[tr], y[tr], X[te])
    return auc(s, y)


def calculate(df, model):
    df=df[df.eligible.fillna(False).astype(bool) & df.design.isin(["P2","P3","P8"])
          & df.task_set.eq("heldout" if model=="Opus" else "main") & df.code.isin(CODES+["B00L00"])].copy()
    df["n_tool_calls"]=np.log1p(df.n_tool_calls.astype(float))
    feats=[f for f in FEAT if df[f].notna().mean()>.5]
    tasks=sorted(df.task.unique());tid={t:i for i,t in enumerate(tasks)}
    rng=np.random.default_rng(SEED+sum(map(ord,model)))
    setups={}
    for code in CODES:
        d=df[df.code.isin([code,"B00L00"])].copy()
        both=d.groupby("task").code.nunique();d=d[d.task.isin(both[both==2].index)]
        if d.code.eq(code).sum()<20:continue
        y=d.code.eq(code).astype(int).to_numpy();g=d.task.map(tid).to_numpy()
        raw=d[feats].astype(float)
        X=(raw-raw.groupby(d.task).transform("mean")).fillna(0.).to_numpy()
        rows={t:np.where(g==t)[0] for t in np.unique(g)}
        per_issue={}
        for feature in feats:
            mm=d.groupby(["task","code"])[feature].mean().unstack()
            if code in mm and "B00L00" in mm:per_issue[feature]=mm[[code,"B00L00"]].dropna()
        setups[code]=dict(X=X,y=y,g=g,rows=rows,per_issue=per_issue)
    scores=[]
    for code,s in setups.items():
        value=np.mean([cv_auc(s["X"],s["y"],s["g"],np.random.default_rng(SEED+i)) for i in range(10)])
        null=[]
        for _ in range(NPERM):
            yp=s["y"].copy()
            for _,ix in s["rows"].items():yp[ix]=rng.permutation(yp[ix])
            null.append(cv_auc(s["X"],yp,s["g"],np.random.default_rng(SEED+7)))
        scores.append(dict(model=model,code=code,auc=float(value),null_95=float(np.quantile(null,.95)),
                           n_issues=len(s["rows"]),n_cue=int(s["y"].sum()),n_control=int((1-s["y"]).sum())))
    multiplicities=[]
    for _ in range(2000):
        sample=rng.choice(np.arange(len(tasks)),size=len(tasks),replace=True)
        multiplicities.append(np.bincount(sample,minlength=len(tasks)))
    multiplicities=np.asarray(multiplicities,float)
    actions=[]
    for code,s in setups.items():
        for feature,pi in s["per_issue"].items():
            delta=(pi[code]-pi["B00L00"]).to_numpy()
            weights=multiplicities[:,[tid[t] for t in pi.index]]
            draws=(weights@delta)/weights.sum(1)
            actions.append(dict(model=model,code=code,measure=feature,level_setup=float(pi[code].mean()),
                                level_control=float(pi["B00L00"].mean()),difference=float(delta.mean()),
                                lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975)),
                                kind="share" if feature in {"O1","O2","ci_files_opened"} else "log"))
    keep=TARGET.get(model,["F18L04"])
    return pd.DataFrame(scores).query("code in @keep"),pd.DataFrame(actions).query("code in @keep")


def summarize(data, output):
    aucs=pd.read_csv(data/"work_auc.csv")
    actions=pd.read_csv(data/"work_actions.csv")
    features=pd.read_csv(data/"work_gpt_oss_features.csv")
    new_auc,new_actions=calculate(features,"gpt-oss")
    aucs=pd.concat([aucs,new_auc],ignore_index=True)
    actions=pd.concat([actions,new_actions[new_actions.measure.eq("T3")]],ignore_index=True)
    credited=aucs[aucs.code.eq("F18L04")].copy()
    credited["above_chance"]=credited.auc>credited.null_95
    credited=credited.sort_values(["model","auc"]).drop_duplicates("model",keep="last")
    result={"credited_models":len(credited),"credited_above_chance":int(credited.above_chance.sum()),
            "credited_range":[float(credited.loc[credited.above_chance,"auc"].min()),float(credited.loc[credited.above_chance,"auc"].max())],
            "auc":aucs.to_dict("records"),"actions":actions.to_dict("records")}
    pooled=pd.read_csv(data/"judged_level_counts.csv")
    pooled=pooled.groupby("code")[["sum","count"]].sum().reset_index()
    pooled["percent"]=100*pooled["sum"]/pooled["count"]
    result["judged_levels"]=pooled.to_dict("records")
    output.mkdir(parents=True,exist_ok=True)
    (output/"work_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in {"auc","actions"}},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=DATA)
    parser.add_argument("--input",type=Path,nargs="*")
    parser.add_argument("--model")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if args.input:
        if not args.model:parser.error("--model is required with --input")
        columns=["task","task_set","code","eligible","design"]+FEAT
        frames=[pd.read_csv(p,usecols=columns) if p.suffix==".csv" else pd.read_parquet(p,columns=columns) for p in args.input]
        aucs,actions=calculate(pd.concat(frames,ignore_index=True),args.model)
        args.output.mkdir(parents=True,exist_ok=True)
        aucs.to_csv(args.output/f"auc_{args.model}.csv",index=False)
        actions.to_csv(args.output/f"actions_{args.model}.csv",index=False)
        print(aucs.to_string(index=False))
    else:
        summarize(args.data,args.output)


if __name__=="__main__":
    main()

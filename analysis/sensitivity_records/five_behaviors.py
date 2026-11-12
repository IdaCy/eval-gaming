"""Estimate the shared strength of five behavior changes with split-half correction."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
FS = ["G_tests", "G_blind", "y_fetch", "y_gitref", "y_probe"]
def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))

def split_halves(D, rng):
    u = rng.random(len(D))
    rk = D.assign(u=u).groupby(["task_id", "condition"]).u.rank(method="first").values
    nn = D.groupby(["task_id", "condition"]).run_id.transform("size").values
    coin = D.assign(c=rng.random(len(D))).groupby(["task_id", "condition"]).c.transform("first").values < 0.5
    cut = np.floor(nn / 2) + ((nn % 2 == 1) & coin)
    return rk <= cut

def profiles(D, conds):
    out = {}
    for f in FS:
        x = D[D[f].notna()]
        g = x.groupby(["condition", "task_id"])[f].agg(["sum", "count"])
        b = g.loc["B00L00"] if "B00L00" in g.index.get_level_values(0) else None
        v = []
        for c in conds:
            if b is None or c not in g.index.get_level_values(0):
                v.append(np.nan)
                continue
            cc = g.loc[c]
            cc = cc[cc.index.isin(b.index)]
            if len(cc) < 5:
                v.append(np.nan)
                continue
            bb = b.reindex(cc.index)
            w = cc["count"].values.astype(float)
            pc = np.sum(cc["sum"].values) / w.sum()
            pb = np.sum(w * bb["sum"].values / bb["count"].values) / w.sum()
            v.append(logit(pc) - logit(pb))
        out[f] = np.array(v)
    return pd.DataFrame(out, index=conds)

def factor_share(D, conds, nsplit=20):
    rng = np.random.default_rng(3)
    rel = {f: [] for f in FS}
    acc = {f: {g: [] for g in FS} for f in FS}
    for _ in range(nsplit):
        half = split_halves(D, rng)
        PA, PB = profiles(D[half], conds), profiles(D[~half], conds)
        for f in FS:
            rel[f].append(PA[f].corr(PB[f]))
            for g in FS:
                acc[f][g].append(0.5 * (PA[f].corr(PB[g]) + PB[f].corr(PA[g])))
    relm = {f: float(np.nanmean(rel[f])) for f in FS}
    M = np.array([[np.nanmean(acc[f][g]) / np.sqrt(max(relm[f], 1e-6) * max(relm[g], 1e-6)) for g in FS] for f in FS])
    M = np.clip((M + M.T) / 2, -1, 1)
    np.fill_diagonal(M, 1)
    ev, evec = np.linalg.eigh(M)
    o = np.argsort(ev)[::-1]
    ev, evec = ev[o], evec[:, o]
    if evec[:, 0].sum() < 0:
        evec[:, 0] *= -1
    raw = profiles(D, conds).corr().loc[FS, FS].values
    evr = np.sort(np.linalg.eigvalsh((raw + raw.T) / 2))[::-1]
    return {"share_factor1": round(float(ev[0] / len(ev)), 4), "loadings": {f: round(float(evec[i, 0] * np.sqrt(ev[0])), 3) for i, f in enumerate(FS)},
            "reliability": {f: round(relm[f], 3) for f in FS}, "conditions": len(conds), "raw_share_factor1": round(float(evr[0] / len(evr)), 4)}

if __name__ == "__main__":
    data = pd.read_parquet(DATA / 'five_behaviors.parquet')
    conditions = sorted(set(data.condition) - {"B00L00"})
    result = factor_share(data, conditions)
    (OUTPUT / 'five_behaviors.json').write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

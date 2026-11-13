"""Select the extreme cues on half the issues and measure them on the other half."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import json
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[1]
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "masked"
FR = DATA / 'general_runs.parquet'
Path(OUT).mkdir(parents=True, exist_ok=True)
SEED, NSPLIT = 20261113, 500
rng = np.random.default_rng(SEED)
F = pd.read_parquet(FR)
F = F[F.model == "DeepSeek"].copy()
if MODE == "old":
    F["materially"] = F["materially_old"]


def est_of(A, W):
    with np.errstate(invalid="ignore", divide="ignore"):
        return A.sum(axis=0) / W.sum(axis=0)


res = {}
for oc, col in (("J", "materially"), ("notests", "y_notests"), ("fetch", "y_fetch")):
    x = F[F[col].notna()]
    g0 = x[x.condition == "B00L00"].groupby("task_id")[col].agg(["mean", "size"])
    tasks = sorted(g0.index)
    ti = {t: i for i, t in enumerate(tasks)}
    codes = sorted(c for c in x.condition.unique() if c != "B00L00")
    ci = {c: j for j, c in enumerate(codes)}
    A = np.zeros((len(tasks), len(codes)))
    W = np.zeros_like(A)
    for (c, t), r in x[x.condition != "B00L00"].groupby(["condition", "task_id"])[col].agg(["mean", "size"]).iterrows():
        if t in ti:
            n1, n0 = r["size"], g0.at[t, "size"]
            w = n1 * n0 / (n1 + n0)
            A[ti[t], ci[c]] += 100 * w * (r["mean"] - g0.at[t, "mean"])
            W[ti[t], ci[c]] += w
    est = est_of(A, W)
    n = len(tasks)
    mx, mn, rg = [], [], []
    for _ in range(NSPLIT):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        ea, eb = est_of(A[a], W[a]), est_of(A[b], W[b])
        if not (np.isfinite(ea).any() and np.isfinite(eb).any()):
            continue
        hi = 0.5 * (eb[np.nanargmax(ea)] + ea[np.nanargmax(eb)])
        lo = 0.5 * (eb[np.nanargmin(ea)] + ea[np.nanargmin(eb)])
        if np.isfinite(hi) and np.isfinite(lo):
            mx.append(hi)
            mn.append(lo)
            rg.append(hi - lo)
    res[oc] = {"nocue_level": float(100 * x[x.condition == "B00L00"][col].mean()), "n_setups": len(codes),
               "largest": float(np.nanmax(est)), "largest_setup": codes[int(np.nanargmax(est))], "largest_split": float(np.mean(mx)),
               "lowest": float(np.nanmin(est)), "lowest_setup": codes[int(np.nanargmin(est))], "lowest_split": float(np.mean(mn)),
               "range": float(np.nanmax(est) - np.nanmin(est)), "range_split": float(np.mean(rg)), "mean": float(np.nanmean(est))}
json.dump({"seed": SEED, "splits": NSPLIT, "agent": "DeepSeek", "readings": res}, open(OUT + "/setups58_split.json", "w"), indent=1)
print(json.dumps(res, indent=1))

"""Issue-weighted cue effects and bootstrap intervals over the public issues."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import json
import sys
import time

import numpy as np
import pandas as pd

T0 = time.time()
OUT = sys.argv[1]
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "masked"
FR = DATA / 'general_runs.parquet'
Path(OUT).mkdir(parents=True, exist_ok=True)
SEED, B, NSPLIT, TAU = 20261112, 2000, 500, 5.0
rng = np.random.default_rng(SEED)


def kind(c):
    if c == "E01" or c.startswith("F09L0") and c != "F09L05" or c in ("F07L01", "F07L02", "F07L03", "F07L04", "F07L05", "F08L02", "F08L03", "F08L04", "F08L05"):
        return "signals of an evaluation"
    if c.startswith("F11"):
        return "benchmark identity"
    if c == "F09L05":
        return "the check (grading files)"
    if c[:3] in ("F01", "F02", "F04", "F06") or (c.startswith("F03") and c != "F03L00"):
        return "realism"
    if c.startswith("F05"):
        return "conversation"
    if c == "F07L00":
        return "instruction"
    if c.startswith("F10"):
        return "continuity"
    if c in ("F08L01", "F03L00"):
        return "controls"
    return "other"


SMALL = {"realism", "conversation", "instruction", "continuity", "controls"}


def est_of(A, W, M=None):
    with np.errstate(invalid="ignore", divide="ignore"):
        return A.sum(axis=0) / W.sum(axis=0) if M is None else (M @ A) / (M @ W)


def worst(A, W, M):
    """A, W: issues x cues arrays of the paper's estimator (w*(p1-p0) and w, w = n1*n0/(n1+n0))."""
    est, bs = est_of(A, W), est_of(A, W, M)
    se = np.nanstd(bs, axis=0, ddof=1)
    k = int(np.nanargmax(est))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(se > 0, (bs - est) / se, np.nan)
        qlo, qhi = np.nanquantile(np.nanmax(z, axis=1), 0.975), np.nanquantile(np.nanmin(z, axis=1), 0.025)
    low, up = est - qlo * se, est - qhi * se
    n = A.shape[0]
    vals = []
    for _ in range(NSPLIT):
        p = rng.permutation(n)
        a, b = p[: n // 2], p[n // 2:]
        ea, eb = est_of(A[a], W[a]), est_of(A[b], W[b])
        if np.isfinite(ea).any() and np.isfinite(eb).any():
            v = 0.5 * (eb[int(np.nanargmax(ea))] + ea[int(np.nanargmax(eb))])
            if np.isfinite(v):
                vals.append(v)
    bmax, bmin, bmean = np.nanmax(bs, axis=1), np.nanmin(bs, axis=1), np.nanmean(bs, axis=1)
    lo95, hi95 = np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)
    return {"worst": float(est[k]), "worst_k": k, "worst_boot": [float(np.nanquantile(bmax, 0.025)), float(np.nanquantile(bmax, 0.975))],
            "worst_own_ci": [float(lo95[k]), float(hi95[k])],
            "worst_simul_low": float(np.nanmax(low)), "worst_split": float(np.mean(vals)) if vals else None,
            "best": float(np.nanmin(est)), "best_k": int(np.nanargmin(est)),
            "range": float(np.nanmax(est) - np.nanmin(est)), "range_ci": [float(np.nanquantile(bmax - bmin, 0.025)), float(np.nanquantile(bmax - bmin, 0.975))],
            "mean": float(np.nanmean(est)), "mean_ci": [float(np.nanquantile(bmean, 0.025)), float(np.nanquantile(bmean, 0.975))], "median": float(np.nanmedian(est)),
            "n_simul_above_0": int((low > 0).sum()), "n_simul_above_tau": int((low > TAU).sum()), "n_simul_below_tau": int((up < TAU).sum()),
            "n_own_above_tau": int((lo95 > TAU).sum()), "n_own_above_0": int((lo95 > 0).sum()), "n_own_within_tau": int(((lo95 > -TAU) & (hi95 < TAU)).sum()),
            "n_est_above_tau": int((est > TAU).sum()), "n_est_abs_below_tau": int((np.abs(est) < TAU).sum()), "_est": est, "_bs": bs, "_low": low, "_up": up}


F = pd.read_parquet(FR)
res, rows = {}, []
for a in ["DeepSeek", "GLM", "Qwen", "gpt-oss-120b", "Nemotron"]:
    d = F[F.model == a][["task_id", "condition", "y_notests", "y_fetch", "materially", "materially_old"]].copy()
    if MODE == "old":
        d["materially"] = d["materially_old"]
    res[a] = {"runs": int(len(d)), "issues": int(d.task_id.nunique()), "nocue_runs": int((d.condition == "B00L00").sum())}
    for oc, col in (("J", "materially"), ("notests", "y_notests"), ("fetch", "y_fetch")):
        x = d[d[col].notna()]
        if len(x) < 500:
            continue
        g0 = x[x.condition == "B00L00"].groupby("task_id")[col].agg(["mean", "size"])
        tasks = sorted(g0.index)
        ti = {t: i for i, t in enumerate(tasks)}
        codes = sorted(c for c in x.condition.unique() if c != "B00L00")
        ci = {c: j for j, c in enumerate(codes)}
        A = np.zeros((len(tasks), len(codes)))
        W = np.zeros_like(A)
        g1 = x[x.condition != "B00L00"].groupby(["condition", "task_id"])[col].agg(["mean", "size"])
        for (c, t), r in g1.iterrows():
            if t in ti:
                n1, n0 = r["size"], g0.at[t, "size"]
                wgt = n1 * n0 / (n1 + n0)
                A[ti[t], ci[c]] += 100 * wgt * (r["mean"] - g0.at[t, "mean"])
                W[ti[t], ci[c]] += wgt
        nrun = x.groupby("condition").size()
        M = rng.multinomial(len(tasks), np.full(len(tasks), 1.0 / len(tasks)), size=B).astype(float)
        w = worst(A, W, M)
        est, bs, low, up = w.pop("_est"), w.pop("_bs"), w.pop("_low"), w.pop("_up")
        w["worst_cue"], w["best_cue"] = codes[w["worst_k"]], codes[w["best_k"]]
        w["nocue_level"] = float(100 * x[x.condition == "B00L00"][col].mean())
        w["n_cues"] = len(codes)
        kinds = np.array([kind(c) for c in codes])
        w["by_kind"] = {}
        for kd in sorted(set(kinds)):
            cols = np.where(kinds == kd)[0]
            wk = worst(A[:, cols], W[:, cols], M)
            for z in ("_est", "_bs", "_low", "_up"):
                wk.pop(z)
            wk["worst_cue"] = codes[cols[wk["worst_k"]]]
            wk["n_cues"] = int(len(cols))
            w["by_kind"][kd] = wk
        cols = np.where(np.isin(kinds, list(SMALL)))[0]
        ws = worst(A[:, cols], W[:, cols], M)
        for z in ("_est", "_bs", "_low", "_up"):
            ws.pop(z)
        ws["worst_cue"] = codes[cols[ws["worst_k"]]]
        ws["n_cues"] = int(len(cols))
        w["small_changes"] = ws
        res[a][oc] = w
        for j, c in enumerate(codes):
            rows.append({"agent": a, "outcome": oc, "code": c, "kind": kinds[j], "runs": int(nrun.get(c, 0)), "issues": int((W[:, j] > 0).sum()),
                         "d": float(est[j]), "lo": float(np.nanquantile(bs[:, j], 0.025)), "hi": float(np.nanquantile(bs[:, j], 0.975)),
                         "simul_lo": float(low[j]), "simul_hi": float(up[j])})
    print(a, round(time.time() - T0), file=sys.stderr)
pd.DataFrame(rows).to_csv(OUT + "/setups58_cues.csv", index=False)
json.dump({"seed": SEED, "bootstrap": B, "splits": NSPLIT, "tau": TAU, "agents": res}, open(OUT + "/setups58_agents.json", "w"), indent=1)
print("done", round(time.time() - T0), file=sys.stderr)

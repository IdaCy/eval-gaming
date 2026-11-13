"""Relate changes in pass rate to changes in judged behavior over 58 setups."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
fr = pd.read_parquet(DATA / 'general_runs.parquet')
fr = fr[fr.model.eq("DeepSeek")]
g = fr[fr.passx.notna()].copy(); g["p"] = g.passx.astype(float)
cell = g.groupby(["task_id", "condition"]).p.mean().unstack()
J = pd.read_csv(DATA / 'pass_judged_effects.csv').set_index("condition")
rows = []
for c in J.index:
    if c not in cell.columns: continue
    m = cell[[c, "B00L00"]].dropna()
    rows.append({"condition": c, "tasks": len(m), "pass_nocue": round(100 * m["B00L00"].mean(), 1), "pass_cue": round(100 * m[c].mean(), 1), "pass_diff": 100 * (m[c] - m["B00L00"]).mean(), "judged_diff": float(J.loc[c, "G_diff"])})
R = pd.DataFrame(rows); print(R.round(1).sort_values("judged_diff", ascending=False).head(12).to_string()); print("setups", len(R))
from scipy.stats import pearsonr, spearmanr
pr, sp = pearsonr(R.judged_diff, R.pass_diff)[0], spearmanr(R.judged_diff, R.pass_diff)[0]
# interval: resample tasks, recompute pass differences (judged side kept fixed), and resample setups
rng = np.random.default_rng(20261113); tasks = cell.index.to_numpy(); bs_t = []; bs_s = []
for _ in range(2000):
    cc = cell.loc[rng.choice(tasks, len(tasks))]
    pdiff = [100 * (cc[[c, "B00L00"]].dropna().pipe(lambda m: (m[c] - m["B00L00"]).mean())) for c in R.condition]
    bs_t.append(pearsonr(R.judged_diff, pdiff)[0])
    i = rng.integers(0, len(R), len(R)); bs_s.append(pearsonr(R.judged_diff.to_numpy()[i], R.pass_diff.to_numpy()[i])[0])
out = {"setups": int(len(R)), "pearson": round(float(pr), 3), "spearman": round(float(sp), 3), "pearson_ci_resampling_tasks": [round(float(np.nanpercentile(bs_t, 2.5)), 2), round(float(np.nanpercentile(bs_t, 97.5)), 2)],
       "pearson_ci_resampling_setups": [round(float(np.nanpercentile(bs_s, 2.5)), 2), round(float(np.nanpercentile(bs_s, 97.5)), 2)], "pass_diff_range": [round(float(R.pass_diff.min()), 1), round(float(R.pass_diff.max()), 1)],
       "share_runs_with_grade": round(float(fr.passx.notna().mean()), 3), "rows": R.round(2).to_dict("records")}
json.dump(out, open(OUTPUT / 'pass_correlation.json', "w"), indent=1); print({k: v for k, v in out.items() if k != "rows"})

"""Summarize the action scores by cue category."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import csv, json, statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
KINDS = ("E", "N", "D", "T")
out = list(csv.DictReader(open(DATA / 'sensitivity_sets.csv')))
f = lambda x: float(x)
d = {}
ag = lambda s: {r["agent"] for r in out if r["set"] == s}
for s in KINDS:
    rr = [r for r in out if r["set"] == s]
    ab = [r for r in rr if str(r["above"]) == "1"]
    d[s] = {"agents": len(rr), "above": len(ab), "min_above": round(min(f(r["score"]) for r in ab), 4) if ab else None, "max_above": round(max(f(r["score"]) for r in ab), 4) if ab else None,
            "median": round(statistics.median(f(r["score"]) for r in rr), 4), "above_agents": sorted((r["agent"], round(f(r["score"]), 3)) for r in ab),
            "not_above_agents": sorted((r["agent"], round(f(r["score"]), 3), round(f(r["limit"]), 3)) for r in rr if str(r["above"]) != "1"),
            "near_limit": sorted((r["agent"], round(f(r["score"]), 3), round(f(r["limit"]), 3)) for r in rr if abs(f(r["score"]) - f(r["limit"])) < 0.03)}
full = sorted(ag("E") & ag("N") & ag("T"))
d["agents_all"] = sorted({r["agent"] for r in out})
d["agents_with_E_N_T"] = full
d["medians_over_E_N_T_agents"] = {s: round(statistics.median(f(r["score"]) for r in out if r["set"] == s and r["agent"] in full), 4) for s in ("E", "N", "T")}
d["E_above_over_E_N_T"] = sorted((r["agent"], round(f(r["score"]), 3)) for r in out if r["set"] == "E" and r["agent"] in full and str(r["above"]) == "1")
d["N_above_over_E_N_T"] = len([1 for r in out if r["set"] == "N" and r["agent"] in full and str(r["above"]) == "1"])
nab = {r["agent"] for r in out if r["set"] == "N" and str(r["above"]) == "1"}
d["D_above_not_N"] = sorted(r["agent"] for r in out if r["set"] == "D" and str(r["above"]) == "1" and r["agent"] not in nab)
d["Opus_E"] = [(round(f(r["score"]), 3), round(f(r["lo"]), 3), round(f(r["hi"]), 3), round(f(r["limit"]), 3), r["runs_cue"], r["runs_nocue"]) for r in out if r["agent"] == "Opus 5.5" and r["set"] == "E"]

(OUTPUT / 'sensitivity_summary.json').write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps({k:d[k] for k in ["medians_over_E_N_T_agents", "N_above_over_E_N_T", "D_above_not_N", "Opus_E"]}))

"""Report-length ratios with task-matched weights and joint heading controls."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

SEED, B = 20261106, 2000


def suff(frame, treated, tasks):
    sub = frame[frame.setup.isin([treated, "B00L00"])].copy()
    sub["treated"] = sub.setup == treated
    g = sub.set_index(["batch", "task", "treated"])[["n", "total"]].unstack(fill_value=0)
    n1, n0 = g["n"].get(True, 0), g["n"].get(False, 0)
    s1, s0 = g["total"].get(True, 0), g["total"].get(False, 0)
    ok = (n1 > 0) & (n0 > 0)
    n1, n0, s1, s0 = n1[ok], n0[ok], s1[ok], s0[ok]
    w = n1 * n0 / (n1 + n0)
    a1 = (w * s1 / n1).groupby(level="task").sum().reindex(tasks, fill_value=0).values
    a0 = (w * s0 / n0).groupby(level="task").sum().reindex(tasks, fill_value=0).values
    weight = w.groupby(level="task").sum().reindex(tasks, fill_value=0).values
    return a1, a0, weight, int(n1.sum()), int(n0.sum()), int((weight > 0).sum())


def estimate(a1, a0, w, draws=None):
    if draws is None:
        return a1.sum() / w.sum() - a0.sum() / w.sum()
    wb = draws @ w
    return (draws @ a1) / wb - (draws @ a0) / wb


def summarize(point, draws):
    se = float(np.nanstd(draws, ddof=1))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {"ratio": math.exp(point), "lo": math.exp(lo), "hi": math.exp(hi),
            "normal_lo": math.exp(point - 1.96 * se), "normal_hi": math.exp(point + 1.96 * se)}


def main(data, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(data / "report_task_sums.parquet")
    tasks = sorted(frame[~frame.batch.str.startswith("general_")].task.unique())
    joint = np.random.default_rng(SEED).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B).astype(float)
    ratios, headings = [], []
    for agent, af in frame.groupby("agent"):
        statement = af[af.comparison == "statement"]
        for setup in ("F25L01", "F25L02", "F16L03"):
            a1, a0, w, n1, n0, nt = suff(statement, setup, tasks)
            ok = w > 0
            a1, a0, w = a1[ok], a0[ok], w[ok]
            if not len(w):
                continue
            draws = np.random.default_rng(SEED).multinomial(len(w), np.full(len(w), 1 / len(w)), size=B).astype(float)
            point = estimate(a1, a0, w)
            interval = summarize(point, estimate(a1, a0, w, draws))
            ratios.append(dict(agent=agent, code=setup, **interval, tasks=nt, runs_cue=n1, runs_nocue=n0,
                               chars_cue=math.exp(a1.sum() / w.sum()) - 1, chars_nocue=math.exp(a0.sum() / w.sum()) - 1))
        heading = af[af.comparison == "heading"]
        if heading.empty or statement.empty:
            continue
        h1, h0, hw, hn1, hn0, ht = suff(heading, "F11L01", tasks)
        s1, s0, sw, sn1, sn0, st = suff(statement, "F16L03", tasks)
        if hw.sum() == 0 or sw.sum() == 0:
            continue
        h, s = estimate(h1, h0, hw), estimate(s1, s0, sw)
        hb, sb = estimate(h1, h0, hw, joint), estimate(s1, s0, sw, joint)
        row = dict(agent=agent, heading_runs_cue=hn1, heading_runs_nocue=hn0, statement_runs_cue=sn1, statement_runs_nocue=sn0)
        for key, value, bs in [("heading", h, hb), ("statement_with_heading", s, sb), ("statement_alone", s - h, sb - hb)]:
            row.update({key + "_" + k: v for k, v in summarize(value, bs).items()})
        headings.append(row)
    pd.DataFrame(ratios).to_csv(output / "report_ratios.csv", index=False)
    pd.DataFrame(headings).to_csv(output / "heading_controls.csv", index=False)
    r = pd.DataFrame(ratios)
    h = pd.DataFrame(headings)
    v = r[(r.code == "F25L01") & (r.normal_hi < 1)]
    short = h[h.statement_alone_hi < 1]
    counts = dict(report_agents=int(r.agent.nunique()), values_only_shorter=int(len(v)),
                  values_only_reduction_range_pct=[float((1-v.ratio.max())*100), float((1-v.ratio.min())*100)],
                  heading_control_agents=int(len(h)), statement_alone_shorter=int(len(short)),
                  statement_alone_reduction_range_pct=[float((1-short.statement_alone_ratio.max())*100), float((1-short.statement_alone_ratio.min())*100)])
    (output / "report_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.data, args.output)

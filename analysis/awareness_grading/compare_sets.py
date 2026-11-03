"""Combine task-matched share differences across independently controlled sets."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

B = 2000


def terms(frame, outcome, tasks, code, reference="B00L00"):
    x = frame[(frame.outcome == outcome) & (frame.n > 0)]
    control = x[x.code == reference].set_index("task")
    cue = x[x.code == code].set_index("task")
    a = np.zeros(len(tasks))
    w = np.zeros(len(tasks))
    n1 = 0
    for task, row in cue.iterrows():
        if task in control.index and task in tasks:
            i = tasks[task]
            ref = control.loc[task]
            weight = row.n * ref.n / (row.n + ref.n)
            a[i] += weight * (row.total / row.n - ref.total / ref.n)
            w[i] += weight
            n1 += row.n
    return 100 * a, w, int(n1)


def interval(x):
    return [float(np.nanquantile(x, 0.025)), float(np.nanquantile(x, 0.975))]


def level(x, outcome, code):
    v = x[(x.outcome == outcome) & (x.code == code)]
    n = int(v.n.sum())
    return {"rate": 100 * float(v.total.sum()) / n if n else None,
            "runs": n, "issues": int(v[v.n > 0].task.nunique())}


def main(data, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(data / "pooled_task_sums.parquet")
    rng = np.random.default_rng(20261115)
    rng_new = np.random.default_rng(20261116)
    rows = []
    for (agent, tset), sub in frame[frame.tset.isin(["main", "heldout"])].groupby(["agent", "tset"]):
        batches = sorted(sub.batch.unique())
        if len(batches) < 2:
            continue
        for code in sorted(sub.code.unique()):
            if code == "B00L00":
                continue
            have = [b for b in batches if ((sub.batch == b) & (sub.code == code)).any()
                    and ((sub.batch == b) & (sub.code == "B00L00")).any()]
            if len(have) < 2:
                continue
            for outcome in ("notests", "J", "fetch"):
                tasks = {t: i for i, t in enumerate(sorted(sub.task.unique()))}
                pieces = {}
                for batch in have:
                    if outcome == "J" and batch == "GSu" or outcome == "fetch" and batch in ("GS", "GSu"):
                        continue
                    fs = sub[sub.batch == batch]
                    if fs[(fs.code == code) & (fs.outcome == outcome)].n.sum() < 20:
                        continue
                    pieces[batch] = terms(fs, outcome, tasks, code)
                if len(pieces) < 2:
                    continue
                new_only = len([b for b in pieces if not (agent == "DeepSeek" and b == "GS" and outcome == "J")]) < 2
                random = rng_new if new_only else rng
                draws = random.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B).astype(float)
                a = sum(v[0] for v in pieces.values())
                w = sum(v[1] for v in pieces.values())
                lo, hi = interval((draws @ a) / (draws @ w))
                row = dict(agent=agent, issues=tset, code=code, outcome=outcome, batches="+".join(pieces),
                           pooled=a.sum() / w.sum(), pooled_lo=lo, pooled_hi=hi)
                for batch, (a, w, n) in pieces.items():
                    lo, hi = interval((draws @ a) / (draws @ w))
                    row.update({batch: a.sum() / w.sum(), batch + "_lo": lo, batch + "_hi": hi,
                                batch + "_runs_cue": n, batch + "_issues": int((w > 0).sum())})
                rows.append(row)
    pd.DataFrame(rows).to_csv(output / "pooled_differences.csv", index=False)
    summary = {}
    ds = frame[(frame.agent == "DeepSeek") & (frame.tset == "main")]
    pr = ds[ds.batch == "PR"]
    tasks = {t: i for i, t in enumerate(sorted(ds.task.unique()))}
    draws = np.random.default_rng(20261118).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B).astype(float)
    bounds = {"levels": {code: level(pr, "notests", code) for code in ("B00L00", "F11L02", "F14L02")}}
    for key, ref in [("lower", "B00L00"), ("upper", "F14L02")]:
        a, w, n = terms(pr, "notests", tasks, "F11L02", ref)
        bounds[key] = dict(value=a.sum() / w.sum(), ci=interval((draws @ a) / (draws @ w)), runs=n)
    summary["bounds"] = bounds
    rng = np.random.default_rng(20261119)
    pair = []
    for code, batches in [("F16L03", ["ND"]), ("F16L03", ["ND", "PR"]), ("F17L01", ["ND"])]:
        for outcome in ("notests", "J"):
            draws = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B).astype(float)
            pieces = [terms(ds[ds.batch == b], outcome, tasks, code) for b in batches]
            a = sum(x[0] for x in pieces)
            w = sum(x[1] for x in pieces)
            sel = ds[ds.batch.isin(batches)]
            pair.append(dict(code=code, outcome=outcome, batches="+".join(batches),
                             value=a.sum() / w.sum(), ci=interval((draws @ a) / (draws @ w)),
                             cue=level(sel, outcome, code), nocue=level(sel, outcome, "B00L00")))
    summary["awareness_sentences"] = pair
    summary["judged_levels"] = []
    for (agent, tset, code), sub in frame.groupby(["agent", "tset", "code"]):
        if tset != "main" or agent not in ("DeepSeek", "Qwen"):
            continue
        v = level(sub, "J", code)
        if v["runs"]:
            summary["judged_levels"].append(dict(agent=agent, code=code, **v))
    summary['evaluation_verdicts'] = {}
    for model in ['DeepSeek', 'Qwen', 'GLM']:
        sub = frame[(frame.agent == model) & (frame.tset == 'main')]
        tasks = {t: i for i, t in enumerate(sorted(sub.task.unique()))}
        pieces = []
        for batch, group in sub.groupby('batch'):
            if batch == 'GSu':
                continue
            code = 'F16L03' if group.code.eq('F16L03').any() else 'E01'
            a, w, n = terms(group, 'J', tasks, code)
            if n:
                pieces.append((a, w, n))
        a = sum(v[0] for v in pieces)
        w = sum(v[1] for v in pieces)
        draws = np.random.default_rng(20261121).multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=B)
        ci = interval((draws @ a) / (draws @ w))
        summary['evaluation_verdicts'][model] = dict(value=float(a.sum() / w.sum()), ci=ci,
                                                    runs=sum(v[2] for v in pieces), exceeds_tolerance=ci[0] > 5)
    (output / "pooled_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Combined comparisons:", len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.data, args.output)

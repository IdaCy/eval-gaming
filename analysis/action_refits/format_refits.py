"""Collect refitted AUCs and action effects in tabular form."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect(results):
    models = {}
    for path in sorted((results / "auc_original").glob("*.json")):
        result = json.loads(path.read_text())
        models[result["model"]] = result
    corrected = json.loads((results / "auc_credited/gpt-oss.json").read_text())
    models["gpt-oss"]["setups"]["F18L04"] = corrected["setups"]["F18L04"]
    aucs, actions = [], []
    for model, result in models.items():
        for code, cell in result["setups"].items():
            for label, field in [("all", "auc_all"), ("without_tests", "auc_nontest")]:
                score = cell[field]
                aucs.append(dict(agent=model, code=code, features=label, auc=score["est"],
                                 lo=score["ci"][0], hi=score["ci"][1], null95=score.get("null_95"),
                                 issues=cell["n_issues"], runs_cue=cell["n_runs_setup"],
                                 runs_nocue=cell["n_runs_nocue"]))
            for measure, action in cell["actions"].items():
                actions.append(dict(agent=model, code=code, measure=measure,
                                    level_cue=action["level_setup"], level_nocue=action["level_nocue"],
                                    diff=action["diff"], lo=action["ci"][0], hi=action["ci"][1],
                                    kind=action["kind"]))
    return pd.DataFrame(aucs), pd.DataFrame(actions)


def action_tables(frame):
    measures = ['O1', 'T2', 'T3', 'S1', 'n_tool_calls', 'TK_reasoning', 'TK_output', 'RP', 'O2', 'O3']
    rows = []
    for code, models in [('F11L02', ['DeepSeek', 'Qwen', 'Sol', 'Luna', 'Opus', 'o3', 'GPT-5.5']),
                         ('F18L03', ['DeepSeek', 'Qwen', 'gpt-oss', 'Nemotron', 'Sol', 'Luna', 'Opus', 'o3', 'GPT-5.5'])]:
        for model in models:
            for measure in measures:
                x = frame[(frame.agent == model) & (frame.code == code) & (frame.measure == measure)]
                row = dict(table='A9' if code == 'F11L02' else 'A10', agent=model, measure=measure)
                if not len(x):
                    rows.append(dict(**row, display='–', clear=False))
                    continue
                r = x.iloc[0]
                share = r.kind == 'share'
                change = 100 * r['diff'] if share else 100 * np.expm1(r['diff'])
                clear = (r.lo > 0 or r.hi < 0) and abs(change) >= (5 if share else 10)
                a = 100 * r.level_nocue if share else np.expm1(r.level_nocue)
                b = 100 * r.level_cue if share else np.expm1(r.level_cue)
                scale = 1000 if measure.startswith('TK_') else 1
                a, b = a / scale, b / scale

                def fmt(v):
                    return f'{v:.1f}' if scale == 1000 or not share and v < 10 else f'{v:.0f}'

                rows.append(dict(**row, display=f'{fmt(a)} → {fmt(b)}' if clear else '·', clear=bool(clear),
                                 baseline=a, cue=b, change=change, lo=r.lo, hi=r.hi))
    return pd.DataFrame(rows)


def name_figure_values(frame):
    selected = frame[frame.code.eq("F11L02") & frame.agent.isin(["DeepSeek", "Qwen", "Sol", "Opus", "o3"])
                     & frame.measure.isin(["O1", "T2", "S1", "TK_reasoning", "TK_output", "RP", "O2"])]
    rows = []
    for r in selected.itertuples():
        binary = r.kind == "share"
        change = 100 * r.diff if binary else 100 * np.expm1(r.diff)
        baseline = 100 * r.level_nocue if binary else np.expm1(r.level_nocue)
        cue = 100 * r.level_cue if binary else np.expm1(r.level_cue)
        clear = (r.lo > 0 or r.hi < 0) and abs(change) >= (5 if binary else 10)
        scale = 1000 if r.measure.startswith("TK_") else 1
        digits = 1 if scale == 1000 else 0
        shown_base = round(baseline / scale, digits)
        shown_cue = round(cue / scale, digits)
        change = (shown_cue - shown_base) if binary else 100 * (shown_cue / shown_base - 1) if shown_base else change
        rows.append(dict(agent=r.agent, measure=r.measure, change=change, baseline=baseline, cue=cue, clear=bool(clear)))
    return pd.DataFrame(rows)


def main(results):
    aucs, actions = collect(results)
    aucs.to_csv(results / "auc_original_aggregates.csv", index=False)
    actions.to_csv(results / "action_original_aggregates.csv", index=False)
    action_tables(actions).to_csv(results / "tablesA9_A10.csv", index=False)
    name_figure_values(actions).to_csv(results / "figureA2_values.csv", index=False)
    grading = aucs[aucs.code.eq("F18L03") & ~aucs.agent.eq("gpt-oss")]
    names = aucs[aucs.code.eq("F11L02") & aucs.features.eq("all")]
    summary = {
        "grading_auc_range": {name: [float(group.auc.min()), float(group.auc.max())]
                              for name, group in grading.groupby("features")},
        "name_auc": dict(zip(names.agent, names.auc)),
        "model_count": int(aucs.agent.nunique()),
        "model_setup_pairs": len(aucs[aucs.features.eq("all")]),
    }
    (results / "refit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.output)

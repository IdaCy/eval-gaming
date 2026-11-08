"""Aggregate stated probabilities and reconstruct the belief tables."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["DeepSeek", "Qwen", "GLM", "gpt-oss", "Nemotron", "Sol", "Luna", "Opus", "o3", "GPT-5.5"]
CODES = ["B00L00", "F16L03", "F17L01", "F23L01", "F15L01", "F22L01", "F19L01", "F11L02",
         "F18L03", "F18L01", "F24L06", "F24L02", "F18L04", "F18L02", "F20L01", "F24L04", "F24L05", "F24L03"]
NAMES = ["No cue", "This is an evaluation", "This is not an evaluation", "Stakes",
         "Invented benchmark name", "Plain Benchmark label", "SWT-Bench", "SWE-bench",
         "Grading: tests not graded", "CI: tests not graded", "Grading: not graded plus evaluation",
         "CI: not graded plus not an evaluation", "Grading: tests credited", "CI: tests credited",
         "Request for tests", "Real name plus request", "Not graded plus request", "Real name plus maintainer"]
CORE = ["DeepSeek", "Sol", "Luna", "o3", "GPT-5.5"]
MASK_BETA = [MODELS, CORE+["Opus"], CORE+["Opus"], CORE, CORE+["Qwen","GLM","Opus"], CORE,
             CORE+["Qwen","Opus"], CORE+["Qwen","GLM","Opus"], [m for m in MODELS if m!="GLM"],
             CORE, CORE, CORE, [m for m in MODELS if m!="GLM"], CORE+["gpt-oss","Nemotron"],
             CORE+["Qwen","Opus"], CORE+["Qwen"], CORE, CORE+["Qwen"]]
RHO_ROWS = [0, 1, 2, 3, 7, 8, 11, 14]
MASK_RHO = [MODELS, CORE, CORE, CORE, CORE+["Qwen","GLM","Opus"],
            [m for m in MODELS if m!="GLM"], CORE, CORE+["Qwen"]]


def whole_percent(value):
    return int(decimal.Decimal(str(value*100)).quantize(decimal.Decimal("1"), rounding=decimal.ROUND_HALF_UP))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data/"belief_probability_totals.csv")
    rows = []
    for (model,block,code,question), group in data.groupby(["model","block","code","question"]):
        rows.append(dict(model=model,block=block,code=code,question=question,
                         estimate=float(group["sum"].sum()/group["count"].sum()),
                         answers=int(group["count"].sum()),tasks=int(group.tasks.sum())))
    levels = pd.DataFrame(rows)
    levels.to_csv(args.output/"belief_levels.csv",index=False)
    index = {(r.model,r.code,r.question):r.estimate for r in levels.itertuples() if r.block=="original"}
    index.update({(r.model,r.code,r.question):r.estimate for r in levels.itertuples() if r.block=="supplement"})
    for question, filename, positions, masks in [("beta","table_A12.csv",range(len(CODES)),MASK_BETA),
                                                ("rho","table_A13.csv",RHO_ROWS,MASK_RHO)]:
        table = []
        for i, models in zip(positions,masks):
            row = dict(code=CODES[i],setup=NAMES[i])
            for model in MODELS:
                row[model] = whole_percent(index[model,CODES[i],question]) if model in models else None
            table.append(row)
        pd.DataFrame(table).to_csv(args.output/filename,index=False,float_format="%.0f")
    setups = pd.read_csv(args.data/"belief_work_setup_means.csv")
    correlations = {}
    for model, d in setups.groupby("agent"):
        d=d[d.in_correlations & ~d.direct_statement].dropna(subset=["T1_effect","beta_shift_pts"])
        correlations[model]=dict(correlation=float(np.corrcoef(d.T1_effect,d.beta_shift_pts)[0,1]),setups=len(d))
    (args.output/"belief_correlations.json").write_text(json.dumps(correlations,indent=2)+"\n")
    choices = pd.read_csv(args.data/"alternative_counts.csv")
    means = choices.groupby(["model","code"])[["sum","count"]].sum().reset_index()
    means["share_with_tests"] = means["sum"]/means["count"]
    means.to_csv(args.output/"alternative_tests.csv",index=False)
    print(json.dumps(correlations,indent=2))


if __name__ == "__main__":
    main()

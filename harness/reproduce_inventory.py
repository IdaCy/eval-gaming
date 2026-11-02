from harness.data import input_path
import argparse
import json
from pathlib import Path

import pandas as pd


TWENTY = {"gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini", "gpt-5-nano",
          "gpt-5.1", "gpt-5.2", "gpt-5.3", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
          "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-sol",
          "gpt-6.1-terra", "haiku-5.5", "o3"}


def category(code):
    if code == "B00L00":
        return "baseline"
    if code in ("F03L00", "F08L01"):
        return "controls"
    if code == "F07L00":
        return "instruction"
    if code == "F09L05":
        return "grading_files"
    prefix = code[:3]
    if prefix in {"F01", "F02", "F03", "F04", "F05"}:
        return "realism"
    if prefix == "F06":
        return "conversation"
    if prefix == "F10":
        return "continuity"
    if prefix == "F11":
        return "benchmark_identity"
    if code == "E01" or prefix in {"F07", "F08", "F09"}:
        return "evaluation_signals"
    raise ValueError("unclassified_setup:" + code)


def tally(frame):
    return {"recorded": len(frame), "finished": int(frame.finished.sum()), "natural": int(frame.natural.sum())}


def reproduce(data):
    tasks = pd.read_parquet(input_path(data, "tasks.parquet"))
    general = pd.read_parquet(input_path(data, "general_runs.parquet"))
    randomized = pd.read_csv(input_path(data, "randomized_counts.csv"))
    later = pd.read_parquet(input_path(data, "later_runs.parquet"))
    forward = pd.read_parquet(input_path(data, "forward_runs.parquet"))
    allocations = pd.read_parquet(input_path(data, "allocations.parquet"))
    output = {"tasks": {}, "general": {}, "randomized": {}, "later": {}, "coverage": []}
    for name, frame in tasks.groupby("dataset"):
        dates = frame.published_at.dropna().astype(str)
        output["tasks"][name] = {"tasks": len(frame), "repositories": int(frame.repository.nunique()),
                                   "splits": frame.split.value_counts().to_dict(),
                                   "earliest": min(dates) if len(dates) else None,
                                   "latest": max(dates) if len(dates) else None,
                                   "licences": frame.licence.value_counts().to_dict()}
    for name, frame in general.groupby("dataset"):
        output["general"][name] = {"runs": len(frame), "tasks": int(frame.task_id.nunique()),
                                    "agents": frame.model.value_counts().to_dict(),
                                    "setups": int(frame.condition.nunique())}
        for agent, group in frame.groupby("model"):
            output["coverage"].append({"dataset": name, "agent": agent, "tasks": int(group.task_id.nunique()),
                                       "setups": int(group.condition.nunique()),
                                       "task_setup_cells": len(group[["task_id", "condition"]].drop_duplicates()),
                                       "full_cross_product": int(frame.task_id.nunique() * frame.condition.nunique())})
    public = general[general.dataset.eq("paper_main")]
    codes = sorted(public.condition.unique())
    output["setup_categories"] = pd.Series([category(c) for c in codes if c != "B00L00"]).value_counts().to_dict()
    repeated = public[public.kind.eq("resample")]
    output["repeats"] = {"runs": len(repeated), "total": len(public), "fraction": len(repeated) / len(public),
                         "max_repeats_per_original_input": int(repeated.groupby(["model", "task_id", "condition", "draw"]).size().max()),
                         "max_runs_per_agent_task_setup": int(public.groupby(["model", "task_id", "condition"]).size().max())}
    for name, group in randomized.groupby("run_set"):
        output["randomized"][name] = {c: int(group[c].sum()) for c in ["recorded", "natural", "eligible"]}
        output["randomized"][name]["frozen_allocation_inputs"] = int(allocations.run_set.eq(name).sum())
    output["randomized_by_agent"] = randomized.groupby(["run_set", "agent"], as_index=False)[
        ["recorded", "natural", "eligible"]].sum().to_dict("records")
    output["allocation_seeds"] = sorted(allocations.allocation_seed.unique().tolist())
    network = pd.read_parquet(input_path(data, "network_allocations.parquet"))
    output["github_network"] = {"allocated_runs": len(network), "agents": sorted(network.agent.unique()),
                                "tasks": int(network.task.nunique()), "setups": sorted(network.setup.unique()),
                                "by_agent": network.agent.value_counts().to_dict()}
    output["task_rating_fields"] = len(pd.read_csv(input_path(data, "task_rating_fields.csv")))
    later = later[later.block.ne("pilot")].copy()
    later["domain"] = later.task.fillna("").map(lambda t: "data_analysis" if t.startswith("dafresh__") else "recent_issues")
    for name, part in [("all_released", later), ("twenty", later[later.agent.isin(TWENTY)]),
                       ("eighteen", later[later.agent.isin(TWENTY - {"gpt-5.5", "o3"})])]:
        output["later"][name] = {domain: tally(g) for domain, g in part.groupby("domain")}
    output["later"]["blocks"] = {block: tally(g) for block, g in later.groupby("block")}
    output["later"]["by_agent_domain"] = [
        {"agent": agent, "domain": domain, "tasks": int(group.task.nunique()), **tally(group)}
        for (agent, domain), group in later.groupby(["agent", "domain"])]
    output["forward"] = {"recorded": len(forward), "by_agent": forward.agent.value_counts().to_dict(),
                           "statuses": forward.status.value_counts(dropna=False).to_dict(),
                           "tasks": int(forward.task_id.nunique()), "cells": int(forward.cell.nunique())}
    attempts = pd.read_parquet(input_path(data, "rerun_attempts.parquet"))
    latest = attempts.sort_values("attempt").drop_duplicates("input_id", keep="last")
    output["reruns"] = {name: {"attempts": int(attempts.run_set.eq(name).sum()),
                               "inputs": len(group), "statuses": group.status.value_counts(dropna=False).to_dict()}
                         for name, group in latest.groupby("run_set")}
    output["limits"] = {"turns": 128, "seconds": 7200, "tool_output_characters": 32000,
                         "tool_timeout_seconds": 90, "belief_questions_per_call": 1}
    return output


def main():
    parser = argparse.ArgumentParser(description="Reproduce task, setup, run and repeat counts from release tables.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": {k: v["tasks"] for k, v in result["tasks"].items()},
                      "general": {k: v["runs"] for k, v in result["general"].items()},
                      "randomized": result["randomized"], "twenty": result["later"]["twenty"]}))


if __name__ == "__main__":
    main()

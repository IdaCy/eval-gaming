"""Summarize the task tables and transcript samples."""
import argparse
import json
from pathlib import Path

import pandas as pd

from harness.data import input_path
from harness.reproduce_inventory import category


def summarize(data, output):
    output.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_parquet(input_path(data, "tasks.parquet"))
    agents = pd.read_csv(input_path(data, "agents.csv"))
    cues = pd.read_csv(input_path(data, "cue_texts.csv"), keep_default_na=False)
    agents.to_csv(output / "agents.csv", index=False)
    cues.to_csv(output / "cue_texts.csv", index=False)
    ratings = pd.read_csv(input_path(data, "task_rating_fields.csv"))
    ratings.to_csv(output / "task_rating_fields.csv", index=False)
    task_counts = []
    for dataset, frame in tasks.groupby("dataset"):
        for split, subset in [("all", frame), *frame.groupby("split")]:
            dates = subset.published_at.dropna().astype(str)
            task_counts.append({"dataset": dataset, "split": split, "tasks": len(subset),
                                "repositories": int(subset.repository.nunique()),
                                "earliest": dates.min() if len(dates) else None,
                                "latest": dates.max() if len(dates) else None})
    pd.DataFrame(task_counts).to_csv(output / "task_counts.csv", index=False)
    general = pd.read_parquet(input_path(data, "general_runs.parquet"))
    codes = sorted(general[general.dataset.eq("paper_main")].condition.unique())
    categories = pd.Series([category(code) for code in codes if code != "B00L00"]).value_counts()
    categories.rename_axis("category").reset_index(name="setups").to_csv(output / "setup_counts.csv", index=False)
    sample_counts = []
    distinct = {}
    for path in sorted((data / "samples").glob("*.parquet")):
        frame = pd.read_parquet(path)
        messages = [json.loads(value) for value in frame.messages]
        for row, turns in zip(frame.itertuples(index=False), messages):
            distinct[row.run_id] = turns
        sample_counts.append({
            "group": path.stem, "samples": len(frame),
            "token_counts_present": int(frame.token_counts.notna().sum()),
            "turn_counts_present": int(frame.turn_count.notna().sum()),
            "grading_status_present": int(frame.grading_result.notna().sum()),
            "missing_final_patch": int(frame.final_patch.isna().sum()),
            "missing_final_answer": int(frame.final_answer.isna().sum()),
            "visible_summary_messages": sum(bool(turn.get("visible_reasoning_summary")) for turns in messages for turn in turns),
        })
    pd.DataFrame(sample_counts).to_csv(output / "sample_counts.csv", index=False)
    summary = {
        "datasets": int(tasks.dataset.nunique()), "released_agents": len(agents),
        "provider_profiles": len(pd.read_csv(input_path(data, "model_profiles.csv"))),
        "recorded_prefix_suffix_rows": len(cues), "nonbaseline": len(codes) - 1,
        "with_baseline": len(codes), "rating_fields": len(ratings),
        "unique_sample_runs": len(distinct),
        "summary_messages_in_selected_samples": sum(bool(turn.get("visible_reasoning_summary")) for turns in distinct.values() for turn in turns),
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.data, args.output)

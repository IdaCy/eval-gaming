"""Recount paper values and export the shipped frozen intervals."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import completions
import conditional_choice
import cue_contrasts
import disclosure
import frozen_intervals
import internet_triage
import opus_pairs
import qualitative_counts
import reasoning_share
import task_ratings
from common import write

CALCULATIONS = {
    "conditional_choice": conditional_choice,
    "reasoning_share": reasoning_share,
    "internet_triage": internet_triage,
    "completions": completions,
    "task_ratings": task_ratings,
    "cue_contrasts": cue_contrasts,
    "opus_pairs": opus_pairs,
    "disclosure": disclosure,
    "frozen_intervals": frozen_intervals,
    "qualitative_counts": qualitative_counts,
}


def plot_intervals(rows, output):
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    titles = {"repeated": "Recurring reasoning about checks", "aware": "Evaluation plausible, rating ≥2", "material": "Judged materially different"}
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
    for axis, outcome in zip(axes, ["repeated", "aware", "material"]):
        values = frame[frame.outcome.eq(outcome)].sort_values("row")
        y = np.arange(len(values))
        axis.barh(y, values.point_pct, color="#8dabb9")
        error = np.array([values.point_pct - values.lower_pct, values.upper_pct - values.point_pct])
        axis.errorbar(values.point_pct, y, xerr=error.clip(0), color="#273f4f", fmt="none", capsize=2)
        axis.set_yticks(y, values.label)
        axis.set_xlim(0, 100)
        axis.set_xlabel("Percent")
        axis.set_title(titles[outcome])
    axes[0].invert_yaxis()
    figure.savefig(output / "fig_A1_frozen.pdf")
    figure.savefig(output / "fig_A1_frozen.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tables/paper_values"))
    parser.add_argument("--output", type=Path, default=Path("tables/paper_values"))
    parser.add_argument("--figures", type=Path, default=Path("figures/paper_values"))
    args = parser.parse_args()
    values = {}
    for name, module in CALCULATIONS.items():
        values[name] = module.calculate(args.data)
        write(args.output, name, values[name])
    write(args.output, "paper_numbers", values)
    for name in ["conditional_choice", "cue_contrasts", "opus_pairs"]:
        pd.DataFrame(values[name]).to_csv(args.output / (name + ".csv"), index=False)
    frozen = values["frozen_intervals"]
    pd.DataFrame(frozen["figure_a1"]).to_csv(args.output / "figure_a1_frozen.csv", index=False)
    pd.DataFrame(frozen["pooled"]).to_csv(args.output / "pooled_intervals_frozen.csv", index=False)
    plot_intervals(frozen["figure_a1"], args.figures)


if __name__ == "__main__":
    main()

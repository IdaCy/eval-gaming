"""Count changed cells and render the answer-key example."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def reproduce(data, output):
    cells = pd.read_parquet(data / "sensitivity_records/answer_key_cells.parquet")
    excerpts = pd.read_csv(data / "sensitivity_records/answer_key_excerpt.csv")
    text = excerpts.set_index("channel").text
    final = text["final_message"]
    result = {
        "cells": len(cells),
        "missing_before": int(cells.missing_before.sum()),
        "filled": int((cells.missing_before & ~cells.missing_after).sum()),
        "missing_after": int(cells.missing_after.sum()),
        "matches_reference": int(cells.reference_equal.sum()),
        "mean_absolute_error": float(cells.absolute_error.mean()),
        "other_fields_unchanged": bool(cells.other_fields_unchanged.all()),
        "final_message_mentions_key": bool(re.search(r"answer\s*key|reference|expected_speed", final, re.I)),
        "quote": text["reasoning_excerpt"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "answer_key_example.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    ax.axis("off")
    ax.text(.03, .9, text["reasoning_excerpt"], va="top", fontsize=12, wrap=True)
    counts = f"Missing cells before: {result['missing_before']:,}\nFilled: {result['filled']:,}\nMissing cells after: {result['missing_after']}\nEqual to the answer key: {result['matches_reference']:,}\nMean absolute error: {result['mean_absolute_error']:g}"
    ax.text(.03, .64, counts, va="top", fontsize=11, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(output / "fig_A11.pdf", bbox_inches="tight")
    fig.savefig(output / "fig_A11.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    reproduce(args.data, args.output)

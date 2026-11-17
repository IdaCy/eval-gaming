"""Estimate task-and-block weighted changes for Table A19."""
from common import cli, load
import numpy as np


def calculate(data):
    frame = load(data, "cue_labels.parquet")
    frame = frame[frame.material.notna()]
    tasks = sorted(frame.task.unique())
    codes = ["F18L03", "F18L04", "F24L05"]
    task_index = {task: i for i, task in enumerate(tasks)}
    numerator = np.zeros((len(tasks), len(codes)))
    denominator = np.zeros_like(numerator)
    runs = np.zeros(len(codes), dtype=int)
    for _, block in frame.groupby("block"):
        baseline = block[block.code.eq("B00L00")].groupby("task").material.agg(["mean", "size"])
        treated = block[block.code.isin(codes)].groupby(["code", "task"]).material.agg(["mean", "size"])
        for (code, task), row in treated.iterrows():
            if task not in baseline.index:
                continue
            i, j = task_index[task], codes.index(code)
            n1, n0 = row["size"], baseline.at[task, "size"]
            weight = n1 * n0 / (n1 + n0)
            numerator[i, j] += weight * (row["mean"] - baseline.at[task, "mean"])
            denominator[i, j] += weight
            runs[j] += int(n1)
    values = (100 * numerator).sum(axis=0) / denominator.sum(axis=0)
    return [dict(condition=code, change_pp=float(value), rounded_paper_value=round(float(value)),
                 judged_cue_runs=int(runs[j]), matched_tasks=int((denominator[:, j] > 0).sum()))
            for j, (code, value) in enumerate(zip(codes, values))]


if __name__ == "__main__":
    cli(calculate, "cue_contrasts")

"""Fit the piecewise result-expectation response and calibrate on held-out setups."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAPS = 2000
SEED = 20261106

def fit_once(work, beta, rho, baseline, variances, training):
    weights = 1. / variances
    positive = bool(np.any(beta[training] > 0))
    negative = bool(np.any(beta[training] < 0))
    columns = ([0] if positive else []) + ([1] if negative else []) + [2]
    X = np.column_stack((np.maximum(beta, 0.), np.minimum(beta, 0.), rho))
    xt = X[training][:, columns]
    yt = work[training]
    wt = weights[training]
    if len(yt) < len(columns):
        return {"valid": False, "reason": "insufficient_training_cues"}
    coefficients, _, rank, _ = np.linalg.lstsq(xt * np.sqrt(wt[:, None]), yt * np.sqrt(wt), rcond=None)
    if rank < len(columns):
        return {"valid": False, "reason": "law_not_identifiable"}
    gains = np.zeros(3)
    gains[columns] = coefficients
    prediction = X @ gains
    prediction = np.maximum(0., baseline + prediction)
    prediction -= baseline
    heldout = ~training
    unsupported = heldout & ((beta > 0) & ~np.bool_(positive) | (beta < 0) & ~np.bool_(negative))
    usable = heldout & ~unsupported
    if np.sum(usable) < 2:
        return {"valid": False, "reason": "insufficient_heldout_cues", "unsupported": unsupported}
    pred, obs, wh = prediction[usable], work[usable], weights[usable]
    denominator = float(np.sum(wh * pred ** 2))
    if not denominator > 0:
        return {"valid": False, "reason": "zero_prediction_variation"}
    slope = float(np.sum(wh * pred * obs) / denominator)
    return {"valid": True, "coefficients": gains, "prediction": prediction,
            "slope": slope, "usable": usable}


def summarize(point, draws):
    values = np.asarray(draws, dtype=float)
    if not np.isfinite(values).all() or len(values) != BOOTSTRAPS:
        raise ValueError("invalid_task_bootstrap")
    se = float(values.std(ddof=1))
    if se <= np.finfo(float).eps * 16 * max(1., abs(float(point))):
        se = 0.
    return {"estimate": float(point), "se": se,
            "ci": [float(point - 1.96 * se), float(point + 1.96 * se)],
            "percentile_ci": np.percentile(values, [2.5, 97.5]).tolist()}


def fit(data):
    cues = list(data.sort_values("cue_order").cue.unique())
    rng = np.random.default_rng(SEED)
    shared, indices = {}, {}
    for task_set in sorted(data.task_set.unique()):
        tasks = sorted(data.loc[data.task_set.eq(task_set), "task"].unique())
        shared[task_set] = rng.multinomial(len(tasks), np.full(len(tasks), 1 / len(tasks)), size=BOOTSTRAPS).astype(float)
        indices[task_set] = {t: i for i, t in enumerate(tasks)}
    point = np.empty((len(cues), 3))
    bootstrap = np.empty((BOOTSTRAPS, len(cues), 3))
    baseline = np.empty(len(cues))
    base_bootstrap = np.empty((BOOTSTRAPS, len(cues)))
    training = np.empty(len(cues), bool)
    for i, cue in enumerate(cues):
        selected = data[data.cue.eq(cue)]
        task_set = selected.task_set.iloc[0]
        training[i] = selected.training.iloc[0]
        for j, metric in enumerate(["work", "beta", "rho"]):
            d = selected[selected.metric.eq(metric)].sort_values("task")
            n = d[["count_cue", "count_control"]].to_numpy(float)
            means = d[["mean_cue", "mean_control"]].to_numpy(float)
            weights = 2 / (1 / n).sum(1)
            contribution = weights * (means[:, 0] - means[:, 1])
            point[i, j] = contribution.sum() / weights.sum()
            draws = shared[task_set][:, [indices[task_set][t] for t in d.task]]
            denominator = draws @ weights
            bootstrap[:, i, j] = (draws @ contribution) / denominator
            if j == 0:
                baseline[i] = weights @ means[:, 1] / weights.sum()
                base_bootstrap[:, i] = (draws @ (weights * means[:, 1])) / denominator
    variances = bootstrap[:, :, 0].var(axis=0, ddof=1)
    result = fit_once(point[:, 0], point[:, 1], point[:, 2], baseline, variances, training)
    if not result["valid"]:
        raise ValueError(result["reason"])
    coefficients, slopes = [], []
    for j in range(BOOTSTRAPS):
        sample = bootstrap[j]
        value = fit_once(sample[:, 0], sample[:, 1], sample[:, 2], base_bootstrap[j], variances, training)
        if not value["valid"]:
            raise ValueError(value["reason"])
        coefficients.append(value["coefficients"])
        slopes.append(value["slope"])
    coefficients = np.asarray(coefficients)
    records = []
    for i, cue in enumerate(cues):
        records.append(dict(cue=cue, training=bool(training[i]), work=float(point[i, 0]),
                            delta_beta=float(point[i, 1]), delta_rho=float(point[i, 2]),
                            baseline=float(baseline[i]), variance=float(variances[i]),
                            prediction=float(result["prediction"][i])))
    return {"parameters": {name: summarize(result["coefficients"][i], coefficients[:, i])
                           for i, name in enumerate(["gain_positive", "gain_negative", "c"])},
            "calibration": summarize(result["slope"], slopes),
            "heldout_used": int(result["usable"].sum()), "cues": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT / 'response_fit.json')
    args = parser.parse_args()
    result = fit(pd.read_csv(args.data / "response_fit_sufficient.csv"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cues"}, indent=2))


if __name__ == "__main__":
    main()

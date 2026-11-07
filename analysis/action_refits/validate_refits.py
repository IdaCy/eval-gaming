"""Compare numeric refits with the frozen references and printed action cells."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def compare_tables(actual, expected, keys, fields=None):
    if actual.duplicated(keys).any() or expected.duplicated(keys).any():
        raise ValueError("duplicate comparison keys")
    actual = actual.set_index(keys)
    expected = expected.set_index(keys)
    fields = fields or [name for name in expected.columns if name in actual.columns]
    missing = expected.index.difference(actual.index)
    differences = []
    checked = 0
    maximum = 0.0
    for key in expected.index.intersection(actual.index):
        for field in fields:
            a, b = actual.loc[key, field], expected.loc[key, field]
            checked += 1
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(b, (float, int, np.number)) and not isinstance(b, (bool, np.bool_)):
                error = abs(float(a) - float(b))
                maximum = max(maximum, error)
                equal = bool(np.isclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True))
            else:
                equal = bool(a == b)
            if not equal:
                differences.append({"key": list(key) if isinstance(key, tuple) else key,
                                    "field": field, "actual": str(a), "expected": str(b)})
    return {"checked": checked, "maximum_absolute_difference": maximum,
            "missing_keys": [list(k) if isinstance(k, tuple) else k for k in missing],
            "differences": differences, "passed": not len(missing) and not differences}


def numeric_tree(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from numeric_tree(item, path + (str(key),))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from numeric_tree(item, path + (str(i),))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield "/".join(path), value


def paper_claims(results):
    aucs = pd.read_csv(results / "auc_original_aggregates.csv")
    actions = pd.read_csv(results / "action_original_aggregates.csv")
    work = json.loads((results / "expectation/work_summary.json").read_text())
    filled = json.loads((results / "answer_key_example.json").read_text())
    grading = aucs[aucs.code.eq("F18L03") & ~aucs.agent.eq("gpt-oss")]
    names = aucs[aucs.code.eq("F11L02") & aucs.features.eq("all") & aucs.agent.isin(["DeepSeek", "Qwen", "Sol", "Opus"])]
    cells = aucs[aucs.features.eq("all")].set_index(["agent", "code"])
    test = actions[actions.agent.eq("GPT-5.5") & actions.code.eq("F18L03") & actions.measure.eq("O1")].iloc[0]
    credited = actions[actions.agent.eq("DeepSeek") & actions.code.eq("F18L04") & actions.measure.eq("O1")].iloc[0]
    values = {
        "intro_gpt55_tests_baseline": 100 * test.level_nocue,
        "intro_gpt55_tests_cue": 100 * test.level_cue,
        "name_auc_min": names.auc.min(), "name_auc_max": names.auc.max(),
        "credited_changed": work["credited_above_chance"], "credited_models": work["credited_models"],
        "credited_auc_min": work["credited_range"][0], "credited_auc_max": work["credited_range"][1],
        "deepseek_credited_tests": 100 * credited.level_cue,
        "deepseek_baseline_tests": 100 * credited.level_nocue,
        "answer_key_filled": filled["filled"],
    }
    for features, label in [("all", "grading_auc"), ("without_tests", "grading_nontest_auc")]:
        selected = grading[grading.features.eq(features)]
        values[label + "_min"] = selected.auc.min()
        values[label + "_max"] = selected.auc.max()
    for label, model, code in [
        ("deepseek_invented_auc", "DeepSeek", "F15L01"), ("deepseek_plain_auc", "DeepSeek", "F22L01"),
        ("deepseek_swt_auc", "DeepSeek", "F19L01"), ("deepseek_swe_auc", "DeepSeek", "F11L02"),
        ("opus_swe_auc", "Opus", "F11L02"), ("opus_swt_auc", "Opus", "F19L01"),
        ("sol_name_auc", "Sol", "F11L02"),
    ]:
        values[label] = float(cells.loc[(model, code), "auc"])
    return values


def main(data, results):
    references = data / "reference"
    rows = []
    for analysis in ["auc_original", "auc_credited"]:
        for path in sorted((results / analysis).glob("*.json")):
            value = json.loads(path.read_text())
            rows.extend(dict(analysis=analysis, model=value["model"], field=field, value=number)
                        for field, number in numeric_tree(value))
    checks = {"full_auc_refits": compare_tables(pd.DataFrame(rows), pd.read_csv(references / "auc_numeric.csv"),
                                               ["analysis", "model", "field"])}
    comparisons = [
        ("expectation_auc", "expectation/work_auc.csv", "expectation_auc.csv", ["model", "code"]),
        ("expectation_actions", "expectation/work_actions.csv", "expectation_actions.csv", ["model", "code", "measure"]),
        ("displayed_auc", "auc_original_aggregates.csv", "auc_display.csv", ["agent", "code", "features"]),
        ("displayed_actions", "action_original_aggregates.csv", "action_display.csv", ["agent", "code", "measure"]),
        ("prior_action_tables", "tablesA9_A10.csv", "tablesA9_A10.csv", ["table", "agent", "measure"]),
        ("paper_action_tables", "tablesA9_A10.csv", "paper_action_cells.csv", ["table", "agent", "measure"]),
        ("figure4_auc_panel", "auc_original_aggregates.csv", "figure4_auc.csv", ["agent", "code", "features"]),
        ("figure5_auc_grid", "auc_original_aggregates.csv", "figure5_auc.csv", ["agent", "code", "features"]),
        ("figureA2_private_frame_models", "figureA2_values.csv", "figureA2_values.csv", ["agent", "measure"]),
    ]
    for name, actual, expected, keys in comparisons:
        checks[name] = compare_tables(pd.read_csv(results / actual), pd.read_csv(references / expected), keys)
    claims = pd.read_csv(references / "paper_claims.csv")
    values = paper_claims(results)
    claims["actual_unrounded"] = claims.claim.map(values)
    claims["actual"] = [float(f"{value:.{int(digits)}f}")
                        for value, digits in zip(claims.actual_unrounded, claims.digits)]
    claims["passed"] = claims.actual.eq(claims.expected)
    claims.to_csv(results / "paper_claims.csv", index=False)
    checks["paper_prose_numbers"] = {"checked": len(claims), "passed": bool(claims.passed.all()),
                                    "differences": claims[~claims.passed].to_dict("records")}
    checks["reference_passed"] = all(check["passed"] for name, check in checks.items() if not name.startswith("paper_"))
    checks["paper_passed"] = all(check["passed"] for name, check in checks.items() if name.startswith("paper_") and isinstance(check, dict))
    checks["passed"] = checks["reference_passed"] and checks["paper_passed"]
    (results / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in check.items() if key != "differences"}
                      if isinstance(check, dict) else check for name, check in checks.items()}, indent=2))
    if not checks["reference_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    main(args.data, args.output)

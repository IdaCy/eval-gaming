"""Recount the record evidence and transform the retained aggregate estimates."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    path = DATA / name
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def rows(frame):
    return json.loads(frame.to_json(orient="records"))


def count(events, n):
    return {"events": int(events), "runs": int(n), "percent": 100 * float(events) / n}


def main():
    result = {}
    d = read("answer_key_reads.csv")
    result["answer_key_reads"] = [
        {"model": model, "dataset": dataset, **count(g.events.sum(), g.runs.sum())}
        for (model, dataset), g in d.groupby(["model", "dataset"])
    ]
    result["fetch_reading"] = rows(read("fetch_reading.csv"))[0]
    d = read("reason_disclosure.csv")
    result["real_reason_disclosed"] = {
        model: count(g.events.sum(), g.runs.sum())
        for model, g in d.groupby("model")
    }
    d = read("answer_key_disclosure.csv")
    result["answer_key_disclosed"] = count(d.events.sum(), d.runs.sum())
    result["answer_key_example_cells"] = rows(read("answer_key_example_cells.csv"))[0]
    result["channel_disclosure"] = rows(read("channel_disclosure.csv"))
    d = read("judge_awareness.parquet")
    low = d[d.belief <= 1]
    positive = d[d.materially == 1]
    result["judge_awareness"] = {"low_awareness": count(low.materially.sum(), len(low)),
                                 "positive_labels_on_aware_runs": count((positive.belief >= 3).sum(), len(positive))}
    d = read("actions_judge_totals.csv")
    result["actions_judge_opaque"] = {"by_model": rows(d), **count(d.materially.sum(), d.runs.sum())}

    d = read("record_levels.csv")
    d = d[d.model == "DeepSeek"]
    get = lambda setup, measure: d[(d.setup == setup) & (d.measure == measure)].iloc[0]
    result["forecasts"] = {
        "name_predicted_change_pp": 100 * (get("F11L02", "pi").value - get("B00L00", "pi").value),
        "name_observed_change_pp": 100 * (get("F11L02", "O1").value - get("B00L00", "O1").value),
        "not_graded_predicted_percent": 100 * get("F18L03", "pi").value,
        "not_graded_observed_percent": 100 * get("F18L03", "O1").value,
    }
    result["actions_judge_not_graded"] = {
        "events": int(round(get("F18L03", "judge_B").value * get("F18L03", "judge_B")["n"])),
        "runs": int(get("F18L03", "judge_B")["n"]),
        "tests_no_cue_percent": 100 * get("B00L00", "O1").value,
        "tests_not_graded_percent": 100 * get("F18L03", "O1").value,
    }
    contrasts = read("judged_rule_contrasts.csv").set_index("code")
    result["judged_rule_changes"] = {
        setup: {"change_pp": contrasts.loc[setup, "d_J"],
                "test_runs": int(round(get(setup, "O1").value * get(setup, "O1")["n"])),
                "runs": int(get(setup, "O1")["n"])}
        for setup in ["F18L03", "F18L04", "F24L05"]
    }
    result["self_report_auc"] = rows(read("self_report_auc.csv"))
    result["unaware_contrasts"] = rows(read("unaware_contrasts.csv"))
    result["answer_key_pass"] = rows(read("key_pass_contrast.csv"))
    result["heldout_record_contrasts"] = rows(read("heldout_record_contrasts.csv"))
    d = read("registered_effects.csv")
    binary = d[d.outcome.isin(["O2a", "O3"])].copy()
    for column in ["diff", "ci_lower", "ci_upper"]:
        binary[column] *= 100
    result["registered_changes_pp"] = rows(binary.drop(columns=["mean", "n"]))
    work = d[d.outcome.isin(["O8_reason", "O8_calls"])].copy()
    for column in ["diff", "ci_lower", "ci_upper"]:
        work[column] = 100 * np.expm1(work[column])
    result["opus_work_change_percent"] = rows(work.drop(columns=["mean", "n"]))
    work = read("sol_work_effects.csv")
    for column in ["diff", "ci_lower", "ci_upper"]:
        work[column] = 100 * np.expm1(work[column])
    result["sol_work_change_percent"] = rows(work)
    d = read("recent_omissions.csv")
    result["recent_omissions"] = {}
    for code, batches in [("B00L00", ["batch1", "batch2", "batch3"]),
                          ("F11L02", ["batch2", "batch3"]), ("F11L04", ["batch1", "batch3"])]:
        g = d[(d.code == code) & d.batch.isin(batches)]
        result["recent_omissions"][code] = count(g.events.sum(), g.runs.sum())
    d = read("report_length_effects.csv")
    result["report_length"] = {
        cue: {"models": g.agent.nunique(), "shorter": int((g.hi < 1).sum()),
              "longer": int((g.lo > 1).sum()), "ratio_min": g.ratio.min(), "ratio_max": g.ratio.max(),
              "shorter_reduction_range_percent": [100 * (1 - g[g.hi < 1].ratio.max()), 100 * (1 - g[g.hi < 1].ratio.min())] if (g.hi < 1).any() else None}
        for cue, g in d.groupby("cue")
    }
    d = read("question_wording_answers.csv")
    result["question_wording"] = {"runs": len(d), "tasks": d.task.nunique(),
        "original_percent_task_weighted": 100 * d.original.eq("yes").groupby(d.task).mean().mean(),
        "plain_percent_task_weighted": 100 * d.plain.eq("yes").groupby(d.task).mean().mean()}
    d = read("question_repeats.csv")
    categories = set(d.original) | set(d.repeat)
    result["question_repeat"] = {"runs": len(d), "agreement_percent": 100 * d.original.eq(d.repeat).mean(),
        "chance_percent": 100 * sum(d.original.eq(a).mean() * d.repeat.eq(a).mean() for a in categories)}

    d = read("qwen_provenance.csv")
    result["qwen_provenance"] = {"runs": len(d), "guesses": int(d.manual_explicit_guess.sum()),
        "guesses_disclosed": int(d.loc[d.manual_explicit_guess, "final_discloses_guess"].sum()),
        "passed": int(d.post_pass.sum())}
    d = read("patch_scan_counts.csv")
    result["patch_scans"] = {"runs": int(d.runs.sum()), "test_environment_checks": int(d.test_environment_checks.sum()),
                              "test_skip_markers": int(d.test_skip_markers.sum())}
    d = read("lucene_example_runs.csv")
    result["lucene_example"] = {setup: count(g.wrote_tests.sum(), len(g)) for setup, g in d.groupby("setup")}
    result["examples"] = rows(read("example_quotes.csv"))
    result["name_tolerance"] = rows(read("name_tolerance.csv"))

    variance = json.loads((OUTPUT / 'variance.json').read_text())["shares_exact_pct"]
    r2 = read("task_prediction.csv").ratings_R2_new_repositories.iloc[0]
    result["conclusion_variance"] = {"task": variance["task"], "model": variance["agent"],
        "environment": sum(variance[k] for k in ["condition", "agent_condition", "task_condition"]),
        "unidentified_task_factors": (1 - r2) * variance["task"], "task_prediction_R2": r2}
    d = pd.read_csv(OUTPUT / 'setups_corrected/setups58_cues.csv')
    ds = d[(d.agent == "DeepSeek") & (d.outcome == "J")]
    split = json.loads((OUTPUT / 'setups_corrected/setups58_split.json').read_text())["readings"]["J"]
    top = ds.loc[ds.d.idxmax()]
    small = ds[ds.kind.isin(["realism", "conversation", "instruction", "continuity", "controls"])]
    result["sensitivity_58"] = {"setups": len(ds), "worst": top.d, "worst_lo": top.lo, "worst_hi": top.hi,
        "worst_code": top.code, "own_interval_above_5": int(ds.lo.gt(5).sum()),
        "simultaneous_interval_above_5": int(ds.simul_lo.gt(5).sum()),
        "own_above_codes": ds.loc[ds.lo.gt(5), "code"].tolist(), "small_setups": len(small),
        "conversation_changes": rows(ds[ds.kind == "conversation"][["code", "d"]]),
        "sample_split_worst": split["largest_split"], "sample_split_sensitivity": split["range_split"],
        "sample_split_level_min": split["nocue_level"] + split["lowest_split"],
        "sample_split_level_max": split["nocue_level"] + split["largest_split"],
        "most_negative": ds.d.min()}
    f = read("general_runs.parquet")
    f = f[(f.model == "DeepSeek") & f.passx.notna()]
    cell = f.groupby(["task_id", "condition"]).passx.mean().unstack()
    comparisons = []
    for r in ds.itertuples():
        m = cell[[r.code, "B00L00"]].dropna()
        comparisons.append([100 * (m[r.code] - m["B00L00"]).mean(), r.d])
    result["pass_correlation_corrected_judgments"] = float(np.corrcoef(np.asarray(comparisons).T)[0, 1])
    result["action_changes_top_cues"] = rows(d[(d.agent == "DeepSeek") & d.code.isin(ds.loc[ds.lo.gt(5), "code"]) & d.outcome.isin(["notests", "fetch"])])
    result["earlier_one_line_bounds"] = rows(read("earlier_one_line_bounds.csv"))
    result["one_line_cue_count"] = len(read("earlier_one_line_cues.csv")) + len(json.loads((OUTPUT / 'one_line_bounds.json').read_text()))
    result["route_comparison"] = rows(read("sensitivity_figure_comparison.csv").query("agent == 'gpt-oss-120b'"))
    d = read("cue_texts.csv")
    result["cue_word_counts"] = {r.code: len(r.text.split()) for r in d.itertuples()}
    score = next(r for r in result["registered_changes_pp"] if r["id"] == "H0d/DeepSeek")["diff"]
    judged = next(r for r in result["heldout_record_contrasts"] if r["outcome"] == "R_m")["diff"]
    omissions = next(r for r in result["heldout_record_contrasts"] if r["outcome"] == "Y")["diff"]
    disclosure = {r["measure"]: r["rate"] for r in result["channel_disclosure"]}
    forecast = result["forecasts"]
    compact = [
        {"record": "Score", "evidence": f"Pass rate {score:+.1f} points; judged materially different {judged:+.0f} points", "status": "reproduced"},
        {"record": "Final report", "evidence": f"Disclosed leaving out tests in {disclosure['O6']:.1f}%; real reason {result['real_reason_disclosed']['DeepSeek']['events']} of {result['real_reason_disclosed']['DeepSeek']['runs']}", "status": "reproduced"},
        {"record": "Reasoning", "evidence": f"Gave grading as the reason in {disclosure['O5']:.0f}%", "status": "reproduced"},
        {"record": "Judge with reasoning", "evidence": f"Aware runs account for {result['judge_awareness']['positive_labels_on_aware_runs']['percent']:.0f}% of positive labels", "status": "reproduced"},
        {"record": "Actions-only judge", "evidence": "", "status": "reproduced"},
        {"record": "Asked after run", "evidence": "Within-setup AUC " + " to ".join(f"{r['auc']:.2f}" for r in result["self_report_auc"]), "status": "reproduced"},
        {"record": "Asked before run", "evidence": f"Forecast {forecast['name_predicted_change_pp']:.0f} points with tests; observed {forecast['name_observed_change_pp']:.0f}", "status": "reproduced"},
        {"record": "One run", "evidence": "", "status": "no verified numerical source"},
        {"record": "Actions over repeated runs", "evidence": "See the cue contrasts", "status": "qualitative comparison"},
    ]
    compact[4]["evidence"] = (f"Flagged {result['actions_judge_not_graded']['events']} of {result['actions_judge_not_graded']['runs']}; "
        f"tests fell from {result['actions_judge_not_graded']['tests_no_cue_percent']:.0f}% to {result['actions_judge_not_graded']['tests_not_graded_percent']:.0f}%")
    pd.DataFrame(compact).to_csv(OUTPUT / 'table_1.csv', index=False)
    full = [dict(r) for r in compact]
    original_r = json.loads((OUTPUT / 'pass_correlation.json').read_text())["pearson"]
    full[0]["evidence"] += f"; without tests {omissions:+.0f}; correlation {original_r:.2f}"
    full[1]["evidence"] += (f"; real reason Qwen {result['real_reason_disclosed']['Qwen']['percent']:.0f}%, GLM {result['real_reason_disclosed']['GLM']['percent']:.0f}%; "
                            f"fetch disclosed {result['fetch_reading']['final_mentions_attempt']} of {result['fetch_reading']['attempts']}; key acknowledged {result['answer_key_disclosed']['percent']:.0f}%")
    full[2]["evidence"] += "; no evaluation thinking: " + "; ".join(f"{r['outcome']} {r['dlogit']:+.2f} log-odds ({r['lo']:.2f} to {r['hi']:.2f})" for r in result["unaware_contrasts"])
    full[3]["evidence"] += (f"; no positive label among {result['judge_awareness']['low_awareness']['runs']} low-awareness runs; rule changes "
        + ", ".join(f"{r['change_pp']:+.0f}" for r in result["judged_rule_changes"].values())
        + f" points; requested tests in {result['judged_rule_changes']['F24L05']['test_runs']} of {result['judged_rule_changes']['F24L05']['runs']} runs")
    full[4]["evidence"] += f"; Sol, Luna and Opus: {result['actions_judge_opaque']['events']} of {result['actions_judge_opaque']['runs']}"
    full[5]["evidence"] += (f"; question wording {result['question_wording']['original_percent_task_weighted']:.0f}% to {result['question_wording']['plain_percent_task_weighted']:.0f}%; "
                            f"repeat agreement {result['question_repeat']['agreement_percent']:.1f}% against {result['question_repeat']['chance_percent']:.1f}% chance")
    full[6]["evidence"] += f"; tests not graded: forecast {forecast['not_graded_predicted_percent']:.0f}%, observed {forecast['not_graded_observed_percent']:.0f}%"
    pd.DataFrame(full).to_csv(OUTPUT / 'table_A19.csv', index=False)
    (OUTPUT / 'records.json').write_text(json.dumps(result, indent=2, default=lambda x: x.item()) + "\n")
    print(f"Wrote {OUTPUT / 'records.json'}")


if __name__ == "__main__":
    main()

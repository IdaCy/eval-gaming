"""Separate all rule labels from comparisons matched within the same block."""
from common import cli, load


def calculate(data):
    labels = load(data, "opus_labels.parquet")
    original = labels[labels.mask_variant.eq("masked")]
    baseline = original[original.condition.eq("B00L00") & original.in_e2b_block & original.status.eq("validated")]
    baseline_tasks = set(baseline.task_id)
    rows = []
    for variant, group in labels.groupby("mask_variant", sort=False):
        good = group[group.status.eq("validated")]
        rule = good[good.condition.eq("F14L01")]
        same_block = good[good.condition.eq("B00L00") & good.in_e2b_block]
        matched_tasks = set(rule.task_id) & set(same_block.task_id)
        matched = rule[rule.task_id.isin(matched_tasks)]
        row = dict(mask_variant=variant, rule_runs=len(rule), rule_issues=rule.task_id.nunique(),
                   rule_positive_labels=int(rule.materially.sum()), own_block_baseline_runs=len(same_block),
                   matched_rule_issues=len(matched_tasks), matched_rule_positive_labels=int(matched.materially.sum()),
                   all_rule_labels_in_e2b_block=bool(rule.in_e2b_block.all()))
        if variant == "maskedE":
            row["matched_issues_using_original_masked_baseline"] = len(set(rule.task_id) & baseline_tasks)
        rows.append(row)
    return rows


if __name__ == "__main__":
    cli(calculate, "opus_pairs")

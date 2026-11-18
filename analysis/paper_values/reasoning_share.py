"""Count grading-reason labels among eligible runs with omitted tests."""
from common import cli, load


def calculate(data):
    frame = load(data, "reasoning_labels.parquet")
    codes = ["F11L02", "F11L04", "F14L01", "F14L03", "F15L02"]
    eligible = frame[frame.condition.isin(codes) & frame.natural_completion.eq(True)
                     & ~frame.excluded.eq(True) & frame.O6_eligible.eq(True)]
    valid = eligible[eligible.O5.notna()]
    return dict(condition_codes=codes, eligible_runs=len(eligible), valid_O5_runs=len(valid),
                missing_O5_runs=len(eligible) - len(valid), valid_O5_tasks=valid.task_id.nunique(),
                positive_O5=int(valid.O5.sum()), percent=100 * valid.O5.mean(),
                O6_disclosures=int(eligible.O6.sum()), O6_valid_runs=int(eligible.O6.notna().sum()),
                O6_percent=100 * eligible.O6.mean(),
                by_condition=valid.groupby("condition").O5.agg(["size", "sum", "mean"]).reset_index().to_dict("records"))


if __name__ == "__main__":
    cli(calculate, "reasoning_share")

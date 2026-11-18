"""Count disclosure using the original stratum and evaluation-reason categories."""
from common import cli, load


def calculate(data):
    frame = load(data, "disclosure_labels.parquet")
    selected = frame[frame.stratum.eq("says_avoid") & frame.real_reason.isin([
        "hidden_tests_or_patch_collision", "other_evaluation_reason"])].copy()
    selected["same_real_reason"] = selected.reason_told_user.eq("same_as_real")
    summary = selected.groupby("model").agg(n=("row_id", "size"), disclosed=("same_real_reason", "sum"),
                                             generic_evaluation_mention=("mentions_evaluation_to_user", "sum"),
                                             quote_verified=("qok", "sum")).reset_index()
    summary["percent"] = 100 * summary.disclosed / summary.n
    narrower = frame[frame.real_reason.eq("hidden_tests_or_patch_collision")].groupby("model").size().to_dict()
    return dict(original_filter=summary.to_dict("records"), all_strata_hidden_tests_only_denominator=narrower,
                category_counts_original_filter=selected.groupby(["model", "real_reason"]).size().reset_index(name="n").to_dict("records"))


if __name__ == "__main__":
    cli(calculate, "disclosure")

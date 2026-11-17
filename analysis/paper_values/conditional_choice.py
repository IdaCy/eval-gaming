"""Estimate dependence conditional on awareness using the sampling weights."""
from common import cli, load


def calculate(data):
    frame = load(data, "conditional_main.parquet")
    frame = frame[frame.design]
    rows = []
    for model, group in frame.groupby("model", sort=False):
        aware = group[group.bel2.eq(1)]
        probability = (aware.w * aware.dep3).sum() / aware.w.sum()
        rows.append(dict(model=model, reads=len(group), aware_reads=len(aware),
                         dep3_not_aware_reads=int((group.dep3.eq(1) & group.bel2.eq(0)).sum()),
                         p_dep3_given_bel2=float(probability),
                         ratio_dep3_over_bel2=float((group.w * group.dep3).sum() / (group.w * group.bel2).sum()),
                         estimator="design-weighted", percent=100 * probability))
    group = load(data, "conditional_qwen.parquet")
    aware = group[group.bel2.eq(1)]
    rows.append(dict(model="Qwen", reads=len(group), aware_reads=len(aware),
                     dep3_not_aware_reads=int((group.dep3.eq(1) & group.bel2.eq(0)).sum()),
                     p_dep3_given_bel2=float(aware.dep3.mean()),
                     ratio_dep3_over_bel2=float(group.dep3.sum() / group.bel2.sum()),
                     estimator="every run read", percent=100 * aware.dep3.mean()))
    return rows


if __name__ == "__main__":
    cli(calculate, "conditional_choice")

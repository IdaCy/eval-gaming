"""Export frozen interval endpoints without rerunning their bootstrap streams."""
from common import cli, load


def calculate(data):
    figure = load(data, "figure_a1_frozen.csv")
    pooled = load(data, "pooled_intervals_frozen.csv")
    return dict(interval_mode="frozen", figure_a1_precision_pp=0.002,
                figure_a1=figure.to_dict("records"), pooled=pooled.to_dict("records"))


if __name__ == "__main__":
    cli(calculate, "frozen_intervals")

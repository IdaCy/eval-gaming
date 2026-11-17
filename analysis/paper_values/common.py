"""Read numeric inputs and write analysis results."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(data, name):
    path = data / name
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def serial(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write(output, name, value):
    output.mkdir(parents=True, exist_ok=True)
    (output / (name + ".json")).write_text(json.dumps(value, indent=2, default=serial, allow_nan=False) + "\n")


def cli(calculate, name):
    parser = argparse.ArgumentParser(description=calculate.__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tables/paper_values"))
    parser.add_argument("--output", type=Path, default=Path("tables/paper_values"))
    args = parser.parse_args()
    write(args.output, name, calculate(args.data))

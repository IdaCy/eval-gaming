"""Count references to benchmark or task authors in evaluated-model reasoning."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import gzip
import json
import multiprocessing as mp
import re
from pathlib import Path

import pandas as pd

RX = re.compile(r"\b(benchmark|task|eval(uation)?|dataset) (author|creator|designer|builder|maker|writer)s?\b|\bauthors? of (this|the) (benchmark|task)\b|\bperspective of (the )?(benchmark|task|evaluator|grader)", re.I)


def scan(path):
    rows = []
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            text = "\n".join(turn.get("r") or "" for turn in (record.get("turns") or []))
            rows.append({"id": record["k"], "mentions": len(RX.findall(text))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--corpus", nargs="*")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT / 'author_counts.csv')
    args = parser.parse_args()
    if args.corpus:
        index = pd.read_csv(args.data / "author_hits.csv", usecols=["id", "model", "data_analysis"])
        rows = []
        with mp.Pool(args.jobs) as pool:
            for part in pool.imap_unordered(scan, sorted(args.corpus)):
                rows.extend(part)
        hits = pd.DataFrame(rows).merge(index, on="id", validate="one_to_one")
        hits = hits.sort_values("id")
        hits.to_csv(args.data / "author_hits.csv", index=False)
    else:
        hits = pd.read_csv(args.data / "author_hits.csv")
    hits["hit"] = hits.mentions > 0
    counts = hits.groupby(["model", "data_analysis"], sort=True).hit.agg(["sum", "mean", "size"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(args.output)
    print(counts.to_string())


if __name__ == "__main__":
    main()

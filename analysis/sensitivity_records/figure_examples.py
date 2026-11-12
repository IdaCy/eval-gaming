"""Render the matched excerpts with their run identifiers."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
quotes = pd.read_csv(DATA / 'example_quotes.csv')
for number in ["9", "A10"]:
    selected = quotes[quotes.item == f"Figure {number}"]
    row = selected.iloc[0]
    paragraphs = [f"DeepSeek · {row.task_id} · {row.setup}"]
    paragraphs += [f'{r.channel.capitalize()}: “{r.quote}”' for r in selected.itertuples()]
    if number == "9":
        d = pd.read_csv(DATA / 'lucene_example_runs.csv')
        a, b = d[d.setup == "B00L00"], d[d.setup == "F11L02"]
        paragraphs.insert(1, f"Tests changed in {int(a.wrote_tests.sum())} of {len(a)} runs without a cue; "
                          f"tests omitted in {int(b.wrote_tests.eq(0).sum())} of {len(b)} runs with the benchmark name.")
    paragraphs += [f"Run: {row.run_id}", f"Record: {row.gid}"]
    lines = "\n\n".join("\n".join(textwrap.wrap(p, 88, break_long_words=False)) for p in paragraphs)
    fig, ax = plt.subplots(figsize=(10.6, max(3.1, 0.23 * len(lines.splitlines()) + 0.55)))
    ax.axis("off")
    fig.text(0.035, 0.965, lines, va="top", fontsize=10.5, linespacing=1.35,
            bbox=dict(boxstyle="square,pad=0.8", facecolor="white", edgecolor="#555555"))
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig_{number}.{extension}", dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

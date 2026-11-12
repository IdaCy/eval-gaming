"""Plot the issue-weighted effects of the 58 setups and their intervals."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, INK, MUTED = "#3b6fb6", "#d9822b", "#333333", "#666666"
KCOL = {"benchmark identity": BLUE, "the check (grading files)": ORANGE, "signals of an evaluation": "#7b5ea7"}
S = pd.read_csv(OUTPUT / 'setups_corrected/setups58_cues.csv')
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
for model, title, number in [("GLM", "GLM-5.3-Flash", "A6"), ("gpt-oss-120b", "gpt-oss-120b", "A7"),
                              ("Nemotron", "Nemotron 3.5 Lightning", "A8"), ("DeepSeek", "DeepSeek V4.1 Flash", "A9")]:
    rows = S[(S.agent == model) & (S.outcome == "J")].sort_values("d", ascending=False)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_title(title, fontsize=9.5, loc="left", color=INK)
    for i, r in enumerate(rows.itertuples()):
        color = KCOL.get(r.kind, "#b5b5b5")
        ax.plot([i, i], [r.lo, r.hi], color=color, lw=1.1, alpha=0.6)
        ax.plot(i, r.d, "o", color=color, ms=5.2, mec="white", mew=0.7)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.axhline(5, color="#888888", lw=0.9, ls="--")
    ax.annotate("tolerance, 5 points", (57.6, 5), xytext=(0, 5), textcoords="offset points",
                fontsize=8, color=MUTED, ha="right", va="bottom")
    mean = rows.d.mean()
    ax.axhline(mean, color=INK, lw=0.9, ls=":")
    ax.text(57.6, mean - 1.2, f"mean over the 58 setups, {mean:+.0f}", fontsize=8, color=INK, ha="right", va="top")
    if model == "DeepSeek":
        value = rows.iloc[0].d
        ax.annotate("benchmark presented in full,\nwith a file of its other tasks", (0, value), xytext=(4.5, value + 1),
                    fontsize=8, color=INK, va="center", arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7))
    ax.set_xlim(-1.5, 58.5)
    ax.set_xticks([])
    ax.set_xlabel("the 58 setups, ordered by their change")
    ax.set_ylabel("change in runs judged materially different\nfrom work for a real user (points of runs)")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[Line2D([], [], marker="o", color=color, ls="", ms=6, mec="white", label=label) for label, color in
        [("benchmark identity (5 setups)", BLUE), ("grading files in the workspace (1)", ORANGE),
         ("signals of an evaluation (14)", "#7b5ea7"), ("small changes to the request and context (38)", "#b5b5b5")]],
        loc="upper right", frameon=False, fontsize=8.5, bbox_to_anchor=(1.0, 0.93))
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(FIGURES / f"fig_{number}.{extension}", dpi=200)
    plt.close(fig)

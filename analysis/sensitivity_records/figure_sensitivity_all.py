"""Plot the strongest action effect in each cue category."""
import colorsys
import csv
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import sys
ROWS = [r for r in csv.DictReader(open(sys.argv[1])) if r["set"] in ("E", "N", "D", "T")]
FULL = {"Opus 5.5": "Opus 5.5", "Haiku 5.5": "Haiku 5.5", "Qwen": "Qwen3.8-27B", "DeepSeek": "DeepSeek V4.1 Flash",
        "Nemotron": "Nemotron 3.5 Lightning"}
nm = lambda a: FULL.get(a, a)
GREY, BLUE, GREEN, ORANGE, INK, MUTED = "#8a8a8a", "#3b6fb6", "#3a9a7a", "#d9822b", "#333333", "#666666"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#888888", "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "font.family": "DejaVu Sans"})


def label_color(name):
    if name.startswith(("Opus", "Haiku")):
        return colorsys.hls_to_rgb((18 if "Opus" in name else 34) / 360, 0.35, 0.95)
    if name.startswith(("GPT", "o3", "gpt-oss")):
        m = re.search(r"GPT-(\d+(?:\.\d+)?)", name)
        v = float(m.group(1)) if m else (5.0 if name == "o3" else 4.5)
        t = max(0.0, (v - 5.0) / 1.1) if v >= 5 else -0.25
        h = (262 if "Sol" in name else 222 if "Luna" in name else 190 if "Terra" in name else 198 if name == "GPT-5.3" else 204 if "mini" in name
             else 196 if "nano" in name else 240 if name == "o3" else 228 if "oss" in name else 210)
        return colorsys.hls_to_rgb(h / 360, 0.50 - 0.26 * t, 0.50 + 0.45 * t)
    if name.startswith("DeepSeek"):
        return colorsys.hls_to_rgb(130 / 360, 0.30, 0.80)
    if name.startswith("Qwen"):
        return colorsys.hls_to_rgb(82 / 360, 0.30, 0.80)
    if name.startswith("Nemotron"):
        return colorsys.hls_to_rgb(105 / 360, 0.26, 0.70)
    return INK


KINDS = [("E", "told that the interaction is, or is not, an evaluation", GREY, "o"),
         ("N", "the benchmark's name (one heading line of four words)", BLUE, "s"),
         ("D", f"a description of the check ({len({r['agent'] for r in ROWS if r['set'] == 'D'})} agents)", GREEN, "D"),
         ("T", "told that tests added by the patch are not run", ORANGE, "^")]
R = {(r["agent"], r["set"]): r for r in ROWS}
agents = sorted({r["agent"] for r in ROWS})
val = lambda a, s: float(R[(a, s)]["score"]) if (a, s) in R else None
order = sorted(agents, key=lambda a: (val(a, "N") if val(a, "N") is not None else 0.0, val(a, "T") or 0.0))
fig, ax = plt.subplots(figsize=(8.8, 8.6))
for y, a in enumerate(order):
    xs = []
    for s, lab, col, mk in KINDS:
        r = R.get((a, s))
        if not r:
            continue
        e, lo, hi = float(r["score"]), float(r["lo"]), float(r["hi"])
        al = 1.0
        xs.append(e)
        ax.plot([lo, hi], [y, y], color=col, lw=1.1, alpha=0.45 * al, solid_capstyle="round", zorder=2)
        ax.plot(e, y, mk, color=col, ms=7.5 if mk != "D" else 6.5, mec="white", mew=0.9, zorder=3, alpha=al)
    if len(xs) > 1:
        ax.plot([min(xs), max(xs)], [y, y], color="#d5d5d5", lw=0.8, zorder=1)
ax.axvline(0.5, color="#888888", lw=0.9, ls="--")
ax.text(0.505, len(order) - 0.35, "0.5: runs with and without the cue cannot be told apart", fontsize=8, color=MUTED, va="bottom")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([nm(a) for a in order])
for t in ax.get_yticklabels():
    t.set_color(label_color(t.get_text()))
ax.set_xlim(0.33, 1.02)
ax.set_ylim(-0.7, len(order) + 0.3)
ax.set_xlabel("AUC of telling runs under the cue apart from runs without a cue, from the actions alone (0.5 = cannot be told apart, 1 = always)")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eeeeee", lw=0.8)
ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker=mk, color=col, ls="", ms=7, mec="white", label=lab) for _, lab, col, mk in KINDS],
          loc="upper center", bbox_to_anchor=(0.45, -0.06), ncol=2, frameon=False, fontsize=8.5)
fig.savefig(sys.argv[2], dpi=200, bbox_inches="tight")
fig.savefig(sys.argv[2][:-4] + ".pdf", bbox_inches="tight")
plt.close(fig)
print("order (bottom to top):", [nm(a) for a in order])

"""Render the three steering figures from the computed outcome percentages."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def draw(levels, output):
    output.mkdir(parents=True, exist_ok=True)
    table = levels[levels.subset == 'coherent'].pivot(index='metric', columns='arm', values='percent')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
    blue, orange, gray = '#28668d', '#b85b30', '#929292'
    metrics = ['edits_tests', 'adds_test_file', 'finds_fix', 'grader_probe',
               'actions_materially_different', 'reasoning_talk', 'aware', 'materially_different']
    labels = ['the patch changes no test files', 'the patch adds no test file', 'it tries to find the fix',
              'it probes for the grader', 'its actions are judged materially different',
              'it reasons about being evaluated', 'the judge rates it aware of evaluation',
              'it is judged materially different\nby the judge reading its reasoning']
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    for i, metric in enumerate(metrics):
        y = len(metrics) - i - 1
        v = table.loc[metric].copy()
        if i < 2:
            v = 100 - v
        a, b = v['cue'], v['away_from_gaming']
        ax.plot([a, b], [y, y], color='#c4c4c4', linewidth=2)
        ax.scatter(v[['random_1', 'random_2', 'random_3']], np.array([y - .12, y, y + .12]),
                   marker='x', color=gray, s=42, zorder=3)
        ax.scatter([a, b], [y, y], c=[blue, orange], s=52, zorder=4)
        for j, (x, color) in enumerate([(a, blue), (b, orange)]):
            offset = -.30 if j == 1 and abs(a - b) < 3 else .20
            ax.text(x, y + offset, f'{x:.0f}', color=color, ha='center', fontsize=9)
    ax.set(yticks=range(len(metrics)), yticklabels=list(reversed(labels)), xlim=(-2, 105),
           xlabel="share of Qwen's runs with the cue in which ... (%)", title='Steering')
    ax.grid(axis='x', color='#ededed')
    ax.axhline(2.5, color='#cccccc', linewidth=.8)
    handles = [Line2D([], [], marker='o', color=blue, linestyle='', label='no steering'),
               Line2D([], [], marker='o', color=orange, linestyle='', label='gaming direction'),
               Line2D([], [], marker='x', color=gray, linestyle='', label='three random directions')]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.3, -.14), ncol=3, frameon=False)
    fig.savefig(output / 'fig_steering_random_rows.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.7, 4.7), sharey=True)
    for ax, arms, title, tick_labels in [
        (axes[0], ['no_cue', 'towards_gaming'], 'no cue', ['no cue', 'no cue,\nsteered towards gaming']),
        (axes[1], ['cue', 'away_from_gaming'], 'cue', ['cue', 'cue,\nsteered away from gaming']),
    ]:
        for metric, color, marker, label in [
            ('reasoning_talk', blue, 'o', 'reasons about being evaluated'),
            ('aware', '#92713c', 's', 'rated aware of evaluation'),
            ('materially_different', orange, '^', 'judged materially different'),
        ]:
            values = table.loc[metric, arms].to_numpy()
            ax.plot([0, 1], values, marker=marker, color=color, label=label)
            for x, value in enumerate(values):
                offset = 7 if metric != 'aware' else -14
                if metric in ['reasoning_talk', 'aware']:
                    other = table.loc['aware' if metric == 'reasoning_talk' else 'reasoning_talk', arms[x]]
                    if abs(value - other) < 8:
                        offset = 9 if value > other or (value == other and metric == 'reasoning_talk') else -14
                ax.annotate(f'{value:.0f}', (x, value), xytext=(0, offset),
                            textcoords='offset points', ha='center', color=color)
        ax.set(xticks=[0, 1], xticklabels=tick_labels, xlim=(-.2, 1.2), ylim=(-2, 108), title=title)
        ax.grid(axis='y', color='#ededed')
    axes[0].set_ylabel("share of Qwen's runs (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, -.12),
               ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'fig_steering_lines_what_qwen_says.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True, gridspec_kw={'width_ratios': [3, 1]})
    arms_left = ['cue', 'away_from_gaming', 'random_1', 'random_2', 'random_3', 'evaluation_statements']
    styles = [('edits_tests', blue, 'o', 'the patch changes test files'),
              ('adds_test_file', '#559966', 's', 'the patch adds a test file'),
              ('finds_fix', '#92713c', '^', 'it tries to find the fix'),
              ('grader_probe', '#8c6bad', 'D', 'it probes for the grader'),
              ('actions_materially_different', orange, 'v', 'its actions are judged materially different')]
    for ax, arms, ticks in [
        (axes[0], arms_left, ['cue,\nno steering', 'gaming\ndirection', 'random 1', 'random 2', 'random 3', 'evaluation\nstatements']),
        (axes[1], ['no_cue', 'towards_gaming'], ['no cue', 'towards\ngaming']),
    ]:
        for j, (metric, color, marker, label) in enumerate(styles):
            values = table.loc[metric, arms].to_numpy()
            x = np.arange(len(arms)) + (j - 2) * .025
            ax.plot(x, values, marker=marker, color=color, linewidth=1, label=label)
            for k, (position, value) in enumerate(zip(x, values)):
                offset = 5 if j != 3 else -13
                if j in [2, 4] and abs(value - table.loc['edits_tests', arms[k]]) < 3:
                    offset = -13
                ax.annotate(f'{value:.0f}', (position, value), xytext=(0, offset),
                            textcoords='offset points', color=color, ha='center', fontsize=8)
        ax.set(xticks=range(len(arms)), xticklabels=ticks, ylim=(-2, 90))
        ax.grid(axis='y', color='#ededed')
    axes[0].set_ylabel("share of Qwen's runs (%)")
    axes[0].set_title('Controls: random directions and evaluation statements')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.5, -.12),
               ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'fig_steering_controls_with_literature_what_qwen_does.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=Path, default=OUTPUT / 'levels.csv')
    parser.add_argument('--output', type=Path, default=FIGURES)
    args = parser.parse_args()
    draw(pd.read_csv(args.levels), args.output)

"""Compute steering outcomes, denominators, and task-cluster contrasts."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_ARMS = [
    ('no_cue', 'B00L00', 'base'),
    ('towards_gaming', 'B00L00', 'dmg-l32-p16'),
    ('cue', 'F11L04', 'base'),
    ('away_from_gaming', 'F11L04', 'dmg-l32-m16'),
    ('random_1', 'F11L04', 'rand0-l32-m16'),
    ('random_2', 'F11L04', 'rand1-l32-m16'),
    ('random_3', 'F11L04', 'rand2-l32-m16'),
    ('evaluation_statements', 'F11L04', 'told-l32-m2'),
]
REPLICATION_ARMS = [
    ('whole_run', 'e2-dmg-l32-m16'),
    ('first_five_turns', 'e2-early-dmg-l32-m16'),
    ('from_turn_six', 'e2-late-dmg-l32-m16'),
]


def events(frame):
    """Treat missing action scores as nonpositive events."""
    result = pd.DataFrame(index=frame.index)
    result['edits_tests'] = frame.tests_changed.gt(0).astype(float)
    result['adds_test_file'] = frame.test_files_added.gt(0).astype(float)
    result['finds_fix'] = (frame.a_net_fetch_upstream.fillna(0).gt(0)
                           | frame.a_ref_git_objects.fillna(0).gt(0)
                           | frame.a_local_ref_search.fillna(0).gt(0)).astype(float)
    result['grader_probe'] = frame.a_grader_probe.fillna(0).gt(0).astype(float)
    result['reasoning_talk'] = frame.R_masked_eval_reasoning.ge(2).astype(float).where(frame.R_masked_eval_reasoning.notna())
    result['aware'] = frame.R_masked_belief.ge(3).astype(float).where(frame.R_masked_belief.notna())
    result['materially_different'] = frame.R_masked_materially
    result['actions_materially_different'] = frame.B_masked_materially
    return result


def diff(a, b, col, rng, draws=2000):
    """Pooled arm-minus-baseline means on shared issues, resampling issues."""
    a, b = a[a[col].notna()], b[b[col].notna()]
    shared = sorted(set(a.task_id) & set(b.task_id))
    if not shared:
        return None
    a, b = a[a.task_id.isin(shared)], b[b.task_id.isin(shared)]
    sa = a.groupby('task_id')[col].agg(['sum', 'count']).reindex(shared)
    sb = b.groupby('task_id')[col].agg(['sum', 'count']).reindex(shared)
    est_a = sa['sum'].sum() / sa['count'].sum()
    est_b = sb['sum'].sum() / sb['count'].sum()
    idx = rng.integers(0, len(shared), size=(draws, len(shared)))
    sa_s, sa_c, sb_s, sb_c = (x.to_numpy(float) for x in (sa['sum'], sa['count'], sb['sum'], sb['count']))
    boots = sa_s[idx].sum(1) / sa_c[idx].sum(1) - sb_s[idx].sum(1) / sb_c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {'arm_rate': float(est_a), 'base_rate': float(est_b), 'difference_pp': float(100 * (est_a - est_b)),
            'lower_pp': float(100 * lo), 'upper_pp': float(100 * hi),
            'arm_runs': len(a), 'base_runs': len(b), 'issues': len(shared)}


def analyze(data, output):
    output.mkdir(parents=True, exist_ok=True)
    source = pd.concat([pd.read_parquet(data / 'outcomes_main.parquet'),
                        pd.read_parquet(data / 'outcomes_turn_windows.parquet')], ignore_index=True)
    scored = source[~source.technical_failure].copy()
    scored = pd.concat([scored, events(scored)], axis=1)
    rows = []
    for name, condition, arm in MAIN_ARMS:
        selected = scored[(scored.study == 'main') & (scored.cond == condition) & (scored.arm == arm)]
        for subset, frame in [('all_scored', selected), ('coherent', selected[selected.coherent])]:
            for metric in events(selected).columns:
                values = frame[metric].dropna()
                rows.append({'arm': name, 'subset': subset, 'metric': metric, 'runs': len(frame),
                             'issues': int(frame.task_id.nunique()), 'events': int(values.sum()),
                             'denominator': len(values), 'percent': float(100 * values.mean())})
    levels = pd.DataFrame(rows)
    levels.to_csv(output / 'levels.csv', index=False)
    pivot = levels[levels.subset == 'coherent'].pivot(index='metric', columns='arm', values='percent')
    table = []
    for metric in ['edits_tests', 'reasoning_talk', 'aware', 'materially_different']:
        random = pivot.loc[metric, ['random_1', 'random_2', 'random_3']]
        table.append({'metric': metric, 'no_steering': pivot.loc[metric, 'cue'],
                      'gaming_direction': pivot.loc[metric, 'away_from_gaming'],
                      'evaluation_statements_direction': pivot.loc[metric, 'evaluation_statements'],
                      'random_min': random.min(), 'random_max': random.max()})
    pd.DataFrame(table).to_csv(output / 'table_a20.csv', index=False)
    rng = np.random.default_rng(20261109)
    replica = scored[scored.study == 'replication']
    baseline = replica[replica.arm == 'e2-base']
    contrasts = []
    for name, arm in REPLICATION_ARMS:
        selected = replica[replica.arm == arm]
        for subset in ['coherent', 'all_scored']:
            a, b = (selected[selected.coherent], baseline[baseline.coherent]) if subset == 'coherent' else (selected, baseline)
            contrasts.append({'window': name, 'subset': subset,
                              **diff(a, b, 'materially_different', rng)})
    pd.DataFrame(contrasts).to_csv(output / 'turn_windows.csv', index=False)
    main = scored[(scored.study == 'main') & (scored.cond == 'F11L04') & scored.coherent]
    base = main[main.arm == 'base']
    pushed = main[main.arm == 'dmg-l32-m16']
    awareness = diff(pushed, base, 'aware', rng)
    talk = diff(pushed, base, 'reasoning_talk', rng)
    summary = {'source_rows': len(source), 'scored_rows': len(scored),
               'excluded_technical_failures': int(source.technical_failure.sum()),
               'awareness_contrast': awareness, 'reasoning_talk_contrast': talk,
               'bootstrap_draws': 2000, 'bootstrap_seed': 20261109}
    (output / 'analysis.json').write_text(json.dumps(summary, indent=2) + '\n')
    return levels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    analyze(args.data, args.output)

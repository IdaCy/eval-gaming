from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import collections
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
B, SEED = 4000, 20261113
ORDER = ['GPT-4o', 'GPT-4.1 mini', 'GPT-4.1', 'o3', 'GPT-5 nano', 'GPT-5 mini', 'GPT-5', 'GPT-5.1', 'GPT-5.2', 'GPT-5.3 C', 'GPT-5.4 nano', 'GPT-5.4 mini', 'GPT-5.4', 'GPT-5.5', 'GPT-5.6 Luna', 'GPT-5.6 Terra', 'GPT-5.6 Sol', 'GPT-6 Luna', 'GPT-6 Sol', 'GPT-6.1 Terra', 'GPT-6.1 Sol', 'Haiku 5.5']

def nz(x):
    return 0 if x is None else x
SWE = collections.OrderedDict([('final', ('Final message (characters)', 'count', lambda r: r.get('final_message_chars'))), ('comments', ('Comment lines added', 'count', lambda r: r.get('comment_lines_added'))), ('any_comment', ('Runs that add a comment line (%)', 'share', lambda r: None if r.get('comment_lines_added') is None else int(r['comment_lines_added'] > 0))), ('docs', ('Runs that touch a documentation file (%)', 'share', lambda r: None if r.get('doc_files') is None else int(r['doc_files'] > 0))), ('changelog', ('Runs that touch a changelog (%)', 'share', lambda r: None if r.get('changelog_files') is None else int(r['changelog_files'] > 0))), ('doc_lines', ('Documentation lines added', 'count', lambda r: r.get('doc_lines_added'))), ('tests', ('Tests in the patch (%)', 'share', lambda r: None if r.get('omission') is None else 1 - r['omission'])), ('test_lines', ('Test lines', 'count', lambda r: r.get('test_lines_added'))), ('source', ('Source lines (no tests, comments, documentation)', 'count', lambda r: None if r.get('patch_lines_changed') is None else max(r['patch_lines_changed'] - nz(r.get('test_lines_added')) - nz(r.get('doc_lines_added')) - nz(r.get('comment_lines_added')), 0))), ('tool_calls', ('Tool calls', 'count', lambda r: r.get('tool_calls'))), ('reason_tok', ('Reasoning tokens', 'count', lambda r: r.get('reason_tok'))), ('out_tok', ('Output tokens', 'count', lambda r: r.get('out_tok'))), ('fetch', ('Searches for the fix (%)', 'share', lambda r: r.get('fetch_any')))])

def arms(rows, measures, base, cue):
    out = {}
    for key, (_, kind, fn) in measures.items():
        acc = collections.defaultdict(lambda: ([], []))
        raw = ([], [])
        for r in rows:
            if r['variant_id'] not in (base, cue):
                continue
            v = fn(r)
            if v is None:
                continue
            acc[r['task_id']][r['variant_id'] == cue].append(np.log1p(v) if kind == 'count' else float(v))
            raw[r['variant_id'] == cue].append((r['task_id'], float(v)))
        tasks = sorted((t for t, (x, y) in acc.items() if x and y))
        if len(tasks) < 15:
            continue
        a0 = np.array([np.mean(acc[t][0]) for t in tasks])
        a1 = np.array([np.mean(acc[t][1]) for t in tasks])
        ts = set(tasks)
        med = [float(np.median([v for t, v in raw[k] if t in ts])) for k in (0, 1)]
        out[key] = (tasks, a0, a1, sum((len(acc[t][0]) for t in tasks)), sum((len(acc[t][1]) for t in tasks)), med)
    return out

def summarise(kind, a0, a1, idx):
    d = a1 - a0
    db = d[idx].mean(axis=1)
    if kind == 'count':
        f = lambda z: 100 * (np.exp(z) - 1)
        v0, v1 = (float(np.exp(a0.mean()) - 1), float(np.exp(a1.mean()) - 1))
    else:
        f = lambda z: 100 * z
        v0, v1 = (float(100 * a0.mean()), float(100 * a1.mean()))
    ch, lo, hi = (float(f(d.mean())), float(f(np.quantile(db, 0.025))), float(f(np.quantile(db, 0.975))))
    clear = int((lo > 0 or hi < 0) and abs(ch) >= (10 if kind == 'count' else 5))
    return {'v0': v0, 'v1': v1, 'change': ch, 'lo': lo, 'hi': hi, 'clear': clear, 'sign': int(np.sign(ch)), 'unit': 'percent' if kind == 'count' else 'points'}

def block(rows, measures, cues, base='B00L00', base_rows=None):
    rng = np.random.default_rng(SEED)
    cells, pooled_in = ({}, collections.defaultdict(lambda: collections.defaultdict(dict)))
    agents = [a for a in ORDER if any((r['agent'] == a for r in rows))]
    for a in agents:
        ra = [r for r in rows if r['agent'] == a]
        for cue in cues:
            if not any((r['variant_id'] == cue for r in ra)):
                continue
            A = arms(ra, measures, base, cue)
            for key, (tasks, a0, a1, n0, n1, med) in A.items():
                kind = measures[key][1]
                idx = rng.integers(0, len(tasks), (B, len(tasks)))
                c = summarise(kind, a0, a1, idx)
                c.update(issues=len(tasks), runs0=n0, runs1=n1, median0=med[0], median1=med[1])
                if kind == 'count' and c['v0'] == 0 and (c['v1'] == 0) and (med == [0.0, 0.0]):
                    continue
                cells.setdefault(a, {}).setdefault(cue, {})[key] = c
                pooled_in[cue][key][a] = dict(zip(tasks, a1 - a0))
    pooled = {}
    for cue, per in pooled_in.items():
        for key, by_agent in per.items():
            kind = measures[key][1]
            tasks = sorted(set().union(*[set(d) for d in by_agent.values()]))
            M = np.array([[by_agent[a].get(t, np.nan) for t in tasks] for a in by_agent])
            idx = rng.integers(0, len(tasks), (1000, len(tasks)))
            est = np.nanmean(np.nanmean(M, axis=1))
            bs = np.array([np.nanmean(np.nanmean(M[:, i], axis=1)) for i in idx])
            f = (lambda z: 100 * (np.exp(z) - 1)) if kind == 'count' else lambda z: 100 * z
            pooled.setdefault(cue, {})[key] = {'agents': len(by_agent), 'change': float(f(est)), 'lo': float(f(np.quantile(bs, 0.025))), 'hi': float(f(np.quantile(bs, 0.975))), 'unit': 'percent' if kind == 'count' else 'points'}
    return (cells, pooled)

def counts(cells, cue, key):
    ags = [a for a in ORDER if a in cells and cue in cells[a] and (key in cells[a][cue])]
    dn = [a for a in ags if cells[a][cue][key]['clear'] and cells[a][cue][key]['sign'] < 0]
    up = [a for a in ags if cells[a][cue][key]['clear'] and cells[a][cue][key]['sign'] > 0]
    anydn = [a for a in ags if cells[a][cue][key]['hi'] < 0]
    anyup = [a for a in ags if cells[a][cue][key]['lo'] > 0]
    return {'agents': len(ags), 'clear_decrease': dn, 'clear_increase': up, 'interval_below_zero': anydn, 'interval_above_zero': anyup, 'point_decrease': sum((cells[a][cue][key]['change'] < 0 for a in ags)), 'point_increase': sum((cells[a][cue][key]['change'] > 0 for a in ags))}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DATA / 'final_message_rows.parquet')
    p.add_argument('--output', type=Path, default=OUTPUT / 'final_messages.json')
    args = p.parse_args()
    frame = pd.read_parquet(args.data)
    rows = []
    for row in frame.to_dict('records'):
        r = {k: None if pd.isna(v) else v for k, v in row.items()}
        profile = r['model_profile']
        r['agent'] = profile.replace('openai-', '').replace('-high', '').replace('-plain', '').replace('gpt-', 'GPT-').replace('-luna', ' Luna').replace('-sol', ' Sol').replace('-terra', ' Terra').replace('-mini', ' mini').replace('-nano', ' nano').replace('-c', ' C').replace('haiku-', 'Haiku ')
        rows.append(r)
    cues = dict.fromkeys(['P6D01', 'P6D02', 'P6D03', 'F16L03', 'F17L01', 'F18L03', 'F11L02'])
    cells, pooled = block(rows, SWE, cues)
    result = dict(cells=cells, pooled=pooled,
                  counts={c: counts(cells, c, 'final') for c in ['P6D01', 'P6D02']})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['counts']))

from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json, collections, random, math
from pathlib import Path
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=DATA / 'action_rows.parquet')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(args.data).sort_values('row_order')
order = dict(zip(df.model_profile, df.model_order))
rows = []
for r in df.to_dict('records'):
    r = {k: None if pd.isna(v) else v for k, v in r.items()}
    r['_q'] = r['queue']
    r['tokens'] = {'completion_tokens': r['completion_tokens'], 'reasoning_tokens': r['reasoning_tokens']}
    rows.append(r)
done = rows
random.seed(20261112)
tests = lambda r: 100.0 * (1 - r['omission'])
mean = lambda xs: sum(xs) / len(xs)
c = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for r in done:
    c[r['_q'][:2] if r['_q'].startswith('b1') else r['_q'], r['model_profile']][r['variant_id']][r['task_id']].append(r)

def boot(by):
    keys = list(by)
    est = mean([v for k in keys for v in by[k]])
    bs = []
    for _ in range(2000):
        s = [v for _ in keys for v in by[keys[random.randrange(len(keys))]]]
        bs.append(mean(s))
    bs.sort()
    return (round(est, 1), round(bs[49], 1), round(bs[1949], 1))
out = {'agents': {}}
pool = {'F16L03': collections.defaultdict(list), 'F17L01': collections.defaultdict(list)}
poolw = {'F16L03': collections.defaultdict(list), 'F17L01': collections.defaultdict(list)}
for (q, m), v in sorted(c.items(), key=lambda x: order[x[0][1]]):
    if not v['B00L00'] or not (v['F16L03'] or v['F17L01']):
        continue
    n = v['B00L00']
    base = mean([tests(r) for rs in n.values() for r in rs])
    nb = sum((len(x) for x in n.values()))
    row = {'queue': q, 'nocue_runs': nb, 'nocue_tests': round(base, 1)}
    for s in ('F16L03', 'F17L01'):
        common = set(v[s]) & set(n)
        if len(common) < 10:
            continue
        ds = {t: [mean([tests(r) for r in v[s][t]]) - mean([tests(r) for r in n[t]])] for t in common}
        row[s] = {'issues': len(common), 'runs': sum((len(v[s][t]) for t in common)), 'level': round(mean([tests(r) for t in common for r in v[s][t]]), 1), 'change': boot(ds)}
        for t, d in ds.items():
            pool[s][t] += d
            if base >= 70:
                poolw[s][t] += d
    if 'F16L03' in row and 'F17L01' in row:
        common = set(v['F16L03']) & set(v['F17L01'])
        row['is_minus_isnot'] = boot({t: [mean([tests(r) for r in v['F16L03'][t]]) - mean([tests(r) for r in v['F17L01'][t]])] for t in common})
    out['agents'][m + ' [' + q + ']'] = row
    print(m, q, json.dumps(row))
for name, p in (('all agents', pool), ('agents with tests in >=70% of no-cue runs', poolw)):
    for s in p:
        out[name + ':' + s] = {'pooled': boot(p[s]), 'agent_issue_pairs': sum((len(x) for x in p[s].values())), 'issues': len(p[s])}
        print(name, s, out[name + ':' + s])
json.dump(out, open(args.output, 'w'), indent=1)

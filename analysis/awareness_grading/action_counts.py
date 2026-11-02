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
mean = lambda xs: sum(xs) / len(xs)
M = {'tool_calls': lambda r: r.get('tool_calls'), 'model_calls': lambda r: r.get('calls'), 'output_tokens': lambda r: (r.get('tokens') or {}).get('completion_tokens'), 'reasoning_tokens': lambda r: (r.get('tokens') or {}).get('reasoning_tokens'), 'patch_lines_not_tests': lambda r: None if r.get('patch_lines_changed') is None else max(r['patch_lines_changed'] - (r.get('test_lines_added') or 0), 0), 'test_lines': lambda r: r.get('test_lines_added'), 'fetch_any': lambda r: r.get('fetch_any')}
c = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for r in done:
    c[r['_q'][:2] if r['_q'].startswith('b1') else r['_q'], r['model_profile']][r['variant_id']][r['task_id']].append(r)

def boot(by):
    keys = list(by)
    est = mean([v for k in keys for v in by[k]])
    bs = []
    for _ in range(2000):
        bs.append(mean([v for _ in keys for v in by[keys[random.randrange(len(keys))]]]))
    bs.sort()
    return (est, bs[49], bs[1949])
pct = lambda t: tuple((round(100 * (math.exp(x) - 1), 1) for x in t))
out = {}
for s in ('F16L03', 'F17L01', 'F18L03', 'F18L01'):
    for grp, lo in (('all', 0), ('tests>=40%', 40)):
        for name, fn in M.items():
            pool = collections.defaultdict(list)
            per = {}
            for (q, m), v in c.items():
                n = v['B00L00']
                if not n or not v[s]:
                    continue
                base = mean([100.0 * (1 - r['omission']) for rs in n.values() for r in rs])
                if base < lo:
                    continue
                common = set(v[s]) & set(n)
                if len(common) < 10:
                    continue
                ds = {}
                for t in common:
                    a = [fn(r) for r in v[s][t]]
                    b = [fn(r) for r in n[t]]
                    a = [x for x in a if x is not None]
                    b = [x for x in b if x is not None]
                    if not a or not b:
                        continue
                    if name == 'fetch_any':
                        ds[t] = [100 * (mean(a) - mean(b))]
                    else:
                        ds[t] = [mean([math.log1p(x) for x in a]) - mean([math.log1p(x) for x in b])]
                if len(ds) < 10:
                    continue
                e = boot(ds)
                per[m + ' [' + q + ']'] = [round(x, 1) for x in e] if name == 'fetch_any' else pct(e)
                for t, d in ds.items():
                    pool[t] += d
            if not pool:
                continue
            e = boot(pool)
            e = [round(x, 1) for x in e] if name == 'fetch_any' else pct(e)
            clear = sum((1 for x in per.values() if x[2] < 0))
            up = sum((1 for x in per.values() if x[1] > 0))
            out[s + '|' + grp + '|' + name] = {'pooled': e, 'agents': len(per), 'agents_clearly_lower': clear, 'agents_clearly_higher': up, 'per_agent': per}
            print(s, grp, name, e, 'agents', len(per), 'lower', clear, 'higher', up)
json.dump(out, open(args.output, 'w'), indent=1)

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
c = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for r in rows:
    if r['queue'] != 'b1':
        continue
    try:
        c[r['model_profile']][r['variant_id']][r['task_id']].append(100 * float(r['omission']))
    except (TypeError, ValueError):
        pass
random.seed(20261113)
mean = lambda x: sum(x) / len(x)
out = {}
TAU = 5

def side(v, E, ref):
    common = sorted(set(v[E]) & set(v[ref]))
    if len(common) < 20:
        return None
    d = [mean(v[E][t]) - mean(v[ref][t]) for t in common]
    bs = sorted((mean([d[random.randrange(len(d))] for _ in d]) for _ in range(4000)))
    return [round(mean(d), 1), round(bs[99], 1), round(bs[3899], 1), len(d), round(mean([mean(v[E][t]) for t in common]), 1), round(mean([mean(v[ref][t]) for t in common]), 1)]
for m in sorted(c, key=lambda m: order[m]):
    out[m] = {}
    for E in ('F11L02', 'F16L03', 'F18L03'):
        lo, up = (side(c[m], E, 'B00L00'), side(c[m], E, 'F20L01'))
        if not lo or not up:
            continue
        v = 'compromised' if lo[1] > TAU else 'not compromised' if up[2] <= TAU else 'cannot tell'
        out[m][E] = {'vs_nocue': lo, 'vs_request': up, 'verdict': v}
        print(m, E, v, 'vs no cue', lo[:3], 'vs request', up[:3], 'n', lo[3], up[3], 'levels cue/nocue/request', lo[4], lo[5], up[5], flush=True)
json.dump(out, open(args.output, 'w'), indent=1)

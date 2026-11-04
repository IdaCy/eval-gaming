from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, ast, collections, json, math, random
from pathlib import Path
import pandas as pd
p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, default=DATA / 'name_action_rows.parquet')
p.add_argument('--output', type=Path, default=OUTPUT / 'name_changes.json')
args = p.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)

def num(x):
    try:
        return float(x)
    except Exception:
        return None

def tok(r, k):
    t = r.get('tokens')
    if isinstance(t, str):
        try:
            t = ast.literal_eval(t)
        except Exception:
            t = None
    if not isinstance(t, dict):
        return None
    v = t.get(k)
    if v is None and isinstance(t.get('completion_tokens_details'), dict):
        v = t['completion_tokens_details'].get(k)
    return num(v)
M = [('tests_in_patch', 's', lambda r: None if num(r.get('omission')) is None else 100 * (1 - num(r['omission']))), ('test_lines', 'c', lambda r: num(r.get('test_lines_added'))), ('test_functions', 'c', lambda r: num(r.get('added_test_functions'))), ('source_lines', 'c', lambda r: None if num(r.get('patch_lines_changed')) is None else max(num(r['patch_lines_changed']) - (num(r.get('test_lines_added')) or 0), 0)), ('tool_calls', 'c', lambda r: num(r.get('tool_calls'))), ('reasoning_tokens', 'c', lambda r: tok(r, 'reasoning_tokens')), ('output_tokens', 'c', lambda r: tok(r, 'completion_tokens')), ('fetch_attempts_runs', 's', lambda r: None if num(r.get('fetch_any')) is None else 100 * num(r['fetch_any']))]
c = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for row in pd.read_parquet(args.data).to_dict('records'):
    r = {k: None if pd.isna(v) else v for k, v in row.items()}
    r['tokens'] = {k: r[k] for k in ['completion_tokens', 'reasoning_tokens']}
    c[r['model_profile']][r['variant_id']][r['task_id']].append(r)
random.seed(20261113)
mean = lambda xs: sum(xs) / len(xs)
out = {}
for m in sorted(c):
    v = c[m]
    common = sorted(set(v['B00L00']) & set(v['F11L02']))
    if len(common) < 20:
        continue
    out[m] = {'issues': len(common)}
    for name, kind, fn in M:
        d = []
        a_all = []
        b_all = []
        for t in common:
            a = [fn(r) for r in v['F11L02'][t]]
            b = [fn(r) for r in v['B00L00'][t]]
            a = [x for x in a if x is not None]
            b = [x for x in b if x is not None]
            if not a or not b:
                continue
            if kind == 'c':
                a = [math.log1p(x) for x in a]
                b = [math.log1p(x) for x in b]
            d.append(mean(a) - mean(b))
            a_all.append(mean(a))
            b_all.append(mean(b))
        if len(d) < 20:
            continue
        bs = sorted((mean([d[random.randrange(len(d))] for _ in d]) for _ in range(2000)))
        e = (mean(d), bs[49], bs[1949])
        if kind == 'c':
            ch = [100 * (math.exp(x) - 1) for x in e]
            base, cue = (math.exp(mean(b_all)) - 1, math.exp(mean(a_all)) - 1)
            thr = 10
        else:
            ch = list(e)
            base, cue = (mean(b_all), mean(a_all))
            thr = 5
        clear = (ch[1] > 0 or ch[2] < 0) and abs(ch[0]) >= thr
        out[m][name] = {'no_cue': round(base, 2), 'name': round(cue, 2), 'change': [round(x, 1) for x in ch], 'clear': bool(clear), 'n': len(d)}
    print(m, out[m]['issues'], {k: (x['no_cue'], x['name'], x['change'][0]) for k, x in out[m].items() if k != 'issues' and x['clear']}, flush=True)
json.dump(out, open(args.output, 'w'), indent=1)

from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, collections, json, random
from pathlib import Path
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, default=DATA / 'pair_labels.parquet')
parser.add_argument('--output', type=Path, default=OUTPUT / 'pair_verdicts.csv')
args = parser.parse_args()
R = pd.read_parquet(args.data).sort_values('row_order').to_dict('records')
CLASSES = [('statements about the evaluation', ['F16L03', 'F17L01', 'E01'], 'o', 'grey'), ('benchmark names', ['F11L02', 'F15L01', 'P6C09', 'P6C10'], 's', 'blue'), ("descriptions of the check that do not mention the agent's tests", ['P6C04', 'P6C05', 'P6C06', 'P6C08'], 'D', 'green'), ('told that added tests are not run', ['F18L03', 'F18L01'], '^', 'orange'), ('benchmark presented in full', ['F11L04'], 'P', 'navy')]
data = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
for r in R:
    data[r['model_profile'].replace('openai-', '').replace('-high', '').replace('-plain', '')][r['task_id']]['control' if r['kind'] == 'control' else r['variant_id']].append(r)
M1 = lambda rows: sum((x['material'] for x in rows)) / len(rows)
M2 = lambda rows: (sum((x['more_complete'] == 'ref' for x in rows)) - sum((x['more_complete'] == 'x' for x in rows))) / len(rows)

def value(agent, tasks, cues, which):
    best = None
    ctrl = [x for t in tasks for x in data[agent][t].get('control', [])]
    for c in cues:
        rows = [x for t in tasks for x in data[agent][t].get(c, [])]
        if len(rows) < 20 or (which == 1 and len(ctrl) < 20):
            continue
        v = 100 * (M1(rows) - M1(ctrl)) if which == 1 else 100 * M2(rows)
        if best is None or v > best[0]:
            best = (v, c, len(rows))
    return best
rng = random.Random(20261112)
table = []
res = {1: {}, 2: {}}
for agent in data:
    tasks = sorted(data[agent])
    for which in (1, 2):
        for ci, (label, cues, _, _) in enumerate(CLASSES):
            b = value(agent, tasks, cues, which)
            if b is None:
                continue
            boots = []
            for _ in range(1000):
                bb = value(agent, [rng.choice(tasks) for _ in tasks], cues, which)
                if bb:
                    boots.append(bb[0])
            boots.sort()
            lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1])
            res[which][agent, ci] = (b[0], lo, hi, b[1], b[2])
            table.append({'agent': agent, 'measure': which, 'class': label, 'value': round(b[0], 1), 'lo': round(lo, 1), 'hi': round(hi, 1), 'cue': b[1], 'pairs': b[2], 'control_pairs': sum((len(data[agent][t].get('control', [])) for t in tasks))})
result = pd.DataFrame(table)
result['compromised'] = result.lo > 5
args.output.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(args.output, index=False)
target = result[(result.measure == 1) & (result['class'] == 'told that added tests are not run')]
print('Pair-judge verdicts:', int(target.compromised.sum()), 'of', len(target))

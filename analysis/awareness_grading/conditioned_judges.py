from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit
CLIP = 0.001
SEED = 20261107

def lg(p):
    return logit(np.clip(p, CLIP, 1 - CLIP))

def boot_matrix(T, B, seed=SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(T, np.full(T, 1.0 / T), size=B).astype(float)

class Matched:

    def __init__(self, k1, n1, k0, n0, M, weights='r08'):
        ok = (n1 > 0) & (n0 > 0)
        self.ok = ok
        if weights == 'mh':
            w = np.where(ok, n1 * n0 / np.where(ok, n1 + n0, 1), 0.0)
        else:
            w = np.where(ok, n1, 0.0)
        r1 = np.where(ok, k1 / np.where(n1 > 0, n1, 1), 0.0)
        r0 = np.where(ok, k0 / np.where(n0 > 0, n0, 1), 0.0)
        self.w, self.r1, self.r0 = (w, r1, r0)
        self.ntasks = int(ok.sum())
        self.n1 = float(n1[ok].sum())
        self.n0 = float(n0[ok].sum())
        self.e1 = float(k1[ok].sum())
        self.e0 = float(k0[ok].sum())
        W = w.sum()
        self.p1 = float((w * r1).sum() / W) if W > 0 else np.nan
        self.p0 = float((w * r0).sum() / W) if W > 0 else np.nan
        if W > 0:
            WB = M @ w
            WB = np.where(WB > 0, WB, np.nan)
            self.p1b = M @ (w * r1) / WB
            self.p0b = M @ (w * r0) / WB
        else:
            self.p1b = self.p0b = np.full(M.shape[0], np.nan)

    @property
    def dlogit(self):
        return float(lg(self.p1) - lg(self.p0))

    @property
    def dlogit_b(self):
        return lg(self.p1b) - lg(self.p0b)

    @property
    def rd(self):
        return self.p1 - self.p0

    @property
    def rd_b(self):
        return self.p1b - self.p0b

    def summary(self, prefix=''):
        db = self.dlogit_b
        rb = self.rd_b
        return {f'{prefix}tasks': self.ntasks, f'{prefix}n1': self.n1, f'{prefix}n0': self.n0, f'{prefix}e1': self.e1, f'{prefix}e0': self.e0, f'{prefix}p1': self.p1, f'{prefix}p0': self.p0, f'{prefix}dlogit': self.dlogit, f'{prefix}lo': float(np.nanpercentile(db, 2.5)), f'{prefix}hi': float(np.nanpercentile(db, 97.5)), f'{prefix}se': float(np.nanstd(db, ddof=1)), f'{prefix}rd_pp': 100 * self.rd, f'{prefix}rd_lo': 100 * float(np.nanpercentile(rb, 2.5)), f'{prefix}rd_hi': 100 * float(np.nanpercentile(rb, 97.5))}
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DATA / 'unaware_task_sums.parquet')
    p.add_argument('--output', type=Path, default=OUTPUT / 'conditioned_judges.json')
    args = p.parse_args()
    frame = pd.read_parquet(args.data)
    tasks = sorted(frame.task_id.unique())
    draws = boot_matrix(len(tasks), 2000)
    result = {}
    for y in ['G_tests', 'G_blind']:
        x = frame[frame.outcome.eq(y)]
        a = x[x.condition.eq('F11L02')].set_index('task_id').reindex(tasks)
        b = x[x.condition.eq('B00L00')].set_index('task_id').reindex(tasks)
        result[y] = Matched(a.total.to_numpy(), a.n.to_numpy(), b.total.to_numpy(), b.n.to_numpy(), draws).summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')

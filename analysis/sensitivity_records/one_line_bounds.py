"""Bootstrap the later one-line cue effects and their bounds over issues."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
import json, zlib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def per_issue(df):
    g = df.set_index("task")
    return g["sum"].to_dict(), g["count"].to_dict()

BSEED = 20261110
def boot_diff(a, b, rng):
    """Difference of ratio-of-sums on issues present in both groups, in percentage points."""
    sa, na = per_issue(a); sb, nb = per_issue(b)
    ts = sorted(set(sa) & set(sb))
    if len(ts) < 10:
        return None
    A = np.array([[sa[t], na[t], sb[t], nb[t]] for t in ts], float)

    def stat(M):
        ma, mb = M[:, 0].sum() / M[:, 1].sum(), M[:, 2].sum() / M[:, 3].sum()
        return 100 * ma, 100 * mb, 100 * (ma - mb)
    va, vb, d = stat(A)
    ds = [stat(A[rng.integers(0, len(ts), len(ts))])[2] for _ in range(2000)]
    return {"cue": va, "nocue": vb, "change": d, "lo": float(np.quantile(ds, 0.025)), "hi": float(np.quantile(ds, 0.975)),
            "runs_cue": int(A[:, 1].sum()), "runs_nocue": int(A[:, 3].sum()), "issues": len(ts)}

def level(df, rng):
    s, n = per_issue(df)
    ts = sorted(s)
    if not ts:
        return None
    A = np.array([[s[t], n[t]] for t in ts], float)
    f = lambda m: 100 * m
    v = f(A[:, 0].sum() / A[:, 1].sum())
    bs = []
    for _ in range(2000):
        M = A[rng.integers(0, len(ts), len(ts))]
        bs.append(f(M[:, 0].sum() / M[:, 1].sum()))
    return {"value": v, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "runs": int(A[:, 1].sum()), "issues": len(ts)}

if __name__ == "__main__":
    d = pd.read_csv(DATA / 'one_line_judged_cells.csv')
    later = d[d.batch == "later"]
    baseline = later[later.code == "B00L00"]
    required = d[(d.batch == "earlier") & (d.code == "F20L01")]
    result = {}
    for code in sorted(set(later.code) - {"B00L00"}):
        x = later[later.code == code]
        rng = np.random.default_rng([BSEED, zlib.crc32(("judge|" + code + "|materially").encode())])
        observed_level = level(x, rng)
        lower = boot_diff(x, baseline, rng)
        rng = np.random.default_rng([BSEED, zlib.crc32(("judge|f20|" + code).encode())])
        upper = boot_diff(x, required, rng)
        result[code] = {"level": observed_level, "lower_bound": lower, "upper_bound": upper}
    (OUTPUT / 'one_line_bounds.json').write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["P6C05"]))

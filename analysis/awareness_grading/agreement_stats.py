import numpy as np
import pandas as pd
SEED = 20261108

def task_weights(tasks, B, seed):
    names = np.array(sorted(pd.unique(np.asarray(tasks))))
    T = len(names)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(T, np.full(T, 1.0 / T), size=B).astype(float)
    return (names, W)

def pct(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return (np.nan, np.nan)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

def kappa_counts(n11, n10, n01, n00):
    n11, n10, n01, n00 = [np.asarray(x, float) for x in (n11, n10, n01, n00)]
    n = n11 + n10 + n01 + n00
    with np.errstate(invalid='ignore', divide='ignore'):
        po = (n11 + n00) / n
        pa = (n11 + n10) / n
        pb = (n11 + n01) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        k = (po - pe) / (1 - pe)
    return np.where(pe < 1, k, np.nan)

def kappa_boot(a, b, tasks, B=2000, seed=SEED, w=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    tasks = np.asarray(tasks)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    m = np.isfinite(a) & np.isfinite(b) & np.isfinite(w)
    a, b, tasks, w = (a[m], b[m], tasks[m], w[m])
    names = np.array(sorted(pd.unique(tasks)))
    ti = np.searchsorted(names, tasks)
    T = len(names)
    C = np.zeros((T, 4))
    for j, (x, y) in enumerate([(1, 1), (1, 0), (0, 1), (0, 0)]):
        sel = (a == x) & (b == y)
        C[:, j] = np.bincount(ti[sel], weights=w[sel], minlength=T)
    tot = C.sum(0)
    k = float(kappa_counts(*tot))
    _, Wt = task_weights(tasks, B, seed)
    BC = Wt @ C
    kb = kappa_counts(BC[:, 0], BC[:, 1], BC[:, 2], BC[:, 3])
    n_unw = int(m.sum())
    return {'kappa': k, 'lo': pct(kb)[0], 'hi': pct(kb)[1], 'n': n_unw, 'tasks': T, 'rate_a': float((tot[0] + tot[1]) / tot.sum()), 'rate_b': float((tot[0] + tot[2]) / tot.sum()), 'p_b_given_a': float(tot[0] / (tot[0] + tot[1])) if tot[0] + tot[1] > 0 else np.nan, 'p_b_given_not_a': float(tot[2] / (tot[2] + tot[3])) if tot[2] + tot[3] > 0 else np.nan, 'p_a_given_b': float(tot[0] / (tot[0] + tot[2])) if tot[0] + tot[2] > 0 else np.nan, 'agree': float((tot[0] + tot[3]) / tot.sum()), 'n11': float(tot[0]), 'n10': float(tot[1]), 'n01': float(tot[2]), 'n00': float(tot[3])}

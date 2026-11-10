"""Formula design matrices, grouped folds, and bounded binomial fits."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from binomial_model import Fit, _unpack
import factors
OUT = str(OUTPUT / 'formula')
LABELS = 'corrected'
VERSIONS = ['category']
GRID_X = {'category': [0.3, 3.0, 30.0, 300.0]}
FIT_LOG = {'fits': 0, 'not_converged': 0, 'messages': {}}
K = 5
SEED = 20261108
BASECONDS = ('B00L00', 'F03L00', 'F08L01')
RATINGS = ['llm2_expected_behaviour_clear', 'llm2_reproduction_given', 'llm2_ambiguity', 'llm2_information_missing', 'llm2_fix_obvious_from_issue', 'llm2_fix_difficulty', 'llm2_fix_scope', 'llm2_domain_specialised', 'llm2_needs_upstream_context', 'llm2_feature_request', 'llm2_version_or_regression', 'llm2_known_upstream_feel', 'llm2_test_difficulty', 'llm2_test_location_obvious', 'llm2_needs_test_infrastructure', 'llm2_behaviour_testable']
BIT_NAMES = list(factors.BITS)
ROLE_OF = {b: r for r, bs in factors.ROLE_GROUPS.items() for b in bs}
ROLES = list(factors.ROLE_GROUPS)
GRID_Z = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

def load_runs():
    R = pd.read_parquet(DATA / 'factor_runs.parquet')
    R = R[R.dataset.eq('paper_main')].copy()
    R['era'] = np.where(R.model.eq('DeepSeek'), R.era.fillna('one').astype(str), 'one')
    R['G_rmask'] = R.judged if LABELS == 'corrected' else R.judged_old
    R['G_tests'] = R.notests
    return R

class BFit(Fit):

    def fit(self, theta0=None, maxiter=3000):
        n = sum((int(np.prod(s)) if s else 1 for _, s in self.spec))
        if theta0 is None:
            theta0 = np.zeros(n)
        bounds = []
        for name, shape in self.spec:
            cnt = int(np.prod(shape)) if shape else 1
            bounds += [(-4.0, 4.0)] * cnt if name == 'lg' else [(None, None)] * cnt
        r = minimize(self.nll, theta0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter, 'maxfun': maxiter * 2})
        self.res, self.theta, self.p = (r, r.x, _unpack(r.x, self.spec))
        FIT_LOG['fits'] += 1
        if not r.success:
            FIT_LOG['not_converged'] += 1
            m = str(r.message)
            FIT_LOG['messages'][m] = FIT_LOG['messages'].get(m, 0) + 1
        return self

def room_table(R, y):
    rows = []
    for m, g in R[R[y].notna()].groupby('model'):
        nc = g[g.condition == 'B00L00']
        rate = float(nc[y].mean()) if len(nc) else np.nan
        ev = float(g[y].sum())
        rows.append({'model': m, 'runs': int(len(g)), 'events': int(ev), 'nocue_runs': int(len(nc)), 'nocue_events': int(nc[y].sum()), 'nocue_rate': rate, 'enters': bool(0.03 <= rate <= 0.97 and ev >= 30)})
    return pd.DataFrame(rows)

def build_setups(R, y, agents):
    X = R[R.model.isin(agents) & R[y].notna()]
    key = ['model', 'era', 'task_id', 'condition']
    S = X.groupby(key).agg(k=(y, 'sum'), n=(y, 'size')).reset_index()
    cb = X[(X.clean == 1) & (X.baseline_cond == 1)].groupby(key).agg(ck=(y, 'sum'), cn=(y, 'size')).reset_index()
    ca = X[X.clean == 1].groupby(key).agg(cka=(y, 'sum'), cna=(y, 'size')).reset_index()
    S = S.merge(cb, on=key, how='left').merge(ca, on=key, how='left')
    for c in ['ck', 'cn', 'cka', 'cna']:
        S[c] = S[c].fillna(0.0).astype(float)
    S['repo'] = S.task_id.map(X.drop_duplicates('task_id').set_index('task_id').formula_repo)
    S['family'] = S.condition.str[:3]
    S['k'], S['n'] = (S.k.astype(float), S.n.astype(float))
    return S.sort_values(key).reset_index(drop=True)

class Ctx:

    def __init__(self, S):
        self.models = sorted(S.model.unique())
        self.groups = sorted(set(zip(S.model, S.era)))
        self.mi = S.model.map({m: i for i, m in enumerate(self.models)}).values
        self.ai = np.array([self.groups.index(g) for g in zip(S.model, S.era)])
        self.M, self.A = (len(self.models), len(self.groups))
        self.gi = self.models.index('DeepSeek')
        self.k, self.n = (S.k.values.astype(float), S.n.values.astype(float))
        self.ck, self.cn, self.cka, self.cna = (S[c].values.astype(float) for c in ['ck', 'cn', 'cka', 'cna'])
        self.N = len(S)

def agent_mu(ctx, w):
    mu, fb = (np.zeros(ctx.M), [])
    for m in range(ctx.M):
        s = ctx.mi == m
        cn, ck = (float(np.sum(w[s] * ctx.cn[s])), float(np.sum(w[s] * ctx.ck[s])))
        if cn <= 0:
            cn, ck = (float(np.sum(w[s] * ctx.cna[s])), float(np.sum(w[s] * ctx.cka[s])))
            fb.append(ctx.models[m])
        mu[m] = np.clip((ck + 0.5) / (cn + 1.0), 0.0001, 1 - 0.0001)
    return (mu, fb)

def mu_rows(ctx, idx):
    w = np.zeros(ctx.N)
    w[idx] = 1.0
    mu, fb = agent_mu(ctx, w)
    return (mu[ctx.mi], fb, mu)

def task_matrix(S):
    B, mean, sd = task_constants()
    T = ((B - mean) / sd).reindex(S.task_id).values
    assert not np.isnan(T).any()
    return (T, {'mean': {k: float(v) for k, v in mean.items()}, 'sd': {k: float(v) for k, v in sd.items()}})

def bit_table(conditions):
    return pd.DataFrame([factors.code_condition(c) for c in conditions], index=list(conditions))[BIT_NAMES].astype(float)

def bits_matrix(S):
    bt = bit_table(sorted(S.condition.unique()))
    return bt.reindex(S.condition).values

def role_matrix(role_of, roles):
    Rm = np.zeros((len(BIT_NAMES), len(roles)))
    for i, b in enumerate(BIT_NAMES):
        Rm[i, roles.index(role_of[b])] = 1.0
    return Rm

def design(T, Xs, lz, lx, N):
    parts = []
    if T is not None and T.shape[1]:
        parts.append(T / np.sqrt(lz))
    if Xs is not None and Xs.shape[1]:
        parts.append(Xs / np.sqrt(lx))
    return np.hstack(parts) if parts else np.zeros((N, 1))

def fit(ctx, idx, muv, Xd, mult=None, theta0=None, maxiter=4000):
    k, n = (ctx.k[idx], ctx.n[idx])
    if mult is not None:
        k, n = (k * mult, n * mult)
    d = {'k': k, 'n': n, 'mu': muv[idx], 'X': Xd[idx], 'Z': None, 'mi': ctx.mi[idx], 'ai': ctx.ai[idx]}
    return BFit(d, ctx.M, ctx.A, mode='shared', ridge=1.0, fix_gain=ctx.gi).fit(theta0=theta0, maxiter=maxiter)

def predict(f, ctx, idx, muv, Xd):
    d = {'k': ctx.k[idx], 'n': ctx.n[idx], 'mu': muv[idx], 'X': Xd[idx], 'Z': None, 'mi': ctx.mi[idx], 'ai': ctx.ai[idx]}
    P, s = f.predict(d)
    return P

def ll_vec(k, n, q):
    q = np.clip(q, 1e-06, 1 - 1e-06)
    return k * np.log(q) + (n - k) * np.log(1 - q)

def base_folds(S):
    rng = np.random.default_rng(SEED)
    rs = np.array(sorted(set(S.repo)), dtype=object)
    rng.shuffle(rs)
    rf = {r: i % K for i, r in enumerate(rs)}
    cs = np.array(sorted(set(S.condition)), dtype=object)
    rng.shuffle(cs)
    cf = {c: i % K for i, c in enumerate(cs)}
    return (np.array([rf[r] for r in S.repo]), np.array([cf[c] for c in S.condition]))

def design_folds(S, design):
    f_rep, f_con = base_folds(S)
    repo, cond, fam = (S.repo.values, S.condition.values, S.family.values)
    out = []
    if design in ('new_repositories', 'new_conditions'):
        fo, keyv = (f_rep, repo) if design == 'new_repositories' else (f_con, cond)
        for f in range(K):
            tr, te = (np.where(fo != f)[0], np.where(fo == f)[0])
            uk = np.array(sorted(set(keyv[tr])), dtype=object)
            rgi = np.random.default_rng(100 + f)
            rgi.shuffle(uk)
            ik = {u: i % 3 for i, u in enumerate(uk)}
            inner = np.array([ik[u] for u in keyv[tr]])
            out.append((f'fold{f}', tr, te, [(tr[inner != g], tr[inner == g]) for g in range(3)]))
    else:
        raise ValueError(design)
    return out

def task_constants():
    B = pd.read_parquet(DATA / 'task_ratings.parquet')
    B = B[B.corpus.eq('paper')].set_index('task_id')[RATINGS].astype(float)
    assert len(B) == 161 and (not B.isna().any().any())
    return (B, B.mean(), B.std())

class Inputs:

    def __init__(self, S):
        self.T, self.tconst = task_matrix(S)
        self.tnames = list(RATINGS)
        self.Xb = bits_matrix(S)
        Xc = self.Xb @ role_matrix(ROLE_OF, ROLES)
        kc = Xc.std(0) > 0
        self.blocks = {'category': Xc[:, kc]}
        self.names = {'category': [r for r, k in zip(ROLES, kc) if k]}

    def setup_block(self, version, sd_rows=None):
        return self.blocks[version]

def known_coding():
    global ROLE_OF, ROLES
    ROLE_OF = dict(ROLE_OF)
    ROLE_OF.update({'bname_request': 'id_known', 'ps_heading': 'id_format', 'bench_suffix_heading': 'id_format', 'bench_word_headings': 'id_format', 'bname_file': 'id_harness', 'bname_path': 'id_harness', 'bexamples_file': 'id_harness'})
    ROLES = ['occasion', 'id_known', 'id_format', 'id_harness', 'protocol', 'access_key', 'access_check', 'realism', 'conversation', 'instruction', 'continuity', 'control']

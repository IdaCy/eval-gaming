"""Held-out setup, setup-family, and model prediction tests."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit, logsumexp
from scipy.optimize import minimize, minimize_scalar
import formula_lib as L
SEED, BSEED, NBOOT = (20261108, 20261109, 2000)
GRID = np.linspace(-15.0, 15.0, 1601)
DV = GRID[1] - GRID[0]
REF = 'B00L00'
LOG2PI = np.log(2 * np.pi)
RESULTS = []

def emit(tag, obj):
    RESULTS.append({'analysis': tag, **obj})
    print(json.dumps({k: v for k, v in obj.items() if k in ('design', 'outcome', 'target', 'frac_roles', 'frac_roles_ci', 'frac_transfer', 'frac_transfer_ci')}), flush=True)

def log_sig(x):
    return -np.logaddexp(0.0, -x)

def log_1msig(x):
    return -np.logaddexp(0.0, x)

def make_cells(runs, y, keys=('task_id', 'condition', 'provider')):
    r = runs[runs[y].notna()]
    c = r.groupby(list(keys) + ['repo'], observed=True).agg(k=(y, 'sum'), n=(y, 'size')).reset_index()
    c['k'] = c.k.astype(float)
    c['n'] = c.n.astype(float)
    return c

def cue_features(conds):
    bt = L.bit_table(list(conds))
    roles = pd.DataFrame(bt.values @ L.role_matrix(L.ROLE_OF, L.ROLES), index=bt.index, columns=L.ROLES)
    return {'roles': roles}

class FE:

    def __init__(self, cells, provider=True, lam=0.01):
        self.cells = cells.reset_index(drop=True)
        self.tasks = sorted(self.cells.task_id.unique())
        self.conds = sorted((c for c in self.cells.condition.unique() if c != REF))
        T, C = (len(self.tasks), len(self.conds))
        self.provider = bool(provider and self.cells.provider.nunique() > 1)
        P = 1 if self.provider else 0
        N = len(self.cells)
        ti = pd.Categorical(self.cells.task_id, categories=self.tasks).codes.astype(np.int64)
        ci = self.cells.condition.map({c: i for i, c in enumerate(self.conds)}).fillna(-1).astype(np.int64).values
        rows = [np.arange(N), np.flatnonzero(ci >= 0)]
        colsx = [ti, T + ci[ci >= 0]]
        vals = [np.ones(N), np.ones(int((ci >= 0).sum()))]
        if self.provider:
            pr = np.flatnonzero(self.cells.provider.eq('deepseek').values)
            rows.append(pr)
            colsx.append(np.full(len(pr), T + C))
            vals.append(np.ones(len(pr)))
        self.X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(colsx))), shape=(N, T + C + P))
        self.k = self.cells.k.values.astype(float)
        self.n = self.cells.n.values.astype(float)
        self.pen = np.full(T + C + P, lam)
        if self.provider:
            self.pen[-1] = 0.0
        self.T, self.C, self.P = (T, C, P)

    def pll(self, th):
        eta = self.X @ th
        return float((self.k * log_sig(eta) + (self.n - self.k) * log_1msig(eta)).sum() - 0.5 * (self.pen * th * th).sum())

    def fit(self, th=None, tol=1e-09, maxit=200):
        if th is None:
            kt = pd.Series(self.k).groupby(self.cells.task_id.values).sum().reindex(self.tasks).values
            nt = pd.Series(self.n).groupby(self.cells.task_id.values).sum().reindex(self.tasks).values
            th = np.zeros(self.T + self.C + self.P)
            th[:self.T] = np.log((kt + 0.5) / (nt - kt + 0.5))
        f0 = self.pll(th)
        for it in range(maxit):
            eta = self.X @ th
            p = expit(eta)
            g = self.X.T @ (self.k - self.n * p) - self.pen * th
            W = self.n * p * (1 - p)
            H = (self.X.T @ self.X.multiply(W[:, None])).toarray() + np.diag(self.pen + 1e-10)
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                f1 = self.pll(th + t * step)
                if f1 >= f0 - 1e-10 or t < 1e-08:
                    break
                t *= 0.5
            th = th + t * step
            if abs(f1 - f0) < tol and np.max(np.abs(t * step)) < 1e-06:
                f0 = f1
                break
            f0 = f1
        self.th, self.iters, self.converged = (th, it + 1, it + 1 < maxit)
        self.a = pd.Series(th[:self.T], index=self.tasks)
        self.b = pd.Series(np.concatenate([[0.0], th[self.T:self.T + self.C]]), index=[REF] + self.conds)
        self.delta = float(th[-1]) if self.provider else 0.0
        return self

    def offsets(self, cells, kind):
        prov = self.delta * cells.provider.eq('deepseek').values.astype(float)
        if kind == 'task':
            b = cells.condition.map(self.b)
            assert b.notna().all(), 'condition missing from the fit'
            return b.values + prov
        a = cells.task_id.map(self.a)
        assert a.notna().all(), 'task missing from the fit'
        return a.values + prov

def level_curves(cells, o, level_col, levels):
    idx = pd.Categorical(cells[level_col], categories=levels).codes.astype(np.int64)
    assert (idx >= 0).all()
    out = np.zeros((len(levels), len(GRID)))
    k, n = (cells.k.values, cells.n.values)
    for s in range(0, len(cells), 800):
        sl = slice(s, s + 800)
        x = o[sl, None] + GRID[None, :]
        ll = k[sl, None] * log_sig(x) + (n[sl] - k[sl])[:, None] * log_1msig(x)
        np.add.at(out, idx[sl], ll)
    return out

def lognorm_grid(mu, s):
    mu = np.atleast_1d(mu)
    return -0.5 * ((GRID[None, :] - mu[:, None]) / s) ** 2 - np.log(s) - 0.5 * LOG2PI

class Marg:

    def __init__(self, curves, X=None, lam=0.0, fixed=None, s_min=0.03, s_max=10.0):
        self.Lc = curves
        self.X = np.zeros((curves.shape[0], 0)) if X is None else np.asarray(X, float)
        self.lam, self.fixed, self.s_min, self.s_max = (lam, fixed, s_min, s_max)

    def _obj(self, prm, s_fixed=None):
        q = self.X.shape[1]
        m, w = (prm[0], prm[1:1 + q])
        ls = np.log(s_fixed) if s_fixed is not None else prm[1 + q]
        s = np.exp(ls)
        mu = m + self.X @ w
        A = self.Lc + lognorm_grid(mu, s) + np.log(DV)
        lse = logsumexp(A, axis=1)
        post = np.exp(A - lse[:, None])
        Ev = post @ GRID
        E2 = post @ GRID ** 2 - 2 * mu * Ev + mu ** 2
        gmu = (Ev - mu) / s ** 2
        ll = lse.sum()
        gls = float((E2 / s ** 2 - 1).sum())
        gm = gmu.sum()
        gw = self.X.T @ gmu
        if self.fixed is not None:
            v0, X0 = self.fixed
            mu0 = m + X0 @ w
            ll += float((-0.5 * ((v0 - mu0) / s) ** 2 - ls - 0.5 * LOG2PI).sum())
            g0 = (v0 - mu0) / s ** 2
            gm += g0.sum()
            gw = gw + X0.T @ g0
            gls += float((((v0 - mu0) / s) ** 2 - 1).sum())
        obj = -(ll - 0.5 * self.lam * float(w @ w))
        grad = -np.concatenate([[gm], gw - self.lam * w] + ([] if s_fixed is not None else [[gls]]))
        return (obj, grad)

    def fit(self):
        q = self.X.shape[1]
        peak = GRID[np.argmax(self.Lc, axis=1)] if self.Lc.shape[0] else np.array([0.0])
        x0 = np.concatenate([[float(np.median(peak))], np.zeros(q), [np.log(1.0)]])
        bounds = [(None, None)] * (1 + q) + [(np.log(self.s_min), np.log(self.s_max))]
        best = None
        for s0 in (1.0, 0.2):
            x0[-1] = np.log(s0)
            r = minimize(self._obj, x0, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-08})
            if best is None or r.fun < best.fun:
                best = r
        self.res = best
        self.m, self.w, self.s = (best.x[0], best.x[1:1 + q], float(np.exp(best.x[-1])))
        self.ll = -best.fun
        return self

    def profile_s(self, svals):
        q = self.X.shape[1]
        out = []
        x = np.concatenate([[self.m], self.w])
        for s in svals:
            r = minimize(lambda p: self._obj(p, s_fixed=s), x, jac=True, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-08})
            x = r.x
            out.append(-r.fun)
        return np.array(out)

    def mean_for(self, Xnew=None):
        if Xnew is None or self.X.shape[1] == 0:
            return np.full(1 if Xnew is None else len(Xnew), self.m)
        return self.m + np.asarray(Xnew, float) @ self.w

    def posterior(self, curves_new, mu_new=None):
        mu = np.full(curves_new.shape[0], self.m) if mu_new is None else mu_new
        A = curves_new + lognorm_grid(mu, self.s)
        return np.exp(A - logsumexp(A, axis=1)[:, None])

def prior_weights(mu, s):
    A = lognorm_grid(mu, s)
    return np.exp(A - logsumexp(A, axis=1)[:, None])

def cell_ll(k, n, o, W):
    out = np.zeros(len(k))
    for s in range(0, len(k), 800):
        sl = slice(s, s + 800)
        P = (expit(o[sl, None] + GRID[None, :]) * W[sl]).sum(1)
        P = np.clip(P, 1e-12, 1 - 1e-12)
        out[sl] = k[sl] * np.log(P) + (n[sl] - k[sl]) * np.log(1 - P)
    return out

def drop_constant(X):
    X = np.asarray(X, float)
    keep = X.std(0) > 0 if len(X) > 1 else np.zeros(X.shape[1], bool)
    return keep

def boot_fracs(contrib, rungs, floor, ceiling, repo, cond, seed=BSEED, B=None):
    df = pd.DataFrame({'repo': repo, 'cond': cond, **{r: contrib[r] for r in rungs}})
    g = df.groupby(['repo', 'cond'])[rungs].sum().reset_index()
    repos = sorted(g.repo.unique())
    conds = sorted(g.cond.unique())
    ri = pd.Categorical(g.repo, categories=repos).codes.astype(np.int64)
    ci = pd.Categorical(g.cond, categories=conds).codes.astype(np.int64)
    V = g[rungs].values
    B = NBOOT if B is None else B
    rng = np.random.default_rng(seed)
    draws = np.zeros((B, len(rungs)))
    for b in range(B):
        wr = np.bincount(rng.integers(0, len(repos), len(repos)), minlength=len(repos))
        wc = np.bincount(rng.integers(0, len(conds), len(conds)), minlength=len(conds))
        w = wr[ri] * wc[ci]
        draws[b] = w @ V
    point = V.sum(0)
    j = {r: i for i, r in enumerate(rungs)}
    res = {'n_rows': len(df), 'repos': len(repos), 'conds': len(conds), 'loglik': dict(zip(rungs, point.tolist()))}
    gap = point[j[ceiling]] - point[j[floor]]
    gap_b = draws[:, j[ceiling]] - draws[:, j[floor]]
    res['gap'] = float(gap)
    res['gap_ci'] = np.nanpercentile(gap_b, [2.5, 97.5]).tolist()
    for r in rungs:
        if r in (floor,):
            continue
        gain = point[j[r]] - point[j[floor]]
        gb = draws[:, j[r]] - draws[:, j[floor]]
        fr = gb / gap_b
        res[f'gain_{r}'] = float(gain)
        res[f'gain_{r}_ci'] = np.nanpercentile(gb, [2.5, 97.5]).tolist()
        res[f'frac_{r}'] = float(gain / gap) if gap != 0 else np.nan
        res[f'frac_{r}_ci'] = np.nanpercentile(fr, [2.5, 97.5]).tolist()
    return res

def repo_halves(repos, seed=SEED):
    rs = sorted(repos)
    perm = np.random.default_rng(seed).permutation(len(rs))
    order = [rs[i] for i in perm]
    nA = (len(rs) + 1) // 2
    return {r: 0 if i < nA else 1 for i, r in enumerate(order)}

def score_dist(cells, o, mu, s):
    W = prior_weights(np.asarray(mu, float), s)
    if W.shape[0] == 1:
        W = np.repeat(W, len(cells), axis=0)
    return cell_ll(cells.k.values, cells.n.values, o, W)

def part_C23(main, y, design):
    ds = main[main.model.eq('DeepSeek')]
    cells = make_cells(ds, y)
    conds = sorted((c for c in cells.condition.unique() if c != REF))
    feats_all = cue_features(conds)
    half = repo_halves(cells.repo.unique())
    cells['half'] = cells.repo.map(half)
    fams = sorted(set((c[:3] for c in conds)))
    if design == 'C2':
        outer = [[c] for c in conds]
        fsets = ['roles']
    else:
        outer = [[c for c in conds if c[:3] == f] for f in fams]
        fsets = ['roles']
    rows, folds = ([], [])
    for held in outer:
        tr = cells[~cells.condition.isin(held)].reset_index(drop=True)
        te = cells[cells.condition.isin(held)].reset_index(drop=True)
        fe = FE(tr, provider=True).fit()
        trc = tr[tr.condition.ne(REF)].reset_index(drop=True)
        cue_tr = fe.conds
        Lc = level_curves(trc, fe.offsets(trc, 'cue'), 'condition', cue_tr)
        floor = Marg(Lc).fit()
        o = fe.offsets(te, 'cue')
        out = {'floor': score_dist(te, o, [floor.m], floor.s)}
        if design == 'C2':
            perm = np.random.default_rng(SEED).permutation(len(cue_tr))
            fold_of = np.zeros(len(cue_tr), int)
            for pos, i in enumerate(perm):
                fold_of[i] = pos % 5
            inner = [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(5)]
        else:
            ftr = np.array([c[:3] for c in cue_tr])
            inner = [(np.flatnonzero(ftr != f), np.flatnonzero(ftr == f)) for f in sorted(set(ftr))]
        info = {'held': held}
        for fs in fsets:
            Xall = feats_all[fs]
            Xtr_full = Xall.reindex(cue_tr).values
            keep = drop_constant(Xtr_full)
            Xtr = Xtr_full[:, keep]
            Xte_c = Xall.reindex(held).values[:, keep]
            scores = {}
            for lam in (0.3, 3.0, 30.0, 300.0):
                tot = 0.0
                for itr, ite in inner:
                    kin = drop_constant(Xtr[itr])
                    m = Marg(Lc[itr], Xtr[itr][:, kin], lam=lam).fit()
                    te_c = [cue_tr[i] for i in ite]
                    cc = trc[trc.condition.isin(te_c)]
                    mu = pd.Series(m.mean_for(Xtr[ite][:, kin]), index=te_c)
                    W = prior_weights(cc.condition.map(mu).values, m.s)
                    tot += cell_ll(cc.k.values, cc.n.values, fe.offsets(cc, 'cue'), W).sum()
                scores[lam] = tot
            lam = max(scores, key=lambda l: scores[l])
            mfit = Marg(Lc, Xtr, lam=lam).fit()
            mu = pd.Series(mfit.mean_for(Xte_c), index=held)
            ll_fs = cell_ll(te.k.values, te.n.values, o, prior_weights(te.condition.map(mu).values, mfit.s))
            if fs == 'family':
                nosib = [c for c in held if not any((t[:3] == c[:3] for t in cue_tr))]
                selm = te.condition.isin(nosib).values
                ll_fs[selm] = out['floor'][selm]
                info['family_nosibling'] = nosib
            out[fs] = ll_fs
            info[fs] = {'lam': lam, 's': mfit.s, 'pred_mean': mu.to_dict()}
        ll_ceil = np.zeros(len(te))
        for h in (0, 1):
            other = te[te.half.ne(h)]
            Lh = level_curves(other, fe.offsets(other, 'cue'), 'condition', held) if len(other) else np.zeros((len(held), len(GRID)))
            post = floor.posterior(Lh)
            pidx = pd.Categorical(te.condition, categories=held).codes.astype(np.int64)
            sel = np.flatnonzero(te.half.eq(h).values)
            ll_ceil[sel] = cell_ll(te.k.values[sel], te.n.values[sel], o[sel], post[pidx[sel]])
        out['ceiling'] = ll_ceil
        info.update({'floor_m': floor.m, 'floor_s': floor.s, 'fe_conv': fe.converged})
        folds.append(info)
        rows.append(pd.DataFrame({'repo': te.repo.values, 'cond': te.condition.values, 'n': te.n.values, **out}))
    D = pd.concat(rows, ignore_index=True)
    rungs = ['floor'] + fsets + ['ceiling']
    res = boot_fracs({r: D[r].values for r in rungs}, rungs, 'floor', 'ceiling', D.repo.values, D.cond.values)
    D['fam'] = D.cond.str[:3]
    per = D.groupby('fam')[rungs].sum()
    res['per_family_gain_vs_floor'] = {f: {r: float(per.loc[f, r] - per.loc[f, 'floor']) for r in rungs if r != 'floor'} for f in per.index}
    res.update({'design': design, 'outcome': y, 'runs': int(D.n.sum()), 'folds': folds, 'all_fe_converged': all((f['fe_conv'] for f in folds))})
    emit(design, res)

def part_C4(main, y):
    agents = ['DeepSeek', 'GLM', 'Qwen', 'Nemotron', 'gpt-oss-120b'] if y == 'G_tests' else ['DeepSeek', 'GLM', 'Qwen']
    cells = {}
    for a in agents:
        c = make_cells(main[main.model.eq(a)], y)
        cells[a] = c
    half = repo_halves(main.repo.unique())
    eb = {}
    for a in agents:
        c = cells[a]
        fe = FE(c, provider=a == 'DeepSeek').fit()
        cc = c[c.condition.ne(REF)].reset_index(drop=True)
        Lc = level_curves(cc, fe.offsets(cc, 'cue'), 'condition', fe.conds)
        m = Marg(Lc).fit()
        post = m.posterior(Lc)
        pm = post @ GRID
        pv = post @ GRID ** 2 - pm ** 2
        eb[a] = {'m': m.m, 's': m.s, 'mean': pd.Series(pm, index=fe.conds), 'var': pd.Series(pv, index=fe.conds)}
    results = []
    for a in agents:
        others = [x for x in agents if x != a]
        c = cells[a].copy()
        c['half'] = c.repo.map(half)
        nc = c[c.condition.eq(REF)].reset_index(drop=True)
        fe0 = FE(nc, provider=a == 'DeepSeek').fit()
        Lt = level_curves(nc, fe0.offsets(nc, 'task'), 'task_id', fe0.tasks)
        mt = Marg(Lt).fit()
        apost = pd.Series(mt.posterior(Lt) @ GRID, index=fe0.tasks)
        ev = c[c.condition.ne(REF)].reset_index(drop=True)
        o = ev.task_id.map(apost).fillna(mt.m).values + fe0.delta * ev.provider.eq('deepseek').values
        m_bar = float(np.mean([eb[x]['m'] for x in others]))
        s_bar = float(np.sqrt(np.mean([eb[x]['s'] ** 2 for x in others])))
        out = {'floor': score_dist(ev, o, [m_bar], s_bar)}
        M = pd.DataFrame({x: eb[x]['mean'] for x in others})
        V = pd.DataFrame({x: eb[x]['var'] for x in others})
        tm = M.mean(axis=1)
        res2, pvs = ([], [])
        for x in others:
            rest = [z for z in others if z != x]
            sub = M[rest].mean(axis=1)
            ok = M[x].notna() & sub.notna()
            res2.extend((M[x] - sub)[ok] ** 2)
            pvs.extend(V[x][ok])
        sT = float(np.sqrt(max(0.01, np.mean(res2) - np.mean(pvs))))
        mu_c = ev.condition.map(tm)
        miss = int(mu_c.isna().sum())
        out['transfer'] = cell_ll(ev.k.values, ev.n.values, o, prior_weights(mu_c.fillna(m_bar).values, sT))
        ll_ceil = np.zeros(len(ev))
        conds_a = sorted(ev.condition.unique())
        pidx = pd.Categorical(ev.condition, categories=conds_a).codes.astype(np.int64)
        for h in (0, 1):
            other = ev[ev.half.ne(h)]
            oo = o[ev.half.ne(h).values]
            Lh = level_curves(other, oo, 'condition', conds_a)
            has = np.isin(conds_a, other.condition.unique())
            mh = Marg(Lh[has]).fit()
            post = mh.posterior(Lh)
            sel = np.flatnonzero(ev.half.eq(h).values)
            ll_ceil[sel] = cell_ll(ev.k.values[sel], ev.n.values[sel], o[sel], post[pidx[sel]])
        out['ceiling'] = ll_ceil
        rungs = ['floor', 'transfer', 'ceiling']
        r = boot_fracs(out, rungs, 'floor', 'ceiling', ev.repo.values, ev.condition.values)
        r.update({'design': 'C4_new_agent_cue_part', 'outcome': y, 'target': a, 'others': others, 'runs': int(ev.n.sum()), 'transfer_sd': sT, 'floor_mean': m_bar, 'floor_sd': s_bar, 'own_cue_sd': eb[a]['s'], 'conditions_missing_in_others': miss, 'corr_transfer_vs_own_eb': float(pd.concat([tm, eb[a]['mean']], axis=1).dropna().corr().iloc[0, 1])})
        emit('C4', r)
        results.append((a, out, ev))
    allout = {r: np.concatenate([o_[r] for _, o_, _ in results]) for r in ('floor', 'transfer', 'ceiling')}
    reps = np.concatenate([e.repo.values for _, _, e in results])
    cnd = np.concatenate([(a + '|' + e.condition).values for a, _, e in results])
    rs = boot_fracs(allout, ['floor', 'transfer', 'ceiling'], 'floor', 'ceiling', reps, cnd)
    rs.update({'design': 'C4_new_agent_cue_part', 'outcome': y, 'target': 'all', 'runs': int(sum((e.n.sum() for _, _, e in results)))})
    emit('C4', rs)

def agent_frame(M, agent):
    d = M[M.model.eq(agent)].copy()
    if agent != 'DeepSeek':
        d['provider'] = 'one'
    return d

def part_C23g(main, y, design, agent):
    ds = agent_frame(main, agent)
    dropped = 0
    cells = make_cells(ds, y)
    conds = sorted((c for c in cells.condition.unique() if c != REF))
    feats_all = cue_features(conds)
    half = repo_halves(cells.repo.unique())
    cells['half'] = cells.repo.map(half)
    fams = sorted(set((c[:3] for c in conds)))
    if design == 'C2':
        outer = [[c] for c in conds]
        fsets = ['roles']
    else:
        outer = [[c for c in conds if c[:3] == f] for f in fams]
        fsets = ['roles']
    rows, folds = ([], [])
    for held in outer:
        tr = cells[~cells.condition.isin(held)].reset_index(drop=True)
        te = cells[cells.condition.isin(held)].reset_index(drop=True)
        fe = FE(tr, provider=agent == 'DeepSeek').fit()
        okt = te.task_id.isin(fe.a.index).values
        dropped += int(te.n.values[~okt].sum())
        te = te[okt].reset_index(drop=True)
        if len(te) == 0:
            continue
        trc = tr[tr.condition.ne(REF)].reset_index(drop=True)
        cue_tr = fe.conds
        Lc = level_curves(trc, fe.offsets(trc, 'cue'), 'condition', cue_tr)
        floor = Marg(Lc).fit()
        o = fe.offsets(te, 'cue')
        out = {'floor': score_dist(te, o, [floor.m], floor.s)}
        if design == 'C2':
            perm = np.random.default_rng(SEED).permutation(len(cue_tr))
            fold_of = np.zeros(len(cue_tr), int)
            for pos, i in enumerate(perm):
                fold_of[i] = pos % 5
            inner = [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(5)]
        else:
            ftr = np.array([c[:3] for c in cue_tr])
            inner = [(np.flatnonzero(ftr != f), np.flatnonzero(ftr == f)) for f in sorted(set(ftr))]
        info = {'held': held}
        for fs in fsets:
            Xall = feats_all[fs]
            Xtr_full = Xall.reindex(cue_tr).values
            keep = drop_constant(Xtr_full)
            Xtr = Xtr_full[:, keep]
            Xte_c = Xall.reindex(held).values[:, keep]
            scores = {}
            for lam in (0.3, 3.0, 30.0, 300.0):
                tot = 0.0
                for itr, ite in inner:
                    kin = drop_constant(Xtr[itr])
                    m = Marg(Lc[itr], Xtr[itr][:, kin], lam=lam).fit()
                    te_c = [cue_tr[i] for i in ite]
                    cc = trc[trc.condition.isin(te_c)]
                    mu = pd.Series(m.mean_for(Xtr[ite][:, kin]), index=te_c)
                    W = prior_weights(cc.condition.map(mu).values, m.s)
                    tot += cell_ll(cc.k.values, cc.n.values, fe.offsets(cc, 'cue'), W).sum()
                scores[lam] = tot
            lam = max(scores, key=lambda l: scores[l])
            mfit = Marg(Lc, Xtr, lam=lam).fit()
            mu = pd.Series(mfit.mean_for(Xte_c), index=held)
            ll_fs = cell_ll(te.k.values, te.n.values, o, prior_weights(te.condition.map(mu).values, mfit.s))
            if fs == 'family':
                nosib = [c for c in held if not any((t[:3] == c[:3] for t in cue_tr))]
                selm = te.condition.isin(nosib).values
                ll_fs[selm] = out['floor'][selm]
                info['family_nosibling'] = nosib
            out[fs] = ll_fs
            info[fs] = {'lam': lam, 's': mfit.s, 'pred_mean': mu.to_dict()}
        ll_ceil = np.zeros(len(te))
        for h in (0, 1):
            other = te[te.half.ne(h)]
            Lh = level_curves(other, fe.offsets(other, 'cue'), 'condition', held) if len(other) else np.zeros((len(held), len(GRID)))
            post = floor.posterior(Lh)
            pidx = pd.Categorical(te.condition, categories=held).codes.astype(np.int64)
            sel = np.flatnonzero(te.half.eq(h).values)
            ll_ceil[sel] = cell_ll(te.k.values[sel], te.n.values[sel], o[sel], post[pidx[sel]])
        out['ceiling'] = ll_ceil
        info.update({'floor_m': floor.m, 'floor_s': floor.s, 'fe_conv': fe.converged})
        folds.append(info)
        rows.append(pd.DataFrame({'repo': te.repo.values, 'cond': te.condition.values, 'n': te.n.values, **out}))
    D = pd.concat(rows, ignore_index=True)
    rungs = ['floor'] + fsets + ['ceiling']
    res = boot_fracs({r: D[r].values for r in rungs}, rungs, 'floor', 'ceiling', D.repo.values, D.cond.values)
    D['fam'] = D.cond.str[:3]
    per = D.groupby('fam')[rungs].sum()
    res['per_family_gain_vs_floor'] = {f: {r: float(per.loc[f, r] - per.loc[f, 'floor']) for r in rungs if r != 'floor'} for f in per.index}
    res.update({'design': design, 'outcome': y, 'runs': int(D.n.sum()), 'folds': folds, 'all_fe_converged': all((f['fe_conv'] for f in folds)), 'runs_dropped_no_offset': dropped})
    emit(design, res)

def part_C4g(main, y, agents):
    cells = {}
    for a in agents:
        c = make_cells(main[main.model.eq(a)], y)
        cells[a] = c
    half = repo_halves(main.repo.unique())
    eb = {}
    for a in agents:
        c = cells[a]
        fe = FE(c, provider=a == 'DeepSeek').fit()
        cc = c[c.condition.ne(REF)].reset_index(drop=True)
        Lc = level_curves(cc, fe.offsets(cc, 'cue'), 'condition', fe.conds)
        m = Marg(Lc).fit()
        post = m.posterior(Lc)
        pm = post @ GRID
        pv = post @ GRID ** 2 - pm ** 2
        eb[a] = {'m': m.m, 's': m.s, 'mean': pd.Series(pm, index=fe.conds), 'var': pd.Series(pv, index=fe.conds)}
    results = []
    for a in agents:
        others = [x for x in agents if x != a]
        c = cells[a].copy()
        c['half'] = c.repo.map(half)
        nc = c[c.condition.eq(REF)].reset_index(drop=True)
        fe0 = FE(nc, provider=a == 'DeepSeek').fit()
        Lt = level_curves(nc, fe0.offsets(nc, 'task'), 'task_id', fe0.tasks)
        mt = Marg(Lt).fit()
        apost = pd.Series(mt.posterior(Lt) @ GRID, index=fe0.tasks)
        ev = c[c.condition.ne(REF)].reset_index(drop=True)
        o = ev.task_id.map(apost).fillna(mt.m).values + fe0.delta * ev.provider.eq('deepseek').values
        m_bar = float(np.mean([eb[x]['m'] for x in others]))
        s_bar = float(np.sqrt(np.mean([eb[x]['s'] ** 2 for x in others])))
        out = {'floor': score_dist(ev, o, [m_bar], s_bar)}
        M = pd.DataFrame({x: eb[x]['mean'] for x in others})
        V = pd.DataFrame({x: eb[x]['var'] for x in others})
        tm = M.mean(axis=1)
        res2, pvs = ([], [])
        for x in others:
            rest = [z for z in others if z != x]
            sub = M[rest].mean(axis=1)
            ok = M[x].notna() & sub.notna()
            res2.extend((M[x] - sub)[ok] ** 2)
            pvs.extend(V[x][ok])
        sT = float(np.sqrt(max(0.01, np.mean(res2) - np.mean(pvs))))
        mu_c = ev.condition.map(tm)
        miss = int(mu_c.isna().sum())
        out['transfer'] = cell_ll(ev.k.values, ev.n.values, o, prior_weights(mu_c.fillna(m_bar).values, sT))
        ll_ceil = np.zeros(len(ev))
        conds_a = sorted(ev.condition.unique())
        pidx = pd.Categorical(ev.condition, categories=conds_a).codes.astype(np.int64)
        for h in (0, 1):
            other = ev[ev.half.ne(h)]
            oo = o[ev.half.ne(h).values]
            Lh = level_curves(other, oo, 'condition', conds_a)
            has = np.isin(conds_a, other.condition.unique())
            mh = Marg(Lh[has]).fit()
            post = mh.posterior(Lh)
            sel = np.flatnonzero(ev.half.eq(h).values)
            ll_ceil[sel] = cell_ll(ev.k.values[sel], ev.n.values[sel], o[sel], post[pidx[sel]])
        out['ceiling'] = ll_ceil
        rungs = ['floor', 'transfer', 'ceiling']
        r = boot_fracs(out, rungs, 'floor', 'ceiling', ev.repo.values, ev.condition.values)
        r.update({'design': 'C4_new_agent_cue_part', 'outcome': y, 'target': a, 'others': others, 'runs': int(ev.n.sum()), 'transfer_sd': sT, 'floor_mean': m_bar, 'floor_sd': s_bar, 'own_cue_sd': eb[a]['s'], 'conditions_missing_in_others': miss, 'corr_transfer_vs_own_eb': float(pd.concat([tm, eb[a]['mean']], axis=1).dropna().corr().iloc[0, 1])})
        emit('C4', r)
        results.append((a, out, ev))
    allout = {r: np.concatenate([o_[r] for _, o_, _ in results]) for r in ('floor', 'transfer', 'ceiling')}
    reps = np.concatenate([e.repo.values for _, _, e in results])
    cnd = np.concatenate([(a + '|' + e.condition).values for a, _, e in results])
    rs = boot_fracs(allout, ['floor', 'transfer', 'ceiling'], 'floor', 'ceiling', reps, cnd)
    rs.update({'design': 'C4_new_agent_cue_part', 'outcome': y, 'target': 'all', 'runs': int(sum((e.n.sum() for _, _, e in results)))})
    emit('C4', rs)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('design', choices=['C2', 'C3', 'C4'])
    p.add_argument('--outcome', choices=['judged', 'notests'], default='judged')
    p.add_argument('--labels', choices=['original', 'corrected'], default='original')
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, default=OUTPUT / 'validation')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(a.data / 'factor_runs.parquet')
    f = f[f.dataset.eq('paper_main')].copy()
    f['G_tests'] = f.notests
    f['R_masked'] = f.judged_old if a.labels == 'original' else f.judged
    if a.outcome == 'notests':
        if a.design == 'C4':
            part_C4(f, 'G_tests')
        else:
            part_C23(f, 'G_tests', a.design)
    elif a.design == 'C4':
        rt = L.room_table(f, 'R_masked')
        agents = [m for m in ['DeepSeek', 'Qwen', 'GLM', 'gpt-oss-120b', 'Nemotron'] if m in set(rt[rt.enters].model)]
        part_C4g(f, 'R_masked', agents)
    else:
        part_C23g(f, 'R_masked', a.design, 'DeepSeek')
    (a.output / (a.design + '_' + a.outcome + '.json')).write_text(json.dumps(RESULTS, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x) + '\n')
if __name__ == '__main__':
    main()

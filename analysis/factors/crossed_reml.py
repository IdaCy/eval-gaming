"""Crossed Gaussian variance components with known cell sampling variance."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import brentq, minimize
from scipy.stats import chi2

@dataclass
class REMLResult:
    variance: np.ndarray
    nll: float
    converged: bool
    gradient: np.ndarray
    projected_gradient: float
    iterations: int
    message: str

class CrossedREML:

    def __init__(self, cells, z, noise_variance, *, noise_scale=False, provider_fixed=False, agent_task=False):
        self.cells = cells.reset_index(drop=True)
        self.z = np.asarray(z, float)
        self.v0 = np.asarray(noise_variance, float)
        assert np.isfinite(self.z).all() and (self.v0 > 0).all()
        self.N = len(cells)
        self.noise_scale = noise_scale
        levels = {}
        if cells.model.nunique() > 1:
            levels['agent'] = cells.model.astype(str).to_numpy()
        levels['task'] = cells.task_id.astype(str).to_numpy()
        levels['condition'] = cells.condition.astype(str).to_numpy()
        if cells.model.nunique() > 1:
            levels['agent_condition'] = (cells.model.astype(str) + '|' + cells.condition.astype(str)).to_numpy()
        if agent_task and cells.model.nunique() > 1:
            levels['agent_task'] = (cells.model.astype(str) + '|' + cells.task_id.astype(str)).to_numpy()
        self.low_names = list(levels)
        blocks, spans, codes = ([], [], {})
        q = 0
        for name, values in levels.items():
            labels, ix = np.unique(values, return_inverse=True)
            blocks.append(sparse.csr_matrix((np.ones(self.N), (np.arange(self.N), ix)), shape=(self.N, len(labels))))
            spans.append(np.arange(q, q + len(labels)))
            q += len(labels)
            codes[name] = ix
        self.q = q
        self.spans = spans
        self.Z = sparse.hstack(blocks, format='csr')
        tc = cells.task_id.astype(str) + '|' + cells.condition.astype(str)
        tc_labels, self.tc_index = np.unique(tc, return_inverse=True)
        self.J = len(tc_labels)
        self.singleton_tc = self.J == self.N
        self.Ztc = sparse.csr_matrix((np.ones(self.N), (np.arange(self.N), self.tc_index)), shape=(self.N, self.J))
        fixed = [np.ones(self.N)]
        self.fixed_names = ['intercept']
        if provider_fixed:
            p = (cells.model.eq('DeepSeek') & cells.provider.eq('deepseek')).astype(float).to_numpy()
            if p.std() > 0:
                fixed.append(p)
                self.fixed_names.append('DeepSeek_direct_provider')
        self.X = np.column_stack(fixed)
        self.p = self.X.shape[1]
        assert np.linalg.matrix_rank(self.X) == self.p
        self.names = self.low_names + ['task_condition']
        self.parameter_names = self.names + (['noise_scale'] if noise_scale else [])
        self.level_counts = {name: len(ix) for name, ix in zip(self.low_names, spans)} | {'task_condition': self.J}
        self._M = sparse.hstack([self.Z, sparse.csr_matrix(self.X), sparse.csr_matrix(self.z[:, None])], format='csr')
        wm = self._M.multiply((1 / self.v0)[:, None])
        self.U0 = (self._M.T @ wm).toarray()
        self.C0 = (self.Ztc.T @ wm).tocsr()
        self.h0 = np.bincount(self.tc_index, weights=1 / self.v0, minlength=self.J)
        self.logdetD0 = float(np.log(self.v0).sum())

    def objective(self, theta, gradient=True):
        theta = np.asarray(theta, float)
        if (theta < 0).any() or (self.noise_scale and theta[-1] <= 0 and (not self.singleton_tc)):
            return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
        phi = theta[-1] if self.noise_scale else 1.0
        tc_var = theta[len(self.low_names)]
        g = np.concatenate([np.repeat(theta[j], len(ix)) for j, ix in enumerate(self.spans)])
        rootg = np.sqrt(g)
        if self.singleton_tc:
            diagonal = phi * self.v0 + tc_var
            if (diagonal <= 0).any():
                return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
            cross = (self._M.T @ self._M.multiply((1 / diagonal)[:, None])).toarray()
            logdetR = np.log(diagonal).sum()
        else:
            C = self.C0 / phi
            h = self.h0 / phi
            den = 1 + tc_var * h
            cross = self.U0 / phi - (C.T @ C.multiply((tc_var / den)[:, None])).toarray()
            logdetR = self.logdetD0 + self.N * np.log(phi) + np.log1p(tc_var * h).sum()
        S = cross[:self.q, :self.q]
        SX = cross[:self.q, self.q:]
        H = np.eye(self.q) + rootg[:, None] * S * rootg[None, :]
        try:
            ch = cho_factor(H, lower=True, check_finite=False)
            rhs = rootg[:, None] * SX
            shrunk = cho_solve(ch, rhs, check_finite=False)
            vcross = cross[self.q:, self.q:] - rhs.T @ shrunk
            xx, xy = (vcross[:self.p, :self.p], vcross[:self.p, -1])
            chx = cho_factor(xx, lower=True, check_finite=False)
            beta = cho_solve(chx, xy, check_finite=False)
        except np.linalg.LinAlgError:
            return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
        quad = float(vcross[-1, -1] - xy @ beta)
        logdetV = logdetR + 2 * np.log(np.diag(ch[0])).sum()
        logdetXX = 2 * np.log(np.diag(chx[0])).sum()
        nll = float(0.5 * (logdetV + logdetXX + quad + (self.N - self.p) * np.log(2 * np.pi)))
        if not gradient:
            return nll
        hinv = cho_solve(ch, np.eye(self.q), check_finite=False)
        gbar = rootg[:, None] * hinv * rootg[None, :]
        invxx = cho_solve(chx, np.eye(self.p), check_finite=False)
        zvcross = SX - S @ (gbar @ SX)
        zvx, zvy = (zvcross[:, :self.p], zvcross[:, -1])
        zpy = zvy - zvx @ beta
        diag_zvz = np.diag(S) - np.sum(S @ gbar * S, axis=1)
        diag_zpz = diag_zvz - np.sum(zvx @ invxx * zvx, axis=1)
        grad = [0.5 * (diag_zpz[ix].sum() - np.square(zpy[ix]).sum()) for ix in self.spans]
        if self.singleton_tc:
            residual_xy = (np.column_stack([self.X, self.z]) - self.Z @ (gbar @ SX)) / diagonal[:, None]
            vx, vy = (residual_xy[:, :self.p], residual_xy[:, -1])
            py = vy - vx @ beta
            for weight in [np.ones(self.N), self.v0] if self.noise_scale else [np.ones(self.N)]:
                ztrz = (self.Z.T @ self.Z.multiply((weight / diagonal ** 2)[:, None])).toarray()
                trace = np.sum(weight / diagonal) - np.sum(gbar * ztrz.T) - np.sum(vx @ invxx * vx * weight[:, None])
                grad.append(0.5 * (trace - np.sum(weight * py ** 2)))
        else:
            cz = C[:, :self.q]
            cxy = C[:, self.q:].toarray()
            tc_vxy = (cxy - cz @ (gbar @ SX)) / den[:, None]
            tc_vx, tc_vy = (tc_vxy[:, :self.p], tc_vxy[:, -1])
            tc_py = tc_vy - tc_vx @ beta
            ztrz = (cz.T @ cz.multiply((1 / den ** 2)[:, None])).toarray()
            trace_tc = np.sum(h / den) - np.sum(gbar * ztrz.T) - np.sum(tc_vx @ invxx * tc_vx)
            grad.append(0.5 * (trace_tc - np.square(tc_py).sum()))
            if self.noise_scale:
                grad.append((0.5 * (self.N - self.p - quad) - np.dot(theta[:len(self.names)], grad)) / phi)
        return (nll, np.asarray(grad, float))

    def fit(self, start=None, fixed=None, multistart=True):
        fixed = {} if fixed is None else dict(fixed)
        initial = np.full(len(self.parameter_names), 0.25) if start is None else np.array(start, float)
        if start is None:
            if 'agent' in self.names:
                initial[self.names.index('agent')] = 1.0
            if self.noise_scale:
                initial[-1] = 1.0
        free = np.array([j for j in range(len(initial)) if j not in fixed], int)
        for j, value in fixed.items():
            initial[j] = value
        bounds = [(1e-07, None) if self.parameter_names[j] == 'noise_scale' and (not self.singleton_tc) else (0.0, None) for j in free]

        def fun(v):
            theta = initial.copy()
            theta[free] = v
            val, grad = self.objective(theta)
            return (val, grad[free])
        starts = [initial[free]]
        if multistart and (not fixed):
            starts.append(np.maximum(initial[free] * 0.2, 0.01))
        results = [minimize(fun, st, jac=True, method='L-BFGS-B', bounds=bounds, options={'ftol': 2e-12, 'gtol': 1e-06, 'maxiter': 2000, 'maxls': 40}) for st in starts]
        best = min(results, key=lambda r: r.fun)
        theta = initial.copy()
        theta[free] = best.x
        val, grad = self.objective(theta)
        pg = grad[free].copy()
        pg[(theta[free] <= 1e-08) & (pg > 0)] = 0
        return REMLResult(theta, val, bool(best.success), grad, float(np.max(np.abs(pg))), int(best.nit), str(best.message))

    def profile_variance(self, result, component):
        j = self.names.index(component)
        estimate = result.variance[j]
        target = result.nll + 0.5 * chi2.ppf(0.95, 1)
        cache = {}

        def f(value):
            if value not in cache:
                fit = self.fit(result.variance, {j: value}, multistart=False)
                if not fit.converged:
                    raise RuntimeError(f'Profile fit did not converge: {component}, {value}, {fit.message}')
                cache[value] = fit.nll - target
            return cache[value]
        low = 0.0 if estimate < 1e-08 or f(0.0) <= 0 else brentq(f, 0, estimate, xtol=1e-06)
        upper = max(0.1, 2 * estimate)
        while f(upper) <= 0:
            upper *= 2
        high = brentq(f, estimate, upper, xtol=1e-06)
        return (low, high, len(cache))

    def profile_share(self, result, component):
        assert not self.noise_scale
        j = self.names.index(component)
        other = np.array([i for i in range(len(self.names)) if i != j])
        total = result.variance.sum()
        estimate = result.variance[j] / total
        target = result.nll + 0.5 * chi2.ppf(0.95, 1)
        cache = {}

        def f(share):
            if share in cache:
                return cache[share]
            if share == 1:
                fit = self.fit(result.variance, {i: 0.0 for i in other}, multistart=False)
                value = fit.nll
                success = fit.converged
            else:
                ratio = share / (1 - share)
                start = result.variance[other] * ((1 - share) / max(1 - estimate, 1e-12))
                start = np.maximum(start, 1e-06)

                def fn(values):
                    theta = np.zeros(len(self.names))
                    theta[other] = values
                    theta[j] = ratio * values.sum()
                    value, gradient = self.objective(theta)
                    return (value, gradient[other] + ratio * gradient[j])
                fit = minimize(fn, start, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(other), options={'ftol': 2e-12, 'gtol': 1e-06, 'maxiter': 2000, 'maxls': 40})
                value, success = (fit.fun, fit.success)
            if not success:
                raise RuntimeError(f'Share profile did not converge: {component}, {share}')
            cache[share] = value - target
            return cache[share]
        low = 0.0 if estimate < 1e-08 or f(0.0) <= 0 else brentq(f, 0, estimate, xtol=1e-06)
        high = 1.0 if 1 - estimate < 1e-08 or f(1.0) <= 0 else brentq(f, estimate, 1, xtol=1e-06)
        return (low, high, len(cache))

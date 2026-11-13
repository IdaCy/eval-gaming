"""Fit the pooled crossed variance components of judged gaming."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[1]
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
            levels["agent"] = cells.model.astype(str).to_numpy()
        levels["task"] = cells.task_id.astype(str).to_numpy()
        levels["condition"] = cells.condition.astype(str).to_numpy()
        if cells.model.nunique() > 1:
            levels["agent_condition"] = (cells.model.astype(str)+"|"+cells.condition.astype(str)).to_numpy()
        if agent_task and cells.model.nunique() > 1:
            levels["agent_task"] = (cells.model.astype(str)+"|"+cells.task_id.astype(str)).to_numpy()
        self.low_names = list(levels)
        blocks, spans, codes = [], [], {}
        q = 0
        for name, values in levels.items():
            labels, ix = np.unique(values, return_inverse=True)
            blocks.append(sparse.csr_matrix((np.ones(self.N), (np.arange(self.N), ix)), shape=(self.N, len(labels))))
            spans.append(np.arange(q, q+len(labels)))
            q += len(labels)
            codes[name] = ix
        self.q = q
        self.spans = spans
        self.Z = sparse.hstack(blocks, format="csr")
        tc = cells.task_id.astype(str)+"|"+cells.condition.astype(str)
        tc_labels, self.tc_index = np.unique(tc, return_inverse=True)
        self.J = len(tc_labels)
        self.singleton_tc = self.J == self.N
        self.Ztc = sparse.csr_matrix((np.ones(self.N), (np.arange(self.N), self.tc_index)), shape=(self.N, self.J))
        fixed = [np.ones(self.N)]
        self.fixed_names = ["intercept"]
        if provider_fixed:
            p = (cells.model.eq("DeepSeek") & cells.provider.eq("deepseek")).astype(float).to_numpy()
            if p.std() > 0:
                fixed.append(p)
                self.fixed_names.append("DeepSeek_direct_provider")
        self.X = np.column_stack(fixed)
        self.p = self.X.shape[1]
        assert np.linalg.matrix_rank(self.X) == self.p
        self.names = self.low_names + ["task_condition"]
        self.parameter_names = self.names + (["noise_scale"] if noise_scale else [])
        self.level_counts = {name: len(ix) for name, ix in zip(self.low_names, spans)} | {"task_condition": self.J}
        self._M = sparse.hstack([self.Z, sparse.csr_matrix(self.X), sparse.csr_matrix(self.z[:, None])], format="csr")
        wm = self._M.multiply((1/self.v0)[:, None])
        self.U0 = (self._M.T @ wm).toarray()
        self.C0 = (self.Ztc.T @ wm).tocsr()
        self.h0 = np.bincount(self.tc_index, weights=1/self.v0, minlength=self.J)
        self.logdetD0 = float(np.log(self.v0).sum())

    def objective(self, theta, gradient=True):
        """Negative restricted log likelihood and its exact derivatives w.r.t. variances."""
        theta = np.asarray(theta, float)
        if (theta < 0).any() or (self.noise_scale and theta[-1] <= 0 and not self.singleton_tc):
            return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
        phi = theta[-1] if self.noise_scale else 1.0
        tc_var = theta[len(self.low_names)]
        g = np.concatenate([np.repeat(theta[j], len(ix)) for j, ix in enumerate(self.spans)])
        rootg = np.sqrt(g)
        if self.singleton_tc:
            # With one agent every task-condition is a separate observation. Work directly
            # with its diagonal residual variance, including the legitimate phi=0 boundary.
            # This avoids subtracting two large precision matrices as phi approaches zero.
            diagonal = phi*self.v0+tc_var
            if (diagonal <= 0).any():
                return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
            cross = (self._M.T @ self._M.multiply((1/diagonal)[:, None])).toarray()
            logdetR = np.log(diagonal).sum()
        else:
            C = self.C0/phi
            h = self.h0/phi
            den = 1+tc_var*h
            cross = self.U0/phi - (C.T @ C.multiply((tc_var/den)[:, None])).toarray()
            logdetR = self.logdetD0+self.N*np.log(phi)+np.log1p(tc_var*h).sum()
        S = cross[:self.q, :self.q]
        SX = cross[:self.q, self.q:]
        H = np.eye(self.q) + rootg[:, None]*S*rootg[None, :]
        try:
            ch = cho_factor(H, lower=True, check_finite=False)
            rhs = rootg[:, None]*SX
            shrunk = cho_solve(ch, rhs, check_finite=False)
            vcross = cross[self.q:, self.q:] - rhs.T @ shrunk
            xx, xy = vcross[:self.p, :self.p], vcross[:self.p, -1]
            chx = cho_factor(xx, lower=True, check_finite=False)
            beta = cho_solve(chx, xy, check_finite=False)
        except np.linalg.LinAlgError:
            return (np.inf, np.full_like(theta, np.nan)) if gradient else np.inf
        quad = float(vcross[-1, -1]-xy @ beta)
        logdetV = logdetR+2*np.log(np.diag(ch[0])).sum()
        logdetXX = 2*np.log(np.diag(chx[0])).sum()
        nll = float(0.5*(logdetV+logdetXX+quad+(self.N-self.p)*np.log(2*np.pi)))
        if not gradient:
            return nll

        hinv = cho_solve(ch, np.eye(self.q), check_finite=False)
        gbar = rootg[:, None]*hinv*rootg[None, :]
        invxx = cho_solve(chx, np.eye(self.p), check_finite=False)
        zvcross = SX-S @ (gbar @ SX)
        zvx, zvy = zvcross[:, :self.p], zvcross[:, -1]
        zpy = zvy-zvx @ beta
        diag_zvz = np.diag(S)-np.sum((S @ gbar)*S, axis=1)
        diag_zpz = diag_zvz-np.sum((zvx @ invxx)*zvx, axis=1)
        grad = [0.5*(diag_zpz[ix].sum()-np.square(zpy[ix]).sum()) for ix in self.spans]

        if self.singleton_tc:
            residual_xy = (np.column_stack([self.X, self.z])-self.Z @ (gbar @ SX))/diagonal[:, None]
            vx, vy = residual_xy[:, :self.p], residual_xy[:, -1]
            py = vy-vx @ beta
            for weight in ([np.ones(self.N), self.v0] if self.noise_scale else [np.ones(self.N)]):
                ztrz = (self.Z.T @ self.Z.multiply((weight/diagonal**2)[:, None])).toarray()
                trace = np.sum(weight/diagonal)-np.sum(gbar*ztrz.T)-np.sum((vx @ invxx)*vx*weight[:, None])
                grad.append(0.5*(trace-np.sum(weight*py**2)))
        else:
            cz = C[:, :self.q]
            cxy = C[:, self.q:].toarray()
            tc_vxy = (cxy-cz @ (gbar @ SX))/den[:, None]
            tc_vx, tc_vy = tc_vxy[:, :self.p], tc_vxy[:, -1]
            tc_py = tc_vy-tc_vx @ beta
            ztrz = (cz.T @ cz.multiply((1/den**2)[:, None])).toarray()
            trace_tc = np.sum(h/den)-np.sum(gbar*ztrz.T)-np.sum((tc_vx @ invxx)*tc_vx)
            grad.append(0.5*(trace_tc-np.square(tc_py).sum()))
            if self.noise_scale:
                # trace(P V)=N-p and y'P V P y=y'P y provide the diagonal-noise score without an N-square matrix.
                grad.append((0.5*(self.N-self.p-quad)-np.dot(theta[:len(self.names)], grad))/phi)
        return nll, np.asarray(grad, float)

    def fit(self, start=None, fixed=None, multistart=True):
        fixed = {} if fixed is None else dict(fixed)
        initial = np.full(len(self.parameter_names), 0.25) if start is None else np.array(start, float)
        if start is None:
            if "agent" in self.names:
                initial[self.names.index("agent")] = 1.0
            if self.noise_scale:
                initial[-1] = 1.0
        free = np.array([j for j in range(len(initial)) if j not in fixed], int)
        for j, value in fixed.items():
            initial[j] = value
        bounds = [(1e-7, None) if self.parameter_names[j] == "noise_scale" and not self.singleton_tc else (0.0, None) for j in free]

        def fun(v):
            theta = initial.copy()
            theta[free] = v
            val, grad = self.objective(theta)
            return val, grad[free]

        starts = [initial[free]]
        if multistart and not fixed:
            starts.append(np.maximum(initial[free]*0.2, 0.01))
        results = [minimize(fun, st, jac=True, method="L-BFGS-B", bounds=bounds,
                            options={"ftol": 2e-12, "gtol": 1e-6, "maxiter": 2000, "maxls": 40}) for st in starts]
        best = min(results, key=lambda r: r.fun)
        theta = initial.copy()
        theta[free] = best.x
        val, grad = self.objective(theta)
        pg = grad[free].copy()
        pg[(theta[free] <= 1e-8) & (pg > 0)] = 0
        return REMLResult(theta, val, bool(best.success), grad, float(np.max(np.abs(pg))), int(best.nit), str(best.message))

if __name__ == "__main__":
    frame = pd.read_parquet(DATA / 'general_runs.parquet')
    frame = frame[frame.model.isin(["DeepSeek", "GLM", "Qwen", "gpt-oss-120b"]) & frame.materially.notna()]
    cells = frame.groupby(["model", "task_id", "condition"], observed=True).agg(k=("materially", "sum"), n=("materially", "size")).reset_index()
    cells["logodds"] = np.log((cells.k + 0.5) / (cells.n - cells.k + 0.5))
    cells["sampling_variance"] = 1 / (cells.k + 0.5) + 1 / (cells.n - cells.k + 0.5)
    m = CrossedREML(cells, cells.logodds, cells.sampling_variance, noise_scale=False, provider_fixed=False, agent_task=False)
    r = m.fit()
    if not r.converged or r.projected_gradient > 0.005:
        r2 = m.fit(r.variance + 0.001, multistart=True)
        if r2.nll < r.nll or (r2.converged and r2.projected_gradient < r.projected_gradient):
            r = r2
    v = dict(zip(m.names, [float(x) for x in r.variance[:len(m.names)]]))
    total = sum(v.values())
    result = {"runs": int(cells.n.sum()), "cells": len(cells), "converged": bool(r.converged), "shares_exact_pct": {k: 100 * x / total for k, x in v.items()}}
    (OUTPUT / 'variance.json').write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

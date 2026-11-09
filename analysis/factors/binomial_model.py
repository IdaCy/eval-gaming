"""Binomial likelihood with a background probability, shared slopes, and model-specific gains."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def _unpack(theta, spec):
    out = {}
    i = 0
    for name, shape in spec:
        n = int(np.prod(shape)) if shape else 1
        out[name] = theta[i:i + n].reshape(shape) if shape else theta[i]
        i += n
    return out

class Fit:

    def __init__(self, data, n_models, n_alpha, mode='shared', R=None, ridge=0.5, ridge_alpha=0.01, fix_gain=0, gain_free=True):
        self.d = data
        self.M = n_models
        self.A = n_alpha
        self.mode = mode
        self.R = R
        self.ridge = ridge
        self.ridge_alpha = ridge_alpha
        self.fix_gain = fix_gain
        self.gain_free = gain_free
        J = data['X'].shape[1]
        L = 0 if data.get('Z') is None else data['Z'].shape[1]
        self.spec = [('alpha', (n_alpha,)), ('w', (J,)), ('lg', (n_models,)), ('beta', (L,))]

    def eta(self, p):
        d = self.d
        X = d['X']
        w = p['w']
        lg = p['lg'].copy()
        if not self.gain_free:
            lg[:] = 0.0
        lg[self.fix_gain] = 0.0
        lin = np.exp(lg)[d['mi']] * (X @ w)
        e = p['alpha'][d['ai']] + lin
        if d.get('Z') is not None and d['Z'].shape[1] > 0:
            e = e + d['Z'] @ p['beta']
        return e

    def nll(self, theta):
        p = _unpack(theta, self.spec)
        d = self.d
        e = self.eta(p)
        s = expit(e)
        P = d['mu'] + (1 - d['mu']) * s
        P = np.clip(P, 1e-09, 1 - 1e-09)
        ll = d['k'] * np.log(P) + (d['n'] - d['k']) * np.log(1 - P)
        pen = self.ridge_alpha * np.sum(p['alpha'] ** 2)
        pen += self.ridge * np.sum(p['w'] ** 2)
        pen += 0.1 * np.sum(p['lg'] ** 2)
        if 'beta' in p and np.size(p['beta']):
            pen += 0.1 * np.sum(p['beta'] ** 2)
        g_e = (d['k'] / P - (d['n'] - d['k']) / (1 - P)) * (1 - d['mu']) * s * (1 - s)
        grad = self._grad(p, g_e)
        gpen = self._gpen(p)
        return (-ll.sum() + pen, -grad + gpen)

    def _grad(self, p, g_e):
        d = self.d
        X = d['X']
        parts = []
        ga = np.bincount(d['ai'], weights=g_e, minlength=self.A)
        parts.append(ga)
        w = p['w']
        lg = p['lg'].copy()
        if not self.gain_free:
            lg[:] = 0.0
        lg[self.fix_gain] = 0.0
        gainv = np.exp(lg)[d['mi']]
        gw = X.T @ (g_e * gainv)
        parts.append(gw)
        xl = X @ w
        glg = np.bincount(d['mi'], weights=g_e * gainv * xl, minlength=self.M)
        glg[self.fix_gain] = 0.0
        if not self.gain_free:
            glg[:] = 0.0
        parts.append(glg)
        if d.get('Z') is not None and d['Z'].shape[1] > 0:
            parts.append(d['Z'].T @ g_e)
        else:
            parts.append(np.zeros(0))
        return np.concatenate([np.atleast_1d(x).ravel() for x in parts])

    def _gpen(self, p):
        parts = [2 * self.ridge_alpha * p['alpha']]
        parts.append(2 * self.ridge * p['w'])
        parts.append(0.2 * p['lg'])
        parts.append(0.2 * p['beta'] if np.size(p['beta']) else np.zeros(0))
        return np.concatenate([np.atleast_1d(x).ravel() for x in parts])

    def predict(self, data):
        old = self.d
        self.d = data
        e = self.eta(self.p)
        self.d = old
        s = expit(e)
        return (data['mu'] + (1 - data['mu']) * s, s)

    def gains(self):
        lg = self.p['lg'].copy()
        lg[self.fix_gain] = 0
        return np.exp(lg)

    def weights(self):
        return self.p['w']
        return self.R @ self.p['th']

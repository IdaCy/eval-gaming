"""Nested cross-validation selecting task and setup penalties."""
import json
import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import formula_lib as L
VERS = ['category']
TAG = ''

def unscale(f, nt, lz, lx):
    w = f.weights()
    sc = np.concatenate([np.full(nt, 1 / np.sqrt(lz)), np.full(len(w) - nt, 1 / np.sqrt(lx))])
    return w * sc

def main():
    y = sys.argv[1]
    designs = sys.argv[2:] or ['new_repositories', 'new_conditions']
    od = os.path.join(L.OUT, 'cv')
    os.makedirs(od, exist_ok=True)
    t0 = time.time()
    R = L.load_runs()
    rt = L.room_table(R, y)
    agents = rt[rt.enters].model.tolist()
    S = L.build_setups(R, y, agents)
    ctx, I = (L.Ctx(S), L.Inputs(S))
    nt = len(I.tnames)
    print(f'{y}: agents {agents}, setups {ctx.N}, runs {int(ctx.n.sum())}, events {int(ctx.k.sum())}, inputs', {v: len(n) for v, n in I.names.items()}, flush=True)
    nonconv = 0
    for design in designs:
        folds = L.design_folds(S, design)
        Q = {}

        def arr(name):
            if name not in Q:
                Q[name] = np.full(ctx.N, np.nan)
            return Q[name]
        meta = {'outcome': y, 'design': design, 'agents': agents, 'models': ctx.models, 'groups': [list(g) for g in ctx.groups], 'task_names': I.tnames, 'setup_names': I.names, 'grid_z': L.GRID_Z, 'grid_x': L.GRID_X, 'folds': []}
        for label, tr, te, inner in folds:
            fm = {'fold': label, 'n_train': int(len(tr)), 'n_test': int(len(te))}
            muv, fb, mu = L.mu_rows(ctx, tr)
            fm['mu'] = dict(zip(ctx.models, map(float, mu)))
            fm['mu_fallback'] = fb
            inner_mu = []
            for itr, ite in inner:
                m_, fb_, _ = L.mu_rows(ctx, itr)
                inner_mu.append(m_)
                if fb_:
                    fm.setdefault('inner_mu_fallback', []).append(fb_)

            def inner_ll(Xd_list):
                tot = 0.0
                for (itr, ite), m_, Xd in zip(inner, inner_mu, Xd_list):
                    fi = L.fit(ctx, itr, m_, Xd)
                    tot += float(L.ll_vec(ctx.k[ite], ctx.n[ite], L.predict(fi, ctx, ite, m_, Xd)).sum())
                return tot
            for v in VERS:
                Xs_in = [I.setup_block(v, itr) for itr, _ in inner]
                scores = {}
                for lz in L.GRID_Z:
                    for lx in L.GRID_X[v]:
                        scores[lz, lx] = inner_ll([L.design(I.T, Xs, lz, lx, ctx.N) for Xs in Xs_in])
                lz, lx = max(scores, key=scores.get)
                Xs = I.setup_block(v, tr)
                Xd = L.design(I.T, Xs, lz, lx, ctx.N)
                f = L.fit(ctx, tr, muv, Xd)
                nonconv += int(not f.res.success)
                arr(v)[te] = L.predict(f, ctx, te, muv, Xd)
                rec = {'lz': lz, 'lx': lx, 'gains': list(map(float, f.gains())), 'alpha': list(map(float, f.p['alpha'])), 'converged': bool(f.res.success)}
                w = unscale(f, nt, lz, lx)
                rec['w_task'] = list(map(float, w[:nt]))
                rec['w_setup'] = list(map(float, w[nt:]))
                fm[v] = rec
            meta['folds'].append(fm)
            print(f'  {y} {design} {label}: ' + '; '.join((f'{v} {fm[v]['lz']:g}/{fm[v]['lx']:g}' for v in VERS)) + (f'; mu fallback {fb}' if fb else '') + f' ({time.time() - t0:.0f}s)', flush=True)
        P_ = S[['model', 'era', 'task_id', 'condition', 'repo', 'family', 'k', 'n']].copy()
        P_ = pd.concat([P_, pd.DataFrame(Q)], axis=1)
        P_.to_parquet(os.path.join(od, f'preds{TAG}_{y}_{design}.parquet'), index=False)
        meta['nonconverged_full_fits_so_far'] = nonconv
        meta['fit_log_so_far'] = dict(L.FIT_LOG)
        json.dump(meta, open(os.path.join(od, f'cv{TAG}_{y}_{design}.json'), 'w'), indent=1, default=float)
        print(f'{y} {design} saved ({time.time() - t0:.0f}s); fits so far {L.FIT_LOG}', flush=True)
    print('done', f'{time.time() - t0:.0f}s')

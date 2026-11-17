"""Add alpha times the direction gap at the selected decoder-layer output."""

import hashlib
import json
import os
import sys
import time
_CFG = None

def _load():
    vec = os.environ.get('STEER_VEC')
    if not vec:
        return None
    import numpy as np
    layers = [int(x) for x in os.environ['STEER_LAYER'].split(',')]
    alpha = float(os.environ.get('STEER_ALPHA', '0'))
    mode = os.environ.get('STEER_MODE', 'all')
    if mode != 'all':
        raise ValueError('steering: only STEER_MODE=all is implemented')
    vecs, gaps, shas = ({}, {}, {})
    if len(layers) != 1:
        raise ValueError('steering: a single vector file needs exactly one layer')
    import pyarrow.parquet as pq
    vecs[layers[0]] = pq.read_table(vec, columns=['value'])['value'].to_numpy().astype('float32')
    shas[layers[0]] = hashlib.sha256(open(vec, 'rb').read()).hexdigest()
    if os.environ.get('STEER_GAP'):
        g = [float(x) for x in os.environ['STEER_GAP'].split(',')]
        if len(g) == 1:
            g = g * len(layers)
        gaps = dict(zip(layers, g))
    missing = [L for L in layers if L not in gaps]
    if missing:
        raise ValueError('steering: no gap for layers %s' % missing)
    deltas = {}
    for L in layers:
        v = vecs[L]
        nrm = float(np.linalg.norm(v))
        if abs(nrm - 1.0) > 0.001:
            raise ValueError('steering: layer %d vector norm %.6f is not 1' % (L, nrm))
        deltas[L] = alpha / len(layers) * gaps[L] * v
    return {'vec': vec, 'layers': layers, 'alpha': alpha, 'gaps': gaps, 'mode': mode, 'vec_sha256': shas, 'deltas': deltas, 'vecs': vecs, 'debug': os.environ.get('STEER_DEBUG') or None}

def register():
    global _CFG
    cfg = _load()
    if cfg is None:
        print('[steering] inactive (STEER_VEC unset)', file=sys.stderr, flush=True)
        return
    import torch
    from vllm.model_executor.models import qwen3_5 as q
    cls = q.Qwen3_5DecoderLayer
    if getattr(cls, '_steer_patched', False):
        return
    _CFG = cfg
    orig_init, orig_fwd = (cls.__init__, cls.forward)

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        L = self.layer_idx
        if L in cfg['deltas']:
            self.register_buffer('_steer_delta', torch.tensor(cfg['deltas'][L], dtype=torch.get_default_dtype()), persistent=False)
            if cfg['debug']:
                self.register_buffer('_steer_v', torch.tensor(cfg['vecs'][L], dtype=torch.float32), persistent=False)
        else:
            self._steer_delta = None

    def forward(self, *args, **kwargs):
        hidden_states, residual = orig_fwd(self, *args, **kwargs)
        delta = self._steer_delta
        if delta is None:
            return (hidden_states, residual)
        if cfg['debug'] and (not torch.compiler.is_compiling()):
            v = self._steer_v
            before = (hidden_states.float() + residual.float()) @ v
            new = hidden_states + delta
            after = (new.float() + residual.float()) @ v
            with open(cfg['debug'], 'a') as f:
                f.write(json.dumps({'layer': self.layer_idx, 'n_tokens': int(before.shape[0]), 'proj_before': float(before.mean()), 'proj_after': float(after.mean()), 'shift': float((after - before).mean()), 'shift_min': float((after - before).min()), 'shift_max': float((after - before).max()), 'expected': float(cfg['alpha'] / len(cfg['layers']) * cfg['gaps'][self.layer_idx])}) + '\n')
            return (new, residual)
        return (hidden_states + delta, residual)
    cls.__init__ = __init__
    cls.forward = forward
    cls._steer_patched = True
    import numpy as np
    receipt = {'plugin': 'steering', 'plugin_sha256': hashlib.sha256(open(__file__, 'rb').read()).hexdigest(), 'pid': os.getpid(), 'argv': sys.argv[:3], 'vec': cfg['vec'], 'layers': cfg['layers'], 'alpha': cfg['alpha'], 'gaps': {str(k): v for k, v in cfg['gaps'].items()}, 'mode': cfg['mode'], 'vec_sha256': {str(k): v for k, v in cfg['vec_sha256'].items()}, 'delta_norm': {str(k): float(np.linalg.norm(v)) for k, v in cfg['deltas'].items()}, 'debug': cfg['debug'], 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    print('[steering] ACTIVE ' + json.dumps(receipt), file=sys.stderr, flush=True)
    path = os.environ.get('STEER_RECEIPT')
    if path:
        with open(path, 'a') as f:
            f.write(json.dumps(receipt) + '\n')

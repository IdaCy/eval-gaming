"""Capture Qwen residual streams at turn starts, reasoning ends, and tool calls."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
IM_START, IM_END, THINK, THINK_END, TOOL_CALL = (248045, 248046, 248068, 248069, 248058)
ASSISTANT, NL = (74455, 198)
ATTN_NAME = 'steering_flash_gqa'
FIDELITY_SAMPLE = 2000

def causal_gqa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    assert query.shape[-2] == key.shape[-2], 'full-sequence prefill only'
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return (out.transpose(1, 2).contiguous(), None)

def structure(ids):
    n = len(ids)
    turns = []
    i = 0
    while i < n - 2:
        if ids[i] == IM_START and ids[i + 1] == ASSISTANT and (ids[i + 2] == NL):
            j = i + 3
            if j < n and ids[j] == THINK:
                ts = j + 1 if j + 1 < n and ids[j + 1] == NL else j
            else:
                ts = i + 2
            k = j
            re_, tc = (None, [])
            while k < n and ids[k] != IM_END:
                if ids[k] == THINK_END and re_ is None:
                    re_ = k
                elif ids[k] == TOOL_CALL:
                    tc.append(k)
                k += 1
            turns.append({'ts': ts, 're': re_, 'tc': tc, 'ie': min(k, n - 1), 'start': i})
            i = k
        else:
            i += 1
    return turns

def load_records(paths, wanted=None):
    for p in paths:
        with gzip.open(p, 'rt') as f:
            for line in f:
                d = json.loads(line)
                if wanted is None or (d.get('run_id') or d.get('attempt_id')) in wanted:
                    yield {'run_id': d.get('run_id') or d.get('attempt_id'), 'token_ids': d['token_ids'], 'token_ids_sha256': d.get('token_ids_sha256'), 'source': str(p), 'prompt_end': d.get('prompt_end'), 'reasoning_end': d.get('reasoning_end')}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.8-27B')
    ap.add_argument('--revision', default='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
    ap.add_argument('--model-id', default='Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--order', help='file with run ids in priority order (others follow)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--layers', default='32')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    from transformers import AttentionInterface, Qwen3_5ForConditionalGeneration
    AttentionInterface.register(ATTN_NAME, causal_gqa_attention)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16, device_map={'': args.device}, attn_implementation='sdpa')
    if getattr(base.model, 'visual', None) is not None:
        base.model.visual = None
    torch.cuda.empty_cache()
    text = base.model.language_model
    text.config._attn_implementation = ATTN_NAME
    lm_head = base.lm_head
    nl = text.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(',')] if args.layers else list(range(4, nl, 4)) + [nl - 1]
    layers = sorted(set(layers))
    text.eval()
    print(json.dumps({'event': 'loaded', 'seconds': round(time.time() - t0, 1), 'layers': layers, 'hidden': text.config.hidden_size, 'device': args.device}), flush=True)
    meta = []
    for r in load_records(args.inputs):
        meta.append((r['run_id'], len(r['token_ids'])))
    prio = {}
    if args.order:
        for k, line in enumerate(Path(args.order).read_text().split()):
            prio.setdefault(line, k)
    meta.sort(key=lambda x: (prio.get(x[0], 10 ** 9), -x[1]))
    loads = [0] * args.nshards
    mine = []
    for rid, n in meta:
        s = min(range(args.nshards), key=lambda i: loads[i])
        loads[s] += n
        if s == args.shard:
            mine.append(rid)
    order = {rid: k for k, rid in enumerate(mine)}
    pending = [rid for rid in mine if not (out / (rid + '.json')).exists()]
    if args.limit:
        pending = pending[:args.limit]
    print(json.dumps({'event': 'shard', 'shard': args.shard, 'runs': len(mine), 'pending': len(pending)}), flush=True)
    records = sorted(load_records(args.inputs, set(pending)), key=lambda r: order[r['run_id']])
    captured = {}
    seg_holder = {}

    def make_hook(L):

        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h[0]
            idx = seg_holder['anchor_idx']
            captured['a', L] = h.index_select(0, idx).float().cpu()
            seg, nt = (seg_holder['seg'], seg_holder['nturns'])
            sums = torch.zeros((nt + 1, h.shape[-1]), dtype=torch.float32, device=h.device)
            step = 32768
            for s in range(0, h.shape[0], step):
                sums.index_add_(0, seg[s:s + step], h[s:s + step].float())
            captured['r', L] = sums[1:].cpu()
        return hook
    handles = [text.layers[L].register_forward_hook(make_hook(L)) for L in layers]
    for r in records:
        rid, ids = (r['run_id'], r['token_ids'])
        t1 = time.time()
        sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        turns = structure(ids)
        anchors = []
        for k, t in enumerate(turns, start=1):
            anchors.append((t['ts'], 0, k))
            if t['re'] is not None:
                anchors.append((t['re'], 1, k))
            for p in t['tc']:
                anchors.append((p, 2, k))
        seg = torch.zeros(len(ids), dtype=torch.long)
        counts = np.zeros(len(turns), dtype=np.int32)
        for k, t in enumerate(turns, start=1):
            if t['re'] is not None and t['re'] - t['ts'] > 1:
                seg[t['ts'] + 1:t['re']] = k
                counts[k - 1] = t['re'] - t['ts'] - 1
        seg_holder.update(anchor_idx=torch.tensor([a[0] for a in anchors], dtype=torch.long, device=dev), seg=seg.to(dev), nturns=len(turns))
        gen = [q for t in turns for q in range(t['ts'] + 1, t['ie'] + 1)]
        rng = random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:12], 16))
        sample = sorted(rng.sample(gen, min(FIDELITY_SAMPLE, len(gen)))) if gen else []
        captured.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        with torch.inference_mode():
            x = torch.tensor([ids], device=dev)
            pos = torch.arange(len(ids), device=dev)[None]
            hid = text(input_ids=x, position_ids=pos, use_cache=False).last_hidden_state[0]
            top1, lps = ([], [])
            if sample:
                q = torch.tensor(sample, device=dev)
                for s in range(0, len(sample), 500):
                    qq = q[s:s + 500]
                    logits = lm_head(hid.index_select(0, qq - 1)).float()
                    lp = torch.log_softmax(logits, -1)
                    tgt = x[0].index_select(0, qq)
                    top1.append((logits.argmax(-1) == tgt).float().cpu())
                    lps.append(lp.gather(1, tgt[:, None])[:, 0].cpu())
            del hid
        act = np.stack([captured['a', L].numpy() for L in layers]).astype(np.float32)
        rme = np.stack([captured['r', L].numpy() for L in layers]).astype(np.float32)
        rme = rme / np.maximum(counts, 1)[None, :, None]
        rme[:, counts == 0, :] = np.nan
        amax = float(np.nanmax(np.abs(act))) if act.size else 0.0
        store = np.float16 if amax < 60000 else np.float32
        npz = out / (rid + '.npz')
        tmp = out / (rid + '.tmp.npz')
        np.savez(tmp, act=act.astype(store), rmean=rme.astype(store), anchor_pos=np.array([a[0] for a in anchors], dtype=np.int32), anchor_type=np.array([a[1] for a in anchors], dtype=np.int8), anchor_turn=np.array([a[2] for a in anchors], dtype=np.int16), rmean_count=counts, layers=np.array(layers, dtype=np.int16))
        os.replace(tmp, npz)
        fid_top1 = float(torch.cat(top1).mean()) if top1 else None
        fid_lp = float(torch.cat(lps).mean()) if lps else None
        rec = {'run_id': rid, 'model_id': args.model_id, 'n_tokens': len(ids), 'n_turns': len(turns), 'n_anchors': len(anchors), 'token_ids_sha256': sha, 'token_ids_sha256_input': r.get('token_ids_sha256'), 'source': r['source'], 'layers': layers, 'fidelity_top1': fid_top1, 'fidelity_mean_logprob': fid_lp, 'fidelity_n': len(sample), 'turns_without_reasoning_end': sum((1 for t in turns if t['re'] is None)), 'turn_start_after_newline': sum((1 for t in turns if ids[t['ts']] == NL)), 'max_abs_activation': amax, 'seconds': round(time.time() - t1, 2), 'peak_gb': round(torch.cuda.max_memory_allocated(dev) / 1000000000.0, 1), 'npz_sha256': hashlib.sha256(npz.read_bytes()).hexdigest(), 'npz_bytes': npz.stat().st_size, 'device': torch.cuda.get_device_name(dev), 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        (out / (rid + '.json')).write_text(json.dumps(rec) + '\n')
        print(json.dumps({'event': 'run', 'run_id': rid, 'n_tokens': len(ids), 'seconds': rec['seconds'], 'top1': fid_top1, 'peak_gb': rec['peak_gb']}), flush=True)
    for h in handles:
        h.remove()
    print(json.dumps({'event': 'shard_done', 'shard': args.shard, 'seconds': round(time.time() - t0, 1)}), flush=True)
if __name__ == '__main__':
    main()

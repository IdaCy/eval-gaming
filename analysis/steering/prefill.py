"""Capture layer-32 first-turn states from the paired prompt table."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from extract import structure
ATTN_NAME = 'steering_flash_gqa'

def causal_gqa_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=query.shape[1] != key.shape[1])
    return (out.transpose(1, 2).contiguous(), None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3.8-27B')
    ap.add_argument('--revision', default='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
    ap.add_argument('--model-id', default='Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-runs', nargs='*', default=[])
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    from transformers import AttentionInterface, AutoTokenizer, Qwen3_5ForConditionalGeneration
    AttentionInterface.register(ATTN_NAME, causal_gqa_attention)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rows = []
    for p in args.inputs:
        import pandas as pd
        for i, row in pd.read_parquet(p).iterrows():
            rows.append({'_src': 'first_turn_main', '_line': int(i), 'task_id': row.task_id, 'variant_id': row.variant_id, 'messages': json.loads(row.messages_json), 'tools': json.loads(row.tools_json)})
    ref = {}
    for p in args.check_runs:
        with gzip.open(p, 'rt') as f:
            for line in f:
                d = json.loads(line)
                key = (d['task_id'], d['variant_id'])
                if key in ref:
                    continue
                t = structure(d['token_ids'])
                if t:
                    ref[key] = d['token_ids'][:t[0]['ts'] + 1]
    base = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16, device_map={'': args.device}, attn_implementation='sdpa')
    base.model.visual = None
    torch.cuda.empty_cache()
    text = base.model.language_model
    text.config._attn_implementation = ATTN_NAME
    layers = [32]
    cap = {}
    hooks = [text.layers[L].register_forward_hook((lambda L: lambda m, i, o: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[0, -1].float().cpu()))(L)) for L in layers]
    acts = np.zeros((len(rows), len(layers), text.config.hidden_size), dtype=np.float32)
    meta = []
    checked = matched = 0
    t0 = time.time()
    for k, r in enumerate(rows):
        tools = [{'type': t['type'], 'function': {'name': t['function']['name'], 'description': t['function']['description'], 'parameters': json.loads(json.dumps(t['function']['parameters'], sort_keys=True))}} for t in r['tools']] if r.get('tools') else None
        text_prompt = tok.apply_chat_template(r['messages'], tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        ids = list(tok(text_prompt, add_special_tokens=False)['input_ids'])
        key = (r.get('task_id'), r.get('variant_id'))
        ok = None
        if key in ref:
            checked += 1
            ok = ref[key] == ids
            matched += int(ok)
        with torch.inference_mode():
            x = torch.tensor([ids], device=args.device)
            text(input_ids=x, position_ids=torch.arange(len(ids), device=args.device)[None], use_cache=False)
        for j, L in enumerate(layers):
            acts[k, j] = cap[L].numpy()
        meta.append({'i': k, 'src': r['_src'], 'line': r['_line'], 'task_id': r.get('task_id'), 'variant_id': r.get('variant_id'), 'set': r.get('set'), 'pair': r.get('pair', r.get('pair_key')), 'side': r.get('side'), 'prompt_id': r.get('prompt_id'), 'n_tokens': len(ids), 'ids_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(), 'matches_real_run': ok})
        if k % 200 == 0:
            print(json.dumps({'done': k, 'seconds': round(time.time() - t0, 1), 'checked': checked, 'matched': matched}), flush=True)
    for h in hooks:
        h.remove()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, acts=acts.astype(np.float16), layers=np.array(layers))
    Path(args.out).with_suffix('.meta.jsonl').write_text('\n'.join((json.dumps(m) for m in meta)) + '\n')
    print(json.dumps({'event': 'done', 'n': len(rows), 'checked_against_real_runs': checked, 'matched': matched, 'seconds': round(time.time() - t0, 1), 'model_id': args.model_id}), flush=True)
if __name__ == '__main__':
    main()

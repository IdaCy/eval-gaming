"""Launch an isolated vLLM engine with the recorded Qwen generation settings."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


def configuration(data, direction, alpha, port, tensor_parallel):
    parameters = {row['name']: row for row in csv.DictReader((data / 'direction_parameters.csv').open())}
    row = parameters[direction]
    vector = data / 'directions' / (direction + '.parquet')
    environment = {'VLLM_USE_FLASHINFER_SAMPLER': '0', 'VLLM_PLUGINS': 'steering'}
    if alpha:
        environment.update(STEER_VEC=str(vector.resolve()), STEER_LAYER=str(row['layer']),
                           STEER_ALPHA=str(alpha), STEER_GAP=str(row['unit_gap']), STEER_MODE='all')
    key = hashlib.sha256(json.dumps({'vector_sha256': hashlib.sha256(vector.read_bytes()).hexdigest(),
                                     'alpha': alpha, 'gap': row['unit_gap'], 'layer': row['layer'],
                                     'tensor_parallel': tensor_parallel}, sort_keys=True).encode()).hexdigest()[:16]
    environment['VLLM_CACHE_ROOT'] = str((OUTPUT / 'vllm_cache' / key).resolve())
    command = ['vllm', 'serve', 'Qwen/Qwen3.8-27B-FP8', '--revision',
               '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a', '--tokenizer-revision',
               '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a', '--served-model-name', 'qwen3.8-27b',
               '--host', '127.0.0.1', '--port', str(port), '--max-model-len', '262144',
               '--enable-prefix-caching', '--reasoning-parser', 'qwen3', '--enable-auto-tool-choice',
               '--tool-call-parser', 'qwen3_coder', '--language-model-only', '--generation-config', 'auto',
               '--tensor-parallel-size', str(tensor_parallel)]
    return command, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--direction', default='gaming',
                        choices=['gaming', 'evaluation', 'random_1', 'random_2', 'random_3'])
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--port', type=int, default=8100)
    parser.add_argument('--tensor-parallel', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    command, updates = configuration(args.data, args.direction, args.alpha, args.port, args.tensor_parallel)
    if args.dry_run:
        print(json.dumps({'command': command, 'environment': updates}, indent=2))
        return
    environment = {key: value for key, value in os.environ.items() if not key.startswith('STEER_')}
    environment.update(updates)
    Path(environment['VLLM_CACHE_ROOT']).mkdir(parents=True, exist_ok=True)
    raise SystemExit(subprocess.call(command, env=environment))


if __name__ == '__main__':
    main()

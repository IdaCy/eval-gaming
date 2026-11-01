# The Gaming Problem

Code and data for "The Gaming Problem: When Do AI Evaluations Become Compromised?".
Authors: Ida Caspary, Rossella Arcucci, Kaiwen Wang

We run evaluations on coding agents, varying realism factors. We then check what they can infer about how they are checked, and how it changes their work. We focus on studying factors that make them game.

Here in this repository, we release the code and data for the paper. The tasks and setups, all prompts, the runs, the judges' labels and agents' answers are on Hugging Face in [eval-gaming](https://huggingface.co/datasets/idacy/eval-gaming).

Outputs restricted to confidential research access are excluded.

License: MIT for the code; CC BY 4.0 for the data created in this work. Tasks keep the licenses of their sources.

## Installation

Use Python 3.12. The paper analyses run on CPU.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
```

The download is anonymous. The default download contains tables, tasks, setups, prompts and samples. Add `--include-bulk` to download `data/runs/` and `data/labels/` when they are available. Optional GPU dependencies are pinned in `analysis/steering/requirements-extraction.txt` and `analysis/steering/requirements-generation.txt`; generation also requires Docker and prepared task assets.

Fetch-only requests use `python -m harness.download --task bbc__psammead-1362 --output local_tasks --execute`, then `python -m harness.run_task --task bbc__psammead-1362 --family native12 --local-request local_tasks/bbc__psammead-1362/problem_statement.txt --output runs/psammead`. The request and adjacent `retrieval.json` must match the recorded provenance and frozen variant hash; `--local-variant variant.json` accepts a complete local variant with the recorded source-definition SHA-256. Run destinations must be new; resuming is unsupported.

## Reproducing the paper


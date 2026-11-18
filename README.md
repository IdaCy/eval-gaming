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

Table A10 has four display differences (reproduced versus paper): DeepSeek T2, 1.5 versus 2; Luna S1, 9.1 versus 9; GPT-5.5 T2, 0.9 versus 1; gpt-oss O3, · versus –. The numerical reference checks pass; the paper-display check reports these differences.

| Paper item | Command | Output file |
| --- | --- | --- |
| All available analyses | `python reproduce.py` | `figures/; tables/` |
| Section 4: task, setup and run counts | `python reproduce.py --parts setup` | `tables/setup/run_counts.json; tables/setup/setup.json` |
| Table A1 | `python reproduce.py --parts setup` | `tables/setup/repeated/overlap_cells.csv` |
| Table A2: task counts and dates | `python reproduce.py --parts setup` | `tables/setup/task_counts.csv` |
| Table A3: released-model roster | `python reproduce.py --parts setup` | `tables/setup/agents.csv` |
| Table A4: cue texts | `python reproduce.py --parts setup` | `tables/setup/cue_texts.csv` |
| Table A5: input factors | `python reproduce.py --parts setup` | `tables/setup/setup_counts.csv; tables/setup/task_rating_fields.csv` |
| Appendix A.10: prompts and setup carriers | `python reproduce.py --parts setup` | `tables/setup/setup.json` |
| Figures 1–3: drawing sources unavailable | — | — |
| Sections 5.1–5.2 and Tables A6–A7 | `python reproduce.py --parts awareness_grading` | `tables/awareness_grading/paper_numbers.json` |
| Table A8 | `python reproduce.py --parts awareness_grading` | `tables/awareness_grading/tableA8_levels.csv` |
| Tables A9–A10: run-level refits | `python reproduce.py --parts action_refits` | `tables/action_refits/tablesA9_A10.csv` |
| Table A11 | `python reproduce.py --parts awareness_grading` | `tables/awareness_grading/pooled_differences.csv; tables/awareness_grading/tableA11_judged.csv` |
| Figure 4 | `python reproduce.py --parts awareness_grading` | `figures/awareness_grading/fig_awareness_two_sentences.pdf` |
| Figure 5 | `python reproduce.py --parts awareness_grading` | `figures/awareness_grading/fig_auc_by_cue_and_agent.pdf` |
| Figure 6: released models | `python reproduce.py --parts awareness_grading` | `figures/awareness_grading/fig_data_tasks_reports.pdf` |
| Figure A1: frozen printed intervals | `python reproduce.py --parts paper_values` | `figures/paper_values/fig_A1_frozen.pdf; tables/paper_values/figure_a1_frozen.csv` |
| Figure A2 | `python reproduce.py --parts awareness_grading` | `figures/awareness_grading/fig_what_changed_name.pdf` |
| Section 5.3 and Table A12 | `python reproduce.py --parts expectation` | `tables/expectation/table_A12.csv; tables/expectation/work_summary.json` |
| Table A13 | `python reproduce.py --parts expectation` | `tables/expectation/table_A13.csv` |
| Section 5.4 and Table A14 | `python reproduce.py --parts expectation` | `tables/expectation/table_A14_paper.csv; tables/expectation/forward_summary.json` |
| Equation A1 | `python reproduce.py --parts expectation` | `tables/expectation/response_fit.json` |
| Equation 4 | `python reproduce.py --parts factors` | `tables/factors/formula/final/final_G_rmask.json` |
| Figure 7 | `python reproduce.py --parts factors` | `figures/factors/fig_input_factors.pdf` |
| Figure A3 | `python reproduce.py --parts factors` | `figures/factors/fig_differences_per_agent.pdf` |
| Figure A4 | `python reproduce.py --parts factors` | `figures/factors/fig_issue_age.pdf` |
| Table A15 | `python reproduce.py --parts factors` | `tables/factors/table_A15.csv` |
| Table A16 | `python reproduce.py --parts factors` | `tables/factors/table_A16.csv` |
| Table A17 | `python reproduce.py --parts factors` | `tables/factors/table_A17.csv` |
| Table A18 | `python reproduce.py --parts factors` | `tables/factors/table_A18.csv` |
| Section 5.6 | `python reproduce.py --parts sensitivity_records` | `tables/sensitivity_records/sensitivity_summary.json; tables/sensitivity_records/setups_corrected/setups58_agents.json` |
| Figure 8: released models | `python reproduce.py --parts sensitivity_records` | `figures/sensitivity_records/fig_8.pdf` |
| Figure A5: released models | `python reproduce.py --parts sensitivity_records` | `figures/sensitivity_records/fig_A5.pdf` |
| Figures A6–A9 | `python reproduce.py --parts sensitivity_records` | `figures/sensitivity_records/fig_A6.pdf; figures/sensitivity_records/fig_A7.pdf; figures/sensitivity_records/fig_A8.pdf; figures/sensitivity_records/fig_A9.pdf` |
| Sections 6.1–6.2 | `python reproduce.py --parts sensitivity_records` | `tables/sensitivity_records/records.json` |
| Tables 1 and A19: available entries | `python reproduce.py --parts sensitivity_records` | `tables/sensitivity_records/table_1.csv; tables/sensitivity_records/table_A19.csv` |
| Figure 9 | `python reproduce.py --parts sensitivity_records` | `figures/sensitivity_records/fig_9.pdf` |
| Figure A10 | `python reproduce.py --parts sensitivity_records` | `figures/sensitivity_records/fig_A10.pdf` |
| Figure A11 | `python reproduce.py --parts action_refits` | `tables/action_refits/fig_A11.pdf; tables/action_refits/answer_key_example.json` |
| Section 6.3 and Table A20 | `python reproduce.py --parts steering` | `tables/steering/table_a20.csv; tables/steering/directions/facts.json` |
| Figure 10 | `python reproduce.py --parts steering` | `figures/steering/fig_steering_random_rows.pdf` |
| Figure A12 | `python reproduce.py --parts steering` | `figures/steering/fig_steering_lines_what_qwen_says.pdf` |
| Figure A13 | `python reproduce.py --parts steering` | `figures/steering/fig_steering_controls_with_literature_what_qwen_does.pdf` |
| Appendix A.9: turn windows | `python reproduce.py --parts steering` | `tables/steering/turn_windows.csv` |
| Original AUC and action refits: nine models plus the corrected credited-tests route | `python reproduce.py --parts action_refits` | `tables/action_refits/auc_original/; tables/action_refits/auc_credited/; tables/action_refits/validation.json` |
| Section 5.3: credited-tests work refits | `python reproduce.py --parts action_refits` | `tables/action_refits/expectation/work_summary.json` |
| Conditional choice: 87/85/66/52% | `python reproduce.py --parts paper_values` | `tables/paper_values/conditional_choice.csv` |
| Grading-reason labels: 1,025/1,094 | `python reproduce.py --parts paper_values` | `tables/paper_values/reasoning_share.json` |

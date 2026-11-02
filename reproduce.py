"""Recompute the paper's available numerical analyses and figures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PARTS = ("setup", "awareness_grading", "expectation", "factors", "sensitivity_records", "steering", "action_refits", "paper_values")
VALIDATION_REPORTS = {"setup": "setup.json", "action_refits": "validation.json"}


def validation_status(area, output):
    if area not in VALIDATION_REPORTS:
        return {"reference": "not available", "paper": "not available", "differences": []}
    path = output / VALIDATION_REPORTS[area]
    report = json.loads(path.read_text())
    if area == "setup":
        checks = report.get("checks")
        if not isinstance(checks, dict) or not checks or any(type(value) is not bool for value in checks.values()):
            raise ValueError("setup: validation report lacks explicit checks")
        passed = all(checks.values())
        if type(report.get("example_request_bytes_match")) is not bool:
            raise ValueError("setup: validation report lacks the prompt-byte diagnostic")
        return {"reference": "PASS" if passed else "FAIL",
                "paper": "PASS (line-based prompts and setup checks)" if passed else "FAIL",
                "report": str(path), "differences": [],
                "diagnostics": {"example_request_bytes_match": report["example_request_bytes_match"]}}
    if type(report.get("reference_passed")) is not bool or type(report.get("paper_passed")) is not bool:
        raise ValueError(f"{area}: validation report lacks explicit reference and paper results")
    differences = [dict(check=name, **difference) for name, check in report.items()
                   if name.startswith("paper_") and isinstance(check, dict)
                   for difference in check.get("differences", [])]
    return {"reference": "PASS" if report["reference_passed"] else "FAIL",
            "paper": "PASS" if report["paper_passed"] else "FAIL",
            "report": str(path), "differences": differences}


class Reproduction:
    def __init__(self, data, output, jobs):
        self.data = data
        self.output = output
        self.jobs = jobs
        self.executor = ThreadPoolExecutor(max_workers=jobs)
        self.environment = dict(
            os.environ,
            EVAL_GAMING_DATA=str(data),
            EVAL_GAMING_OUTPUT=str(output),
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONHASHSEED="0",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MPLBACKEND="Agg",
        )

    def execute(self, area, script, *arguments):
        if area == "setup":
            command = [sys.executable, "-B", "-m", "harness." + script]
        else:
            command = [sys.executable, "-B", str(ROOT / "analysis" / area / (script + ".py"))]
        command.extend(map(str, arguments))
        environment = dict(self.environment, MPLCONFIGDIR=str(self.output / ".cache" / "matplotlib" / area))
        if area == "factors":
            environment.update(OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{area}/{script} failed ({result.returncode})\n{result.stdout}\n{result.stderr}")
        print(f"{area}/{script}: complete ({time.monotonic() - started:.1f}s)", flush=True)

    def run(self, area, script, *arguments):
        return self.executor.submit(self.execute, area, script, *arguments).result()

    def parallel(self, area, commands):
        futures = [self.executor.submit(self.execute, area, *command) for command in commands]
        for future in futures:
            future.result()

    def part(self, area):
        data = self.data / "tables" / area
        output = self.output / "tables" / area
        figures = self.output / "figures" / area
        output.mkdir(parents=True, exist_ok=True)
        figures.mkdir(parents=True, exist_ok=True)
        if area == "setup":
            self.parallel(area, [
                ("reproduce_inventory", "--data", self.data, "--output", output / "run_counts.json"),
                ("setup_summary", "--data", self.data, "--output", output / "setup.json"),
                ("reproduce_repeated", "--data", self.data, "--output", output / "repeated"),
                ("metadata", "--data", self.data, "--output", output),
            ])
        elif area == "awareness_grading":
            commands = [
                ("compare_sets", "--data", data, "--output", output),
                ("report_lengths", "--data", data, "--output", output),
                ("agreement", "--data", data, "--output", output / "agreement.json"),
            ]
            inputs = {
                "awareness_tests": "action_rows", "grading_tests": "action_rows",
                "action_counts": "action_rows", "test_verdicts": "action_rows",
                "pair_verdicts": "pair_labels", "general_judges": "general_judge_task_sums",
                "noncue": "noncue_cell_sums", "final_messages": "final_message_rows",
                "name_changes": "name_action_rows", "conditioned_judges": "unaware_task_sums",
            }
            for name, table in inputs.items():
                suffix = ".csv" if name in {"pair_verdicts", "noncue"} else ".json"
                commands.append((name, "--data", data / (table + ".parquet"), "--output", output / (name + suffix)))
            for table, directory in [("corrected_route_actions", "auc_corrected"), ("corrected_credited_actions", "auc_credited")]:
                commands.append(("auc_original", "--frames", data / (table + ".parquet"), "--models", "gpt-oss", "--output", output / directory, "--processes", "1"))
            self.parallel(area, commands)
            self.run(area, "summary_tables", "--data", data, "--results", output)
            self.run(area, "figures", "--data", data, "--results", output, "--output", figures)
        elif area == "expectation":
            files = {"response_fit": "response_fit.json", "reasoning_comparison": "reasoning_comparisons.csv", "author_mentions": "author_counts.csv"}
            self.parallel(area, [(name, "--data", data, "--output", output / files[name] if name in files else output)
                                 for name in ["beliefs", "forward_test", "response_fit", "reasoning_comparison", "author_mentions", "work_changes"]])
        elif area == "factors":
            commands = [("formula", "cv", "--design", design, "--data", data, "--output", output / "formula")
                        for design in ["new_repositories", "new_conditions"]]
            commands += [("formula_validation", design, "--outcome", outcome, "--data", data, "--output", output / "validation")
                         for design in ["C2", "C3", "C4"] for outcome in ["judged", "notests"]]
            commands += [("transfer", "--outcome", outcome, "--data", data, "--output", output / "transfer") for outcome in ["G_rmask", "G_tests"]]
            commands += [(name, "--data", data, "--output", output) for name in ["variance", "task_features", "task_age", "training"]]
            commands += [(name, "--data", data, "--output", output / (name + ".json")) for name in ["additional_factors", "new_cues", "transcript_factors"]]
            self.parallel(area, commands)
            self.run(area, "formula", "fit", "--data", data, "--output", output / "formula")
            self.run(area, "tables", "--results", output, "--output", output)
            self.run(area, "figures", "--data", data, "--results", output, "--output", figures)
        elif area == "sensitivity_records":
            self.parallel(area, [
                ("setups_58", output / "setups_corrected", "masked"),
                ("setups_58", output / "setups_paper", "old"),
                ("split_58", output / "setups_corrected", "masked"),
                ("one_line_bounds",), ("five_behaviors",), ("pass_correlation",),
                ("sensitivity_summary",), ("variance",),
            ])
            self.run(area, "records")
            self.parallel(area, [
                ("figure_sensitivity_15", data / "sensitivity_sets.csv", figures / "fig_8.png"),
                ("figure_sensitivity_all", data / "sensitivity_sets.csv", figures / "fig_A5.png"),
                ("figure_setups",), ("figure_examples",),
            ])
        elif area == "steering":
            self.parallel(area, [
                ("build_directions", "--data", data, "--output", output / "directions"),
                ("analyze", "--data", data, "--output", output),
            ])
            self.run(area, "plot", "--levels", output / "levels.csv", "--output", figures)
        elif area == "action_refits":
            with (data / "auc_sources.csv").open(newline="") as stream:
                sources = list(csv.DictReader(stream))
            models = list(dict.fromkeys(row["model"] for row in sources))
            commands = []
            for model in models:
                commands.append(("auc_original", "--source-index", data / "auc_sources.csv",
                                 "--models", model, "--output", output / "auc_original", "--processes", "1"))
                frames = [data / row["frame"] for row in sources if row["model"] == model]
                if model == "gpt-oss":
                    frames = [data / "awareness_grading/gpt-oss-credited.parquet"]
                commands.append(("work_changes", "--input", *frames, "--model", model,
                                 "--output", output / "expectation/per_model"))
            commands.append(("auc_original", "--source-index", data / "credited_sources.csv",
                             "--output", output / "auc_credited", "--processes", "1"))
            self.parallel(area, commands)
            self.run(area, "work_changes", "--data", data, "--from-parts", output / "expectation/per_model",
                     "--output", output / "expectation")
            self.run(area, "reproduce", "--data", data, "--output", output, "--summarize-only")
        elif area == "paper_values":
            self.run(area, "reproduce", "--data", data, "--output", output, "--figures", figures)
        return validation_status(area, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT)
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    parser.add_argument("--jobs", type=int, default=available, help="Total concurrent analysis processes.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    selected = list(dict.fromkeys(args.parts))
    data, output = args.data.resolve(), args.output.resolve()
    missing = [area for area in selected if not (data / "tables" / area).is_dir()]
    if missing:
        parser.error("Missing data tables: " + ", ".join(missing) + ". Run python download_data.py first.")
    workers = min(args.jobs, len(selected))
    reproduction = Reproduction(data, output, args.jobs)
    started = time.monotonic()
    failures, validations = [], {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(area, pool.submit(reproduction.part, area)) for area in selected]
        for area, future in futures:
            try:
                validations[area] = future.result()
                if validations[area]["reference"] == "FAIL":
                    failures.append((area, "reference validation failed"))
            except Exception as error:
                failures.append((area, str(error)))
                validations[area] = {"reference": "not completed", "paper": "not completed", "differences": []}
    reproduction.executor.shutdown(wait=True)
    for area in selected:
        status = validations[area]
        print(f"{area}: reference validation={status['reference']}; paper validation={status['paper']}", flush=True)
        for name, value in status.get("diagnostics", {}).items():
            print(f"  diagnostic {name}: {json.dumps(value)}", flush=True)
        for difference in status["differences"]:
            print(f"  {difference['check']}: {json.dumps(difference, ensure_ascii=False)}", flush=True)
    (output / "validation_summary.json").write_text(json.dumps(validations, indent=2, ensure_ascii=False) + "\n")
    for area, error in failures:
        print(f"FAILED {area}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(1)
    print(f"Completed {len(selected)} parts in {time.monotonic() - started:.1f}s. Outputs: {output / 'figures'}, {output / 'tables'}")


if __name__ == "__main__":
    main()

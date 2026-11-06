"""Refit AUCs and action intervals from numeric run frames."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--processes", type=int, default=min(9, os.cpu_count() or 1))
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    code = Path(__file__).resolve().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLCONFIGDIR=str(output / ".matplotlib"))
    commands = []

    def run(name, script, *arguments):
        started = time.monotonic()
        command = [sys.executable, "-B", str(code / script), *map(str, arguments)]
        with (output / f"{name}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = {"step": name, "elapsed_seconds": time.monotonic() - started, "success": True}
        commands.append(record)
        print(name, "complete", flush=True)

    if not args.summarize_only:
        with ThreadPoolExecutor(max_workers=3) as pool:
            original = pool.submit(run, "auc_original", "auc_original.py", "--source-index", data / "auc_sources.csv",
                                   "--output", output / "auc_original", "--processes", args.processes)
            a = pool.submit(run, "auc_credited", "auc_original.py", "--source-index", data / "credited_sources.csv",
                            "--output", output / "auc_credited", "--processes", 1)
            b = pool.submit(run, "work_changes", "work_changes.py", "--data", data,
                            "--output", output / "expectation", "--processes", args.processes)
            a.result()
            b.result()
            original.result()
    run("format_refits", "format_refits.py", "--output", output)
    run("answer_key_example", "answer_key_example.py", "--data", data, "--output", output)
    run("validation", "validate_refits.py", "--data", data, "--output", output)
    (output / "reproduction.json").write_text(json.dumps(commands, indent=2) + "\n")


if __name__ == "__main__":
    main()

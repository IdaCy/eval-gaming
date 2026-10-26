"""Fetch upstream task records, input files, and repositories at their recorded commits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import urllib.parse
import urllib.request

import pandas as pd

from harness.download_inputs import download


def issue_api(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in {"github.com", "api.github.com"}:
        raise ValueError("Expected a public GitHub issue URL")
    if parsed.netloc == "api.github.com":
        return url
    return "https://api.github.com/repos" + parsed.path


def fetch_repository(row, directory, repositories):
    commit = row.base_commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Repository commit must be a full SHA-1")
    if row.repository in repositories:
        url = "https://api.github.com/repositories/" + str(repositories[row.repository])
        with urllib.request.urlopen(url, timeout=120) as stream:
            clone = json.load(stream)["clone_url"]
    else:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row.repository):
            raise ValueError("Invalid repository identifier")
        clone = "https://github.com/" + row.repository + ".git"
    target = directory / "repository"
    target.mkdir(exist_ok=False)
    subprocess.run(["git", "init", "--quiet", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--quiet", "--depth=1", clone, commit], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--quiet", "--detach", "FETCH_HEAD"], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit:
        raise ValueError("Fetched repository does not match the recorded commit")
    return {"repository": row.repository, "commit": actual}


def fetch_task(row, data, output, with_repository=False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", row.task_id):
        raise ValueError("Invalid task identifier")
    directory = output / row.task_id
    directory.mkdir(parents=True, exist_ok=False)
    result = {"task_id": row.task_id, "source_revision": row.source_revision,
              "base_commit": row.base_commit, "task_text_sha256": row.task_text_sha256, "files": []}
    if row.fetch_kind == "benchmark":
        records = pd.read_parquet(data / "tasks/swe_rebench_v2.parquet")
        selected = records[records.instance_id.str.lower().eq(row.task_id.lower())]
        if len(selected) != 1 or selected.iloc[0].base_commit != row.base_commit:
            raise ValueError("Pinned benchmark task identity differs")
        target = directory / "upstream_task.parquet"
        selected.to_parquet(target, index=False)
        text = selected.iloc[0].problem_statement
        result["upstream_text_matches_recorded_prompt"] = hashlib.sha256(text.encode()).hexdigest() == row.task_text_sha256
        if not result["upstream_text_matches_recorded_prompt"]:
            raise ValueError("Pinned benchmark request hash differs")
        request = directory / "problem_statement.txt"
        request.write_bytes(text.encode("utf-8"))
        result["files"].append({"path": request.name, "bytes": request.stat().st_size,
                                "sha256": hashlib.sha256(request.read_bytes()).hexdigest()})
        result["files"].append({"path": target.name, "bytes": target.stat().st_size,
                                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    elif row.fetch_kind == "github_issue":
        result["files"].append(download(issue_api(row.source_url), directory / "issue.json"))
    elif row.fetch_kind == "data_files":
        files = pd.read_parquet(data / "tasks/downloads.parquet")
        for item in files[files.task_id.eq(row.task_id)].itertuples(index=False):
            relative = Path(item.workspace_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid input filename")
            result["files"].append(download(item.url, directory / "inputs" / relative, item.sha256,
                                            int(item.bytes) if pd.notna(item.bytes) else None))
    else:
        raise ValueError("Unknown task source type")
    if with_repository and row.fetch_kind != "data_files":
        repositories = pd.read_csv(data / "tasks/repositories.csv").set_index("repository").github_id.to_dict()
        result["repository"] = fetch_repository(row, directory, repositories)
    (directory / "retrieval.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data")
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repositories", action="store_true")
    parser.add_argument("--jobs", type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    refs = pd.read_parquet(args.data / "tasks/text_sources.parquet")
    selected = refs[refs.task_id.isin(args.task)]
    missing = set(args.task) - set(selected.task_id)
    if missing:
        parser.error("No fetch record for: " + ", ".join(sorted(missing)))
    if not args.execute:
        print(selected.to_json(orient="records", indent=2))
        return
    results, failures = [], []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(selected))) as pool:
        futures = {pool.submit(fetch_task, row, args.data, args.output, args.repositories): row.task_id
                   for row in selected.itertuples(index=False)}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append({"task_id": futures[future], "error": str(error)})
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"completed": results, "failures": failures}
    (args.output / "retrieval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)

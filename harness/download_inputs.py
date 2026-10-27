from harness.data import input_path
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import pandas as pd


def download(url, target, expected=None, expected_bytes=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)
            digest.update(block)
            total += len(block)
    if expected and digest.hexdigest() != expected:
        temporary.unlink()
        raise ValueError("download_sha256_mismatch")
    if expected_bytes is not None and total != expected_bytes:
        temporary.unlink()
        raise ValueError("download_size_mismatch")
    temporary.replace(target)
    return {"url": url, "sha256": digest.hexdigest(), "bytes": total}


def main():
    parser = argparse.ArgumentParser(description="Download licensed source inputs and pinned upstream repositories.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--task", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repositories", action="store_true")
    args = parser.parse_args()
    tasks = pd.read_parquet(input_path(args.data, "tasks.parquet"))
    if args.task:
        tasks = tasks[tasks.task_id.isin(args.task)]
    downloads = pd.read_parquet(input_path(args.data, "downloads.parquet"))
    downloads = downloads[downloads.task_id.isin(tasks.task_id)]
    records = []
    for row in downloads.itertuples(index=False):
        relative = Path(row.workspace_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid_workspace_download_path")
        target = args.output / row.task_id / "seed" / relative
        info = {"task": row.task_id, "url": row.url, "target": str(target), "licence": row.licence}
        if args.execute:
            info.update(download(row.url, target, row.sha256, int(row.bytes) if pd.notna(row.bytes) else None))
        records.append(info)
    if args.repositories:
        identities = pd.read_csv(args.data / "tasks" / "repositories.csv").set_index("repository").github_id.to_dict()
        for row in tasks[tasks.dataset.ne("data_analysis")].itertuples(index=False):
            source = json.loads(row.source)
            repository = row.repository
            target = args.output / row.task_id / "seed"
            repository_api = ("https://api.github.com/repositories/" + str(identities[repository])
                              if repository in identities else "https://api.github.com/repos/" + repository)
            url = "https://github.com/" + repository + ".git" if "/" in repository else repository_api
            if args.execute:
                if repository in identities:
                    request = urllib.request.Request(repository_api, headers={"Accept": "application/vnd.github+json"})
                    with urllib.request.urlopen(request, timeout=120) as response:
                        url = json.load(response)["clone_url"]
                target.mkdir(parents=True, exist_ok=False)
                subprocess.run(["git", "init", str(target)], check=True)
                subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", url, row.base_commit], check=True)
                subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
                if row.dataset == "recent_issues":
                    issue = source["url"].rstrip("/").split("/")[-1]
                    download(repository_api + "/issues/" + issue,
                             args.output / row.task_id / "issue.json")
            records.append({"task": row.task_id, "url": url, "commit": row.base_commit,
                            "licence": row.licence, "target": str(target)})
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

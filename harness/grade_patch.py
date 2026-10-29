from harness.data import input_path
import argparse
import base64
import json
from pathlib import Path

import pandas as pd

from harness.check import evaluate


def main():
    parser = argparse.ArgumentParser(description="Grade a patch in a separate container with hidden tests.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--task", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pd.read_parquet(input_path(args.data, "grader_packages.parquet"))
    selected = rows[rows.task_id.eq(args.task)]
    if len(selected) != 1:
        raise ValueError("task_grader_package_unavailable")
    package = json.loads(selected.iloc[0].package)
    if "data_config" in package:
        scorer = Path(__file__).with_name("data_scorer.py").read_bytes()
        config = json.dumps(package.pop("data_config"), ensure_ascii=False).encode()
        package["eval_script"] = (
            "#!/bin/bash\nset -uo pipefail\ncd /workspace/project\n"
            "base64 -d > /tmp/data_scorer.py <<'SCORER'\n" + base64.b64encode(scorer).decode() + "\nSCORER\n"
            "base64 -d > /tmp/data_config.json <<'CONFIG'\n" + base64.b64encode(config).decode() + "\nCONFIG\n"
            "python3 -I /tmp/data_scorer.py < /tmp/data_config.json\n")
    result = evaluate(package, args.image, args.patch.read_bytes(), args.output)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "task": args.task}))


if __name__ == "__main__":
    main()

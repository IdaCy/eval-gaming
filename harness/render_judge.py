from harness.data import input_path
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Write the paper's judge prompt without making an API request.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--kind", choices=["with_reasoning", "actions_only", "pair"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = pd.read_csv(input_path(args.data, "prompts.csv")).set_index("name").text.to_dict()
    text = prompts["judge_" + args.kind]
    if args.kind != "pair":
        text = prompts["judge_frame"].rstrip("\n") + "\n\n" + text
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"role": "system", "content": text}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "api_calls": 0}))


if __name__ == "__main__":
    main()

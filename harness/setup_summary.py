from harness.data import input_path
import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from harness.actor import ActorDeadlineExceeded, effective_specification, run_actor
from harness.configuration import TOOLS, actor_config
from harness.environment import environment_binding
from harness.runtime import TOOL_SECONDS
from harness.run_task import contexts, load_task


class MemoryRuntime:
    def __init__(self, mode):
        self.mode = mode
        self.saved = {}
        self.calls = 0
        self.closed = False
        self.claimed = False

    def claim_actor(self):
        if self.claimed:
            raise FileExistsError('actor destination already claimed')
        self.claimed = True

    def save(self, name, value):
        self.saved[name] = copy.deepcopy(value)

    def start(self, spec, workspace, files):
        self.effective = spec
        self.workspace = workspace
        self.files = files

    def release_actor(self, digest):
        self.digest = digest

    def before_turn(self, turn):
        if self.mode == "deadline":
            raise ActorDeadlineExceeded("episode_deadline")

    def exchange(self, turn, payload):
        self.calls += 1
        if self.mode == "natural":
            message = {"role": "assistant", "content": "Done."}
            finish = "stop"
        else:
            count = 17 if self.mode == "tool_batch_limit" else 1
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{turn}_{i}", "type": "function", "function": {
                    "name": "bash", "arguments": '{"command":"true"}'}}
                for i in range(count)]}
            finish = "tool_calls"
        return {"choices": [{"message": message, "finish_reason": finish}]}

    def execute_tool(self, turn, index, call, limit):
        return {"role": "tool", "tool_call_id": call["id"], "content": "exit status: 0"}

    def checkpoint(self, name):
        return {"complete": True, "archives": [], "errors": []}

    def elapsed_seconds(self):
        return 0.0

    def terminal(self, status):
        self.status = status

    def close(self):
        self.closed = True
        return {"state": "closed"}

    def failure(self, stage, error):
        self.saved["failure"] = {"stage": stage, "error": error}


class MemoryProtocol:
    name = "offline"

    def request(self, config, messages, seed, tools):
        return {"messages": messages, "tools": tools, "seed": seed}

    def check_response(self, response, messages):
        return response

    def replay(self, message):
        return message


def reproduce(data):
    profile = SimpleNamespace(model="offline", family="chat", max_output_tokens=1024,
                              record=lambda: {"id": "offline"})
    config = actor_config(profile)
    prompts = pd.read_csv(input_path(data, "prompts.csv")).set_index("name").text.to_dict()
    tasks = pd.read_parquet(input_path(data, "tasks.parquet"))
    variants = pd.read_parquet(input_path(data, "variants.parquet"))
    checks = {}
    checks["tools_match_paper"] = TOOLS == json.loads(prompts["tools"])
    checks["software_system_prompt_matches"] = config["system"] == prompts["system_prompt"].rstrip("\n")
    bindings = sum(environment_binding(json.loads(r.spec)) is not None for r in tasks.itertuples())
    checks["unique_task_family_setup"] = not variants.duplicated(["task_id", "family", "setup"]).any()
    task = "apache__lucene-11760"
    spec, baseline = load_task(data, task, "main60", "B00L00")
    initial, beliefs = contexts(data, spec, baseline, "B00L00")
    checks["example_request_lines_match"] = baseline["request"].splitlines() == prompts["request_nocue"].splitlines()
    checks["four_independent_belief_contexts"] = len(beliefs) == 4 and all(
        len(value) == len(initial) and value[:-1] == initial[:-1] and
        value[-1]["content"].startswith(initial[-1]["content"] + "\n\n")
        for value in beliefs.values())
    _, name = load_task(data, task, "main60", "F11L02")
    checks["benchmark_heading_present"] = name["request"].startswith(
        "SWE-bench Multilingual repository task\n\n## Problem statement\n\n")
    files = {}
    for setup in ["F09L01", "F09L03", "F09L04", "F09L05", "F10L05", "F11L04", "F11L05"]:
        _, variant = load_task(data, task, "main60", setup)
        files[setup] = {"workspace": variant.get("workspace", spec["workspace"]),
                        "files": [{"path": x["path"], "mode": x.get("mode", 420)} for x in variant.get("files", [])],
                        "environment": variant.get("env", [])}
        if setup in {"F09L03", "F09L04"}:
            effective = effective_specification(spec, variant)
            checks[setup + "_shell_banner"] = "BASH_ENV=/opt/workspace-shell-env" in effective["actor_env"] and any(
                x["path"] == "/opt/workspace-shell-env" and x.get("content") for x in variant["files"])
    checks["five_work_note_files"] = len(files["F10L05"]["files"]) == 5
    checks["two_grading_files"] = len(files["F09L05"]["files"]) == 2
    checks["unreadable_evaluator_file"] = files["F09L01"]["files"][0]["mode"] == 0
    da = tasks[tasks.dataset.eq("data_analysis")].iloc[0]
    dspec = json.loads(da.spec)
    checks["data_system_prompt_matches"] = dspec["system_prompt"] == prompts["system_prompt_data"].rstrip("\n")
    answer_key_variants = 0
    for row in variants[variants.dataset.eq("data_analysis")].itertuples():
        value = json.loads(row.value)
        if any(x.get("path") == "grading/reference/expected_answers.csv" for x in value.get("files", [])):
            answer_key_variants += 1
    checks["data_answer_key_placement"] = answer_key_variants > 0
    lower_case = []
    for row in variants[variants.task_id.eq(task) & variants.family.eq("main60") & variants.setup.str.startswith("F02")].itertuples():
        value = json.loads(row.value)
        lower_case.append({"setup": row.setup, "eligible": bool(row.eligible),
                           "edits": len(value.get("positions", [])),
                           "all_lower_case": value.get("request") == baseline["request"].lower()})
    results = {}
    for mode, status, calls in [("natural", "natural_completion", 1),
                                ("turn_limit", "administratively_censored_turns", 128),
                                ("deadline", "administratively_censored_runtime", 0),
                                ("tool_batch_limit", "technical_failure", 1)]:
        runtime = MemoryRuntime(mode)
        result = run_actor(spec, baseline, config, {"run_id": mode, "seed": 1, "stage": "full",
                           "continue_to_end": True}, runtime, protocol=MemoryProtocol())
        checks[mode] = result["end_status"] == status and result["calls"] == calls and runtime.closed
        results[mode] = {"status": result["end_status"], "calls": result["calls"], "tool_results": result["tool_results"]}
    prompt_hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in prompts.items()}
    return {"checks": {k: bool(v) for k, v in checks.items()}, "mock_execution": results,
            "limits": {"turns": config["max_actor_turns"], "seconds": config["max_episode_seconds"],
                       "tool_output_characters": config["tool_output_characters"], "tool_timeout_seconds": TOOL_SECONDS},
            "environment_bindings": bindings, "example_setup_files": files,
            "answer_key_variants": answer_key_variants, "prompt_sha256": prompt_hashes,
            "example_lower_case_levels": lower_case,
            "example_request_bytes_match": baseline["request"].encode() == prompts["request_nocue"].encode(),
            "cue_texts": pd.read_csv(input_path(data, "cue_texts.csv"), keep_default_na=False).to_dict("records"),
            "forward_cues": pd.read_csv(input_path(data, "forward_cues.csv"), keep_default_na=False).to_dict("records"),
            "api_calls": 0, "container_runs": 0}


def main():
    parser = argparse.ArgumentParser(description="Verify frozen prompts, cue carriers and run limits without API calls.")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    failed = [name for name, passed in result["checks"].items() if not passed]
    print(json.dumps({"checks": len(result["checks"]), "failed": failed, "api_calls": 0}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

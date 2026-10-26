"""Locate inputs within the downloaded dataset."""
from pathlib import Path

TASK_FILES = {'grading_inputs.parquet', 'tasks.parquet', 'licences.csv', 'grader_packages.parquet', 'downloads.parquet'}
SETUP_FILES = {'tokenizer.parquet', 'variants.parquet', 'model_profiles.csv', 'agents.csv'}
PROMPT_FILES = {'forward_cues.csv', 'cue_texts.csv', 'prompts.csv'}


def input_path(root, name):
    directory = ('tasks' if name in TASK_FILES else 'setups' if name in SETUP_FILES
                 else 'prompts' if name in PROMPT_FILES else 'tables/setup')
    return Path(root) / directory / name

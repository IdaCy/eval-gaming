"""Extract numeric action features from explicitly indexed run artifacts."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, json
from pathlib import Path
import pandas as pd
from measures.actions import extract, pass_outcomes, frame_values, decode_context, decode_patch
from measures.data_reports import measure as report_measure


def read_optional(folder, name):
    path = folder / name
    return path.read_bytes() if path.is_file() else None


def measure_run(patch, context, meta, check, model, cause, exclude_pass=False):
    result = frame_values(extract(patch, context, meta, cause=cause, model=model))
    result.update(frame_values(pass_outcomes(check, patch, cause=cause, excluded=exclude_pass)))
    messages, context_error = decode_context(context)
    patch_text, patch_error = decode_patch(patch)
    if cause in {'natural_completion', 'turn_cap'} and context_error is None and patch_error is None:
        result.update(report_measure(messages, patch_text))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, default=DATA / 'action_sources.csv')
    p.add_argument('--runs-root', type=Path, default=DATA_ROOT / 'runs')
    p.add_argument('--output', type=Path, default=OUTPUT / 'extracted_actions.parquet')
    args = p.parse_args()
    root = args.runs_root.resolve()
    rows = []
    for row in pd.read_csv(args.index).to_dict('records'):
        folder = (root / row['folder']).resolve()
        if not folder.is_relative_to(root):
            raise ValueError('run folder must be inside runs root')
        values = measure_run(read_optional(folder, 'candidate.patch'),
                             read_optional(folder, 'actor_final_context.json') or read_optional(folder, 'end_context.json'),
                             read_optional(folder, 'meta.json'),
                             read_optional(folder, 'task_check.json'),
                             row['model'], row['completion_cause'], bool(row.get('exclude_pass', False)))
        rows.append(dict(task=row['task'], code=row['code'], model=row['model'], **values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)


if __name__ == '__main__':
    main()

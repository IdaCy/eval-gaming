"""Count retrieval and incorporation in the selected manual sample."""
from common import cli, load


def calculate(data):
    frame = load(data, "internet_triage.parquet")
    requested = frame[frame.github_contacts.gt(0)]
    return dict(sample_runs=len(frame), request_runs=len(requested), fix_seen=int(requested.fix_seen.sum()),
                incorporated=int(requested.incorporated.sum()),
                late_fetch_indices=requested.loc[~requested.incorporated, "idx"].tolist(),
                selected_sample=True)


if __name__ == "__main__":
    cli(calculate, "internet_triage")

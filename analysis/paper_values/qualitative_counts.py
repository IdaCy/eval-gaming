"""Count regex hits and the bounded manual screening sample."""
from common import cli, load


def calculate(data):
    regex = load(data, "grader_regex_counts.parquet")
    candidates = load(data, "visibility_candidates.parquet")
    scope = load(data, "visibility_scope.csv").iloc[0]
    if not candidates.run_id.is_unique:
        raise ValueError("Duplicate screening candidate")
    return dict(grader_address_screen=dict(runs=len(regex), command_regex_positive_runs=int(regex.p1.gt(0).sum()),
                                          final_message_regex_positive_runs=int(regex.p2.gt(0).sum()),
                                          not_a_semantic_exhaustive_search=True),
                reasoning_visibility_screen=dict(first_retrieval=int(candidates.first.sum()),
                                                 second_retrieval=int(candidates.second.sum()),
                                                 overlap=int((candidates.first & candidates.second).sum()),
                                                 union_candidates=len(candidates), scanned_runs=int(scope.scanned_runs),
                                                 models=int(scope.models), reported_positive_candidates=int(scope.reported_positive_candidates),
                                                 zero_is_retrieval_limited_and_not_newly_adjudicated=True))


if __name__ == "__main__":
    cli(calculate, "qualitative_counts")

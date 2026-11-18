"""Verify rating coverage and the arithmetic mean of two task raters."""
from common import cli, load


def calculate(data):
    a, b, average = [load(data, "task_ratings_" + name + ".parquet").set_index(["corpus", "task_id"]).sort_index()
                     for name in ["r1", "r2", "avg"]]
    items = [column for column in a if column.startswith("llm2_")]
    if not (a.index.is_unique and a.index.equals(b.index) and a.index.equals(average.index)):
        raise ValueError("Task rating indices differ")
    error = ((a[items] + b[items]) / 2 - average[items]).abs()
    result = dict(tasks_per_rater=len(a), items=len(items), cells=len(a) * len(items),
                  rater_1=a.rater.value_counts().to_dict(), rater_2=b.rater.value_counts().to_dict(),
                  different_rater_cells=int((a[items] != b[items]).sum().sum()),
                  max_absolute_mean_error=float(error.max().max()),
                  missing_rating_cells=int(a[items].isna().sum().sum() + b[items].isna().sum().sum()),
                  items_list=items)
    if result["max_absolute_mean_error"] != 0 or result["missing_rating_cells"] != 0:
        raise ValueError("Incomplete or inconsistent task ratings")
    return result


if __name__ == "__main__":
    cli(calculate, "task_ratings")

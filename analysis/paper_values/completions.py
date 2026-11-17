"""Count allocated and usable completions separately for each experimental block."""
from common import cli, load


def calculate(data):
    frame = load(data, "completion_slots.parquet")
    good = frame[frame.status.eq("natural_completion") & frame.omission.notna()]
    return dict(allocated_by_block=frame.groupby("block").size().to_dict(),
                natural_usable_by_block=good.groupby("block").size().to_dict(),
                zg1_by_condition=good[good.block.eq("zg1")].groupby("variant_id").size().to_dict(),
                all_slot_states=frame.groupby(["block", "status"], dropna=False).size().reset_index(name="n").fillna("missing").to_dict("records"))


if __name__ == "__main__":
    cli(calculate, "completions")

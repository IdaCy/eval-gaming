"""Download the public inputs for the paper analyses."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

DATASET = "idacy/eval-gaming"
SMALL_DIRECTORIES = ("tables", "tasks", "setups", "prompts", "samples")
BULK_DIRECTORIES = ("runs", "labels")


def download(destination, revision="main", include_bulk=False):
    destination = Path(destination)
    directories = SMALL_DIRECTORIES + (BULK_DIRECTORIES if include_bulk else ())
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        token=False,
        allow_patterns=[f"{name}/**" for name in directories] + ["LICENSE", "LICENSES.md", "README.md", "licenses/**"],
    )
    for name in BULK_DIRECTORIES:
        (destination / name).mkdir(parents=True, exist_ok=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--include-bulk", action="store_true", help="Also download runs/ and labels/ when available.")
    args = parser.parse_args()
    print(download(args.data, args.revision, args.include_bulk))


if __name__ == "__main__":
    main()

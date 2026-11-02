"""Agreement between structured human, reasoning and action labels."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse
import json
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from agreement_stats import kappa_boot


def main(data, output):
    rows = []
    human = pd.read_csv(data / "human_labels.csv")
    for label, column, seed in [("human_reasoning", "mR_mat", 1210), ("human_actions", "mB_mat", 1211)]:
        row = kappa_boot(human.h_mat, human[column], human.task_id, seed=seed)
        rows.append(dict(comparison=label, **row))
    careful = pd.read_parquet(data / "careful_labels.parquet")
    for label, subset, seed in [("careful_all", careful, 20261119)] + [("careful_" + name, group, 20261129) for name, group in careful.groupby("model")]:
        rows.append(dict(comparison=label, **kappa_boot(subset.c_mat, subset.mR_mat, subset.task_id, seed=seed)))
    mask = pd.read_parquet(data / "mask_agreement.parquet")
    rows.append(dict(comparison="masked_unmasked", **kappa_boot(mask.unmasked, mask.masked, mask.task_id, seed=20261139)))
    ranks = pd.read_csv(data / "measure_rank_agreement.csv")
    result = {"agreement": rows, "human_runs": len(human), "human_material": int(human.h_mat.sum()),
              "human_positive_identified_reasoning": int(((human.h_mat == 1) & (human.mR_mat == 1)).sum()),
              "human_positive_identified_actions": int(((human.h_mat == 1) & (human.mB_mat == 1)).sum()),
              "rank_cues": len(ranks), "rank_auc": float(spearmanr(ranks.auc, ranks.d_J)[0]),
              "rank_auc_without_tests": float(spearmanr(ranks.auc_nontest, ranks.d_J)[0])}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(pd.DataFrame(rows)[["comparison", "n", "kappa", "lo", "hi"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT / 'agreement.json')
    args = parser.parse_args()
    main(args.data, args.output)

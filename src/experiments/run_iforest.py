"""Run Isolation Forest baseline across all SMD machines and seeds.

Usage:
    uv run python -m src.experiments.run_iforest --machine all --seeds 0 1 2
    uv run python -m src.experiments.run_iforest --machine machine-1-1 --no-wandb
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.smd import (
    SMD_ROOT,
    _load_labels,
    _load_series,
    compute_scaler,
    list_machines,
)
from src.evaluation.metrics import evaluate
from src.models.iforest import IsolationForestScorer

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def run_one(machine: str, seed: int, root: Path = SMD_ROOT) -> dict:
    train_full = _load_series(root / "train" / f"{machine}.txt")
    test = _load_series(root / "test" / f"{machine}.txt")
    test_labels = _load_labels(root / "test_label" / f"{machine}.txt")

    n_val = max(1, int(len(train_full) * 0.2))
    train = train_full[:-n_val]

    mean, std = compute_scaler(train)
    train_s = (train - mean) / std
    test_s = (test - mean) / std

    model = IsolationForestScorer(random_state=seed)
    model.fit(train_s)
    scores = model.score(test_s)

    metrics = evaluate(scores, test_labels)
    metrics["machine"] = machine
    metrics["seed"] = seed
    metrics["model"] = "isolation_forest"
    metrics["n_train"] = int(len(train))
    metrics["n_test"] = int(len(test))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output", type=str, default="results/iforest.csv")
    args = parser.parse_args()

    machines = list_machines() if args.machine == "all" else [args.machine]
    print(f"Running IF on {len(machines)} machine(s) × {len(args.seeds)} seed(s) "
          f"= {len(machines) * len(args.seeds)} runs")

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(
            project="tsfm-anomaly-bench",
            name=f"iforest_{args.machine}",
            config=vars(args),
        )

    all_results: list[dict] = []
    for machine in machines:
        for seed in args.seeds:
            r = run_one(machine, seed)
            all_results.append(r)
            print(
                f"  {machine} seed={seed}: "
                f"F1={r['best_f1']:.3f}  PA-F1={r['pa_best_f1']:.3f}  "
                f"AUC-PR={r['auc_pr']:.3f}"
            )
            if use_wandb:
                wandb.log(
                    {f"per_run/{k}": v for k, v in r.items() if isinstance(v, (int, float))}
                )

    # Save CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(all_results[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} rows to {out_path}")

    # Aggregate
    f1s = np.array([r["best_f1"] for r in all_results])
    pa_f1s = np.array([r["pa_best_f1"] for r in all_results])
    aucs = np.array([r["auc_pr"] for r in all_results])

    print("\n=== Aggregate (mean ± std across all runs) ===")
    print(f"  Best F1:    {np.nanmean(f1s):.3f} ± {np.nanstd(f1s):.3f}")
    print(f"  PA-Best F1: {np.nanmean(pa_f1s):.3f} ± {np.nanstd(pa_f1s):.3f}")
    print(f"  AUC-PR:     {np.nanmean(aucs):.3f} ± {np.nanstd(aucs):.3f}")

    if use_wandb:
        wandb.log(
            {
                "agg/best_f1_mean": float(np.nanmean(f1s)),
                "agg/best_f1_std": float(np.nanstd(f1s)),
                "agg/pa_best_f1_mean": float(np.nanmean(pa_f1s)),
                "agg/pa_best_f1_std": float(np.nanstd(pa_f1s)),
                "agg/auc_pr_mean": float(np.nanmean(aucs)),
                "agg/auc_pr_std": float(np.nanstd(aucs)),
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()

"""Small, nested chronological grid search for crossover timing models.

Tuning uses earlier rolls only. Because this grid was designed after looking
at the historical test sample, all findings remain exploratory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.timing_model import BASELINE, FEATURES, evaluate, prepare_decisions, summarize

ALPHAS = (1000.0, 100.0, 10.0, 1.0, 0.1)
FEATURE_SETS = {
    "volume": FEATURES[:2],
    "open_interest": ("next_oi_share", "next_oi_share_change_3d"),
    "full": FEATURES,
}
DECISION_DAYS = (-10, -5)


def candidate_grid() -> dict:
    """15 ridge configurations plus baseline; ties prefer baseline, then order."""
    grid = {BASELINE: ((), None)}
    for name, features in FEATURE_SETS.items():
        for alpha in ALPHAS:
            grid[f"ridge_{name}_a{alpha:g}"] = (features, alpha)
    return grid


def run_grid_search(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune each cutoff separately; compare them on identical held-out rolls.

    Each outer forecast uses at least eight completed training rolls. Inside
    that history, expanding validation starts with six rolls. Imputation and
    scaling are re-fitted for each training split by the shared evaluator.
    The cutoff itself is not selected using the test errors.
    """
    candidates = candidate_grid()
    runs, tuning_rows = {}, []
    for cutoff in DECISION_DAYS:
        predictions, tuning = evaluate(
            prepare_decisions(panel, cutoff), candidates=candidates
        )
        predictions["decision_cutoff"] = cutoff
        tuning["decision_cutoff"] = cutoff
        runs[cutoff] = predictions
        tuning_rows.append(tuning)
    common = set.intersection(*(set(run.roll_id) for run in runs.values()))
    if not common:
        raise ValueError("No common held-out rolls for the two decision cutoffs")
    forecasts, scores = [], []
    for cutoff, run in runs.items():
        paired = run.loc[run.roll_id.isin(common)].copy()
        summary, _ = summarize(paired)
        summary["decision_cutoff"] = cutoff
        summary["excluded_test_rolls"] = run.roll_id.nunique() - len(common)
        forecasts.append(paired)
        scores.append(summary)
    return (pd.concat(forecasts, ignore_index=True),
            pd.concat(scores, ignore_index=True),
            pd.concat(tuning_rows, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/zn_roll_panel.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/issue-20/grid-search"))
    args = parser.parse_args()
    predictions, summary, tuning = run_grid_search(pd.read_parquet(args.panel))
    args.output.mkdir(parents=True, exist_ok=True)
    choices = predictions.loc[predictions.model == "selected"]
    for name, frame in {"predictions": predictions, "summary": summary,
                        "inner_validation": tuning, "selected_models": choices}.items():
        frame.to_csv(args.output / f"{name}.csv", index=False)
    config = {
        "status": "exploratory; grid designed after examining the historical test sample",
        "alphas": ALPHAS, "feature_sets": FEATURE_SETS, "cutoffs": DECISION_DAYS,
        "candidate_order": list(candidate_grid()), "min_train_rolls": 8,
        "min_inner_rolls": 6, "selection_metric": "rounded-day MAE on inner roll holdouts",
        "tie_rule": "baseline first, then grid insertion order",
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    policy = summary.loc[summary.model.isin([BASELINE, "selected"])]
    lines = ["# Exploratory nested grid search", "",
             "Search: 3 feature sets × 5 ridge strengths = 15 learned configurations, "
             "plus the historical-median baseline. Two cutoffs are evaluated separately.", "",
             "Each test roll is held out while the configuration is selected through "
             "expanding validation inside earlier completed rolls. Preprocessing uses "
             "each training split only. Baseline wins validation ties.", "",
             "| Cutoff | Policy | Rolls | MAE (panel days) | Difference from baseline |",
             "|---:|---|---:|---:|---:|"]
    for row in policy.itertuples():
        lines.append(f"| {row.decision_cutoff} | {row.model} | {row.rolls} | {row.mae:.3f} | {row.mae_minus_baseline:+.3f} |")
    lines += ["", "## Interpretation", "",
              "The `selected` rows measure the grid-search policy on held-out rolls. "
              "The individual candidate scores in `summary.csv` are diagnostics. Choosing "
              "the best candidate or cutoff from those final scores reuses the test set.", "",
              "This search was proposed after examining the same historical sample. "
              "Nested validation prevents direct per-fold tuning leakage, but does not "
              "make this an untouched research test set. Confirm any improvement on new "
              "rolls or independently validated additional history. The earliest outer "
              "folds have only two inner validation rolls, so selection is uncertain.", "",
              "Relative days retain the upstream session-label and OI availability "
              "limitations. The -5 cutoff has more recent information and less lead time "
              "than -10. Timing MAE is not a measurement of execution costs.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(policy.to_string(index=False))
    print("\nConfigurations chosen using earlier rolls:")
    print(choices.groupby(["decision_cutoff", "selected_model"]).size().to_string())
    print(f"\nFull grid diagnostics and report saved to {args.output}")


if __name__ == "__main__":
    main()

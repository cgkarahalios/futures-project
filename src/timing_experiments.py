"""Exploratory trajectory forecasts at two fixed decision cutoffs.

Added after seeing the original test results. These experiments are not fresh
confirmation. They leave the original candidate set and default run unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.timing_model import (
    CANDIDATES, evaluate, prepare_decisions, summarize,
)

EXPERIMENTAL_CANDIDATES = {
    **CANDIDATES,
    "trend_shrunk": (("volume_logit_trend_bday",), None),
}
DECISION_DAYS = (-10, -5)


def trajectory_decisions(panel: pd.DataFrame, decision_day: int) -> pd.DataFrame:
    """Extrapolate five past volume shares to 50%, using only as-of rows.

    A straight line in log-odds approximates the middle of an S-shaped migration
    path. Degenerate, flat or decreasing paths produce a missing projection;
    the predictor then falls back to the historical baseline.
    """
    decisions = prepare_decisions(panel, decision_day)
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    projections = []
    for row in decisions.itertuples():
        history = frame.loc[
            (frame["roll_id"] == row.roll_id) & (frame["date"] <= row.decision_date)
        ].sort_values("date").tail(5)
        share = history["next_volume_share"].to_numpy(dtype=float)
        time = history["bdays_to_fnd"].to_numpy(dtype=float)
        projected = np.nan
        if len(share) == 5 and np.isfinite(share).all() and ((share > 0) & (share < 1)).all():
            logits = np.log(share / (1 - share))
            # Center at the decision observation for numerical stability.
            x = np.column_stack([np.ones(5), time - row.decision_bday])
            intercept, slope = np.linalg.lstsq(x, logits, rcond=None)[0]
            if np.isfinite(slope) and slope > 1e-8:
                projected = row.decision_bday - intercept / slope
        projections.append(projected)
    decisions["volume_logit_trend_bday"] = projections
    return decisions


def run_experiments(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the two information cutoffs on their common held-out roll cohort."""
    runs, tuning_rows = {}, []
    for day in DECISION_DAYS:
        predictions, tuning = evaluate(
            trajectory_decisions(panel, day), candidates=EXPERIMENTAL_CANDIDATES
        )
        predictions["decision_cutoff"] = day
        tuning["decision_cutoff"] = day
        runs[day] = predictions
        tuning_rows.append(tuning)
    common = set.intersection(*(set(run["roll_id"]) for run in runs.values()))
    if not common:
        raise ValueError("No common test rolls across decision cutoffs")
    predictions, comparisons = [], []
    for day, run in runs.items():
        paired = run.loc[run["roll_id"].isin(common)].copy()
        summary, _ = summarize(paired)
        summary["decision_cutoff"] = day
        summary["excluded_test_rolls"] = run.roll_id.nunique() - len(common)
        predictions.append(paired)
        comparisons.append(summary)
    return (pd.concat(predictions, ignore_index=True),
            pd.concat(comparisons, ignore_index=True),
            pd.concat(tuning_rows, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/zn_roll_panel.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/issue-20/exploratory"))
    args = parser.parse_args()
    predictions, summary, tuning = run_experiments(pd.read_parquet(args.panel))
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    tuning.to_csv(args.output / "inner_validation.csv", index=False)
    config = {"status": "exploratory; designed after inspecting existing test results",
              "cutoffs": DECISION_DAYS, "lookback_observations": 5,
              "trend_weight": 0.5, "min_train_rolls": 8, "min_inner_rolls": 6,
              "candidates": EXPERIMENTAL_CANDIDATES,
              "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest()}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    lines = ["# Exploratory timing improvements", "",
             "Designed after inspecting the original test sample. These results need "
             "confirmation on new data. The default model comparison is unchanged.", "",
             "The trajectory model fits a line to the log-odds of the five most recent "
             "volume shares, extrapolates to 50%, corrects its typical bias using prior "
             "rolls, and pulls the prediction halfway toward the historical median.", "",
             "Comparisons use identical test rolls at both cutoffs. Later forecasts "
             "have more information and less lead time; they are a different task.", "",
             "| Cutoff | Model | Rolls | MAE (panel days) | MAE minus baseline |",
             "|---:|---|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.decision_cutoff} | {row.model} | {row.rolls} | {row.mae:.3f} | {row.mae_minus_baseline:+.3f} |")
    lines += ["", "Only `selected` measures training-only candidate selection. Do not "
              "choose a model or cutoff by its final test score and call that fresh "
              "out-of-sample evidence. The current session-label and OI publication-time "
              "limitations still apply. Timing accuracy does not establish cheaper execution.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(summary.to_string(index=False))
    print(f"\nExploratory results saved to {args.output}")


if __name__ == "__main__":
    main()

"""Generate and evaluate walk-forward liquidity crossover forecasts.

This module predicts whether the next ZN Treasury futures contract's volume
share will first exceed 50% within a specified number of business days.
The input feature panel is produced by notebook 06 and contains observations
grouped by quarterly roll.

Two logistic regression models are fitted independently: a calendar-only
baseline using business days to First Notice Day (FND), and a model using
calendar and liquidity features. Each test roll is held out completely;
training uses only rolls whose final observation predates the test roll's
first observation. No random day-level train/test split is performed.

Features are assumed to be available after the close of their observation
date. Forecasts must not be used for decisions earlier than the following
trading session. Realized time to crossover is diagnostic information only
and is never an input feature.

The module provides a command-line entry point for saving out-of-sample
probabilities and reporting Brier scores and log losses.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "bdays_to_fnd",
    "next_volume_share",
    "next_volume_share_change_3d",
    "next_volume_share_change_5d",
    "next_oi_share",
    "next_oi_share_change_3d",
    "log_spread_volume",
)


def _classifier():
    """Create the logistic regression pipeline used for both models.

    Missing feature values are replaced with zero before standardization.
    Logistic regression is fitted with a fixed random seed and a higher
    iteration limit to improve convergence consistency across roll windows.

    Returns:
        sklearn.pipeline.Pipeline: An unfitted pipeline containing a
        constant-value imputer, standard scaler, and logistic regression
        classifier.
    """
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=0),
    )


def walk_forward_predictions(
    panel: pd.DataFrame, horizon: int = 3, min_train_rolls: int = 6
) -> pd.DataFrame:
    """Generate out-of-sample crossover probabilities by historical roll.

    Rolls are ordered by their first and last observation dates. For each
    eligible test roll, fit separate calendar-only and liquidity-feature
    logistic regression pipelines on earlier, fully completed rolls.
    Exclude rows with missing target labels from training and evaluation,
    and skip test rolls without enough previous history or without both
    target classes represented in training.

    The `days_until_crossover` column, if present, is never used because it
    reveals the future realized event date.

    Args:
        panel (pd.DataFrame): Daily feature panel from notebook 06. Requires
            `roll_id`, `date`, every column listed in `FEATURES`, and a
            binary or missing `crossover_within_{horizon}d` target. Each
            (`roll_id`, `date`) pair must be unique. The label should indicate
            whether the first volume crossover occurs within the following
            `horizon` business days.
        horizon (int): Forecast horizon in business days. Defaults to 3.
            The matching target column must exist in `panel`.
        min_train_rolls (int): Minimum number of fully completed prior
            rolls required before a held-out roll is considered. Defaults
            to 6.

    Returns:
        pd.DataFrame: Predictions for eligible held-out roll-days, sorted
        by date and roll ID, with columns:

            - `roll_id`: Identifier of the held-out quarterly roll.
            - `date`: Feature observation date.
            - `bdays_to_fnd`: Business days relative to First Notice Day.
            - `horizon`: Forecast horizon in business days.
            - `train_rolls`: Number of completed earlier rolls considered.
            - `actual`: Observed binary target for this horizon.
            - `p_calendar`: Calendar-only estimated event probability.
            - `p_model`: Liquidity-feature estimated event probability.

    Raises:
        ValueError: If required columns are missing, the horizon or minimum
            training-roll count is nonpositive, roll-date observations are
            duplicated, labels are not binary or missing, or no eligible
            test rolls can be evaluated.

    Notes:
        Features are evaluated as of the end of each observation date, so
        the forecast is actionable no earlier than the next session. The
        function does not estimate transaction costs or trading performance.
    """
    target = f"crossover_within_{horizon}d"
    required = {"roll_id", "date", target, *FEATURES}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if horizon < 1 or min_train_rolls < 1:
        raise ValueError("horizon and min_train_rolls must be positive")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["roll_id", "date"]).any():
        raise ValueError("Duplicate roll/date observations")
    if not bool(frame[target].dropna().isin([0, 1]).all()):
        raise ValueError("Labels must be binary (or missing)")
    frame = frame.sort_values(["date", "roll_id"]).reset_index(drop=True)
    windows = frame.groupby("roll_id")["date"].agg(["min", "max"]).sort_values(
        ["min", "max"]
    )
    roll_ids = windows.index.tolist()
    outputs = []

    for i, roll_id in enumerate(roll_ids):
        test = frame.loc[
            (frame["roll_id"] == roll_id) & frame[target].notna()
        ].copy()
        if test.empty:
            continue
        test_start = windows.loc[roll_id, "min"]
        completed = [
            other for other in roll_ids[:i]
            if windows.loc[other, "max"] < test_start
        ]
        if len(completed) < min_train_rolls:
            continue
        train = frame.loc[
            frame["roll_id"].isin(completed) & frame[target].notna()
        ]
        if train[target].nunique() < 2:
            continue

        for fields, output_column in (
            (("bdays_to_fnd",), "p_calendar"),
            (FEATURES, "p_model"),
        ):
            clf = _classifier()
            clf.fit(train[list(fields)].astype(float), train[target].astype(int))
            test[output_column] = clf.predict_proba(test[list(fields)].astype(float))[:, 1]

        test["horizon"] = horizon
        test["train_rolls"] = len(completed)
        outputs.append(
            test[[
                "roll_id", "date", "bdays_to_fnd", "horizon",
                "train_rolls", target, "p_calendar", "p_model",
            ]].rename(columns={target: "actual"})
        )

    if not outputs:
        raise ValueError(
            "No eligible test rolls; check historical coverage and labels."
        )
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["date", "roll_id"]
    ).reset_index(drop=True)


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate calendar and liquidity-feature probability forecasts.

    Compute Brier score and binary log loss separately for each held-out
    roll, then take the unweighted mean across rolls. This prevents rolls
    with more eligible daily observations from dominating the summary.
    Probabilities are clipped to [1e-6, 1 - 1e-6] for numerical stability.

    Args:
        predictions (pd.DataFrame): Output from
            `walk_forward_predictions`. Must include `roll_id`, binary
            `actual`, and the forecast columns `p_calendar` and `p_model`.

    Returns:
        pd.DataFrame: One row per model (`calendar` and `features`) with
        columns `model`, `rolls`, `observations`,
        `brier_roll_mean`, and `log_loss_roll_mean`. Lower scores indicate
        better probabilistic predictions.

    Notes:
        Averaging by roll avoids overweighting longer rolls but does not
        imply that the underlying daily labels are independent. The scores
        do not measure the execution value of a forecast.
    """
    scores = []
    for name, col in (("calendar", "p_calendar"), ("features", "p_model")):
        each_roll = []
        for _, group in predictions.groupby("roll_id"):
            y = group["actual"].astype(int)
            p = group[col].clip(1e-6, 1 - 1e-6)
            each_roll.append((
                brier_score_loss(y, p),
                log_loss(y, p, labels=[0, 1]),
            ))
        values = np.asarray(each_roll, dtype=float)
        scores.append({
            "model": name,
            "rolls": len(values),
            "observations": len(predictions),
            "brier_roll_mean": values[:, 0].mean(),
            "log_loss_roll_mean": values[:, 1].mean(),
        })
    return pd.DataFrame(scores)


def main() -> None:
    """Run the walk-forward evaluation from the command line.

    Read the feature panel, compute out-of-sample probabilities for the
    selected prediction horizon, print roll-averaged Brier and log-loss
    scores, and save the probabilities as CSV.

    Command-line arguments:
        --input: Path to the feature panel Parquet file. Defaults to
            `data/processed/zn_early_indicator_panel.parquet`.
        --horizon: Number of business days (1, 3, or 5). Defaults to 3.
        --min-train-rolls: Minimum completed historical rolls. Defaults to 6.
        --output-dir: Directory for the prediction CSV. Defaults to
            `data/processed`.

    Raises:
        FileNotFoundError: If the input Parquet file cannot be found.
        ValueError: If the feature panel is invalid or no held-out roll
            qualifies for evaluation.

    Notes:
        Output is written to `zn_crossover_predictions_{horizon}d.csv`
        inside the selected directory. The program does not download
        market data; prepare the input with notebook 06 first.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "data/processed/zn_early_indicator_panel.parquet"
    ))
    parser.add_argument("--horizon", type=int, choices=[1, 3, 5], default=3)
    parser.add_argument("--min-train-rolls", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    forecasts = walk_forward_predictions(
        frame, args.horizon, args.min_train_rolls
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"zn_crossover_predictions_{args.horizon}d.csv"
    forecasts.to_csv(output, index=False)
    print(score_predictions(forecasts).to_string(index=False))
    print(f"Saved {len(forecasts)} out-of-sample predictions to {output}")


if __name__ == "__main__":
    main()

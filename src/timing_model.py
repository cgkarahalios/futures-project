"""Compare crossover timing models using nested, chronological roll holdouts.

Run ``python -m src.timing_model`` on the existing roll panel. No network or
Databento credentials are used. Relative days are the upstream panel's observed
date positions, not verified exchange sessions; see the generated date audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import __version__ as sklearn_version

BASELINE = "historical_median"
FEATURES = (
    "next_volume_share", "next_volume_share_change_3d",
    "next_volume_share_change_5d", "next_oi_share",
    "next_oi_share_change_3d", "log_spread_volume",
)
# Fixed before evaluating held-out results. Baseline wins validation ties.
CANDIDATES = {
    BASELINE: ((), None),
    "ridge_reduced_10": (FEATURES[:2], 10.0),
    "ridge_full_100": (FEATURES, 100.0),
    "ridge_full_10": (FEATURES, 10.0),
}
TARGET = "actual_crossover_bday"


def prepare_decisions(panel: pd.DataFrame, decision_day: int = -10,
                      legacy_snapshot: bool = False) -> pd.DataFrame:
    """Make one snapshot per roll, using no row later than the decision cutoff.

    Feature differences use only previous observations within that roll. A
    missing value on the selected row stays missing (no groupby.first backfill).
    Outcomes are attached separately and never enter the predictor matrix.
    ``legacy_snapshot`` deliberately reproduces notebook 08's column-wise
    groupby.first behavior for diagnosis only, never for candidate selection.
    """
    required = {"roll_id", "date", "bdays_to_fnd", "next_volume_share",
                "next_oi_share", "spread_volume"}
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if decision_day >= 0:
        raise ValueError("decision_day must be negative")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame[list(required - {"next_volume_share", "next_oi_share"})].isna().any().any():
        raise ValueError("Missing roll identifiers, dates, offsets, or spread volume")
    for key in ("date", "bdays_to_fnd"):
        if frame.duplicated(["roll_id", key]).any():
            raise ValueError(f"Duplicate roll/{key} observations")
    frame = frame.sort_values(["roll_id", "date"])
    if not frame.groupby("roll_id")["bdays_to_fnd"].apply(
        lambda x: x.is_monotonic_increasing
    ).all():
        raise ValueError("Relative-day offsets must increase with date")
    offsets = frame["bdays_to_fnd"].to_numpy(dtype=float)
    if not np.isfinite(offsets).all() or not np.equal(offsets, np.rint(offsets)).all():
        raise ValueError("Relative-day offsets must be finite integers")
    for field, lags in (("next_volume_share", (3, 5)), ("next_oi_share", (3,))):
        for lag in lags:
            frame[f"{field}_change_{lag}d"] = frame.groupby("roll_id")[field].diff(lag)
    frame["log_spread_volume"] = np.log1p(frame["spread_volume"].clip(lower=0))

    windows = frame.groupby("roll_id")["date"].agg(
        roll_start="min", roll_end="max"
    )
    events = frame.loc[frame["next_volume_share"] > 0.5].groupby("roll_id").head(1)
    events = events[["roll_id", "date", "bdays_to_fnd"]].rename(columns={
        "date": "actual_crossover_date", "bdays_to_fnd": TARGET,
    })
    # Only the earlier side of a missing cutoff is eligible.
    if legacy_snapshot:
        snapshots = frame.assign(distance=(frame["bdays_to_fnd"] - decision_day).abs())
        snapshots = snapshots.sort_values(["roll_id", "distance", "date"]).groupby(
            "roll_id", as_index=False
        ).first()
    else:
        snapshots = frame.loc[frame["bdays_to_fnd"] <= decision_day].groupby("roll_id").tail(1)
    snapshots = snapshots.rename(columns={"date": "decision_date", "bdays_to_fnd": "decision_bday"})
    rows = snapshots[["roll_id", "decision_date", "decision_bday", *FEATURES]]
    rows = rows.merge(windows, on="roll_id", validate="one_to_one")
    rows = rows.merge(events, on="roll_id", how="left", validate="one_to_one")
    return rows.sort_values(["decision_date", "roll_id"]).reset_index(drop=True)


def date_audit(panel: pd.DataFrame, rolls: pd.DataFrame | None = None) -> pd.DataFrame:
    """Expose upstream timestamp/calendar ambiguities without guessing corrections."""
    rows = []
    for roll_id, group in panel.groupby("roll_id", sort=True):
        group = group.sort_values("date")
        dates = pd.to_datetime(group["date"])
        zero = dates.loc[group["bdays_to_fnd"] == 0]
        row = {
            "roll_id": roll_id, "observations": len(group),
            "sunday_rows": int((dates.dt.dayofweek == 6).sum()),
            "saturday_rows": int((dates.dt.dayofweek == 5).sum()),
            "offset_gaps": int((group["bdays_to_fnd"].diff().dropna() != 1).sum()),
            "offset_zero_date": zero.iloc[0] if len(zero) else pd.NaT,
        }
        if rolls is not None:
            mapping = rolls.loc[(rolls["front"] + "->" + rolls["next"]) == roll_id]
            if len(mapping) != 1:
                raise ValueError(f"Expected one FND mapping for {roll_id}")
            fnd = pd.Timestamp(mapping.iloc[0]["fnd"])
            row["supplied_fnd"] = fnd
            row["offset_zero_matches_supplied_fnd"] = bool(len(zero) and zero.iloc[0] == fnd)
        rows.append(row)
    return pd.DataFrame(rows)


def _training_rows(decisions: pd.DataFrame, test: pd.Series) -> pd.DataFrame:
    # Even overlapping windows must be fully completed before the forecast.
    return decisions.loc[
        (decisions["roll_end"] < test["decision_date"])
        & decisions[TARGET].notna()
        & (decisions["actual_crossover_date"] > decisions["decision_date"])
    ]


def _raw_prediction(train: pd.DataFrame, test: pd.Series, model: str,
                    candidates: dict | None = None) -> float:
    candidates = CANDIDATES if candidates is None else candidates
    y = train[TARGET].to_numpy(dtype=float)
    if model == BASELINE:
        return float(np.median(y))
    if model == "trend_shrunk":
        # Experimental trajectory estimate: correct systematic bias using only
        # historical rolls, then shrink halfway toward their median crossover.
        projection = train["volume_logit_trend_bday"].replace([np.inf, -np.inf], np.nan)
        valid = projection.notna()
        value = test["volume_logit_trend_bday"]
        if valid.sum() < 5 or not np.isfinite(value):
            return float(np.median(y))
        bias = (train.loc[valid, TARGET] - projection.loc[valid]).median()
        return float(0.5 * np.median(y) + 0.5 * (value + bias))
    features, alpha = (FEATURES, None) if model == "ols_asof" else candidates[model]
    xs, xt = [], []
    for feature in features:
        series = pd.to_numeric(train[feature], errors="raise").replace([np.inf, -np.inf], np.nan)
        # Match notebook 08's historical eligibility rule for OLS reproduction.
        if series.notna().sum() < 5:
            continue
        median = series.median()
        filled = series.fillna(median)
        sd = filled.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            continue
        mean = filled.mean()
        value = float(test[feature])
        value = value if np.isfinite(value) else median
        xs.append(((filled - mean) / sd).to_numpy())
        xt.append((value - mean) / sd)
    if not xs:
        return float(np.median(y))
    x = np.column_stack(xs)
    if model == "ols_asof":
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        return float(np.array([1.0, *xt]) @ beta)
    return float(Ridge(alpha=alpha).fit(x, y).predict(np.array([xt]))[0])


def _forecast(train: pd.DataFrame, test: pd.Series, model: str,
              candidates: dict | None = None) -> tuple[float, int]:
    raw = _raw_prediction(train, test, model, candidates)
    # Features summarize this observation's entire day. The next observed row
    # is the earliest actionable choice, never the same day.
    rounded = int(np.rint(np.clip(raw, test["decision_bday"] + 1, 0)))
    return raw, rounded


def select_model(train: pd.DataFrame, min_inner_rolls: int = 6,
                 candidates: dict | None = None) -> tuple[str, dict, int]:
    """Select a candidate with expanding validation entirely inside outer training."""
    candidates = CANDIDATES if candidates is None else candidates
    errors = {name: [] for name in candidates}
    for _, validation in train.iterrows():
        history = _training_rows(train, validation)
        if len(history) < min_inner_rolls:
            continue
        for name in candidates:
            _, pred = _forecast(history, validation, name, candidates)
            errors[name].append(abs(pred - validation[TARGET]))
    scores = {name: float(np.mean(values)) if values else None for name, values in errors.items()}
    count = len(errors[BASELINE])
    selected = min(candidates, key=lambda name: scores[name]) if count else BASELINE
    return selected, scores, count


def evaluate(decisions: pd.DataFrame, min_train_rolls: int = 8,
             min_inner_rolls: int = 6,
             candidates: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return forecasts and inner-validation scores, one observation per test roll."""
    if min_train_rolls < 5 or min_inner_rolls < 5:
        raise ValueError("Require at least five training rolls")
    candidates = CANDIDATES if candidates is None else candidates
    if BASELINE not in candidates:
        raise ValueError("Candidate comparison must include the historical baseline")
    predictions, tuning = [], []
    for _, test in decisions.iterrows():
        train = _training_rows(decisions, test)
        if len(train) < min_train_rolls:
            continue
        # Missing/unobserved events cannot be scored; they are not invented.
        if pd.isna(test[TARGET]) or test["actual_crossover_date"] <= test["decision_date"]:
            continue
        selected, scores, count = select_model(train, min_inner_rolls, candidates)
        for name, score in scores.items():
            tuning.append({"roll_id": test["roll_id"], "candidate": name,
                           "inner_mae": score, "inner_rolls": count, "selected": name == selected})
        for name in (*candidates, "ols_asof", "selected"):
            fitted = selected if name == "selected" else name
            raw, pred = _forecast(train, test, fitted, candidates)
            predictions.append({
                "roll_id": test["roll_id"], "model": name, "selected_model": fitted,
                "decision_date": test["decision_date"], "decision_bday": test["decision_bday"],
                "training_rolls": len(train), "latest_training_end": train["roll_end"].max(),
                "actual": test[TARGET], "raw_prediction": raw, "prediction": pred,
                "error": pred - test[TARGET], "absolute_error": abs(pred - test[TARGET]),
            })
    if not predictions:
        raise ValueError("No eligible test rolls; check coverage, decision day and training size")
    return pd.DataFrame(predictions), pd.DataFrame(tuning)


def reproduce_legacy(decisions: pd.DataFrame, min_train_rolls: int = 8) -> pd.DataFrame:
    """Reproduce notebook 08 on its original snapshots, never select from these."""
    rows = []
    for _, test in decisions.iterrows():
        train = _training_rows(decisions, test)
        if len(train) < min_train_rolls or pd.isna(test[TARGET]):
            continue
        ols = _raw_prediction(train, test, "ols_asof")
        pred = int(np.rint(np.clip(ols, test["decision_bday"], 0)))
        baseline = int(np.rint(train[TARGET].median()))
        rows.append({"roll_id": test["roll_id"], "actual": test[TARGET],
                     "legacy_ols_prediction": pred, "historical_median_prediction": baseline,
                     "legacy_ols_absolute_error": abs(pred - test[TARGET]),
                     "historical_median_absolute_error": abs(baseline - test[TARGET])})
    return pd.DataFrame(rows)


def summarize(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired errors and leave-one-test-roll-out sensitivity; no model refitting."""
    baseline = predictions.loc[predictions["model"] == BASELINE].set_index("roll_id")["absolute_error"]
    summary, sensitivity = [], []
    for model, group in predictions.groupby("model", sort=False):
        errors = group.set_index("roll_id")["absolute_error"]
        if set(errors.index) != set(baseline.index):
            raise ValueError("Models must be evaluated on identical rolls")
        delta = errors - baseline
        summary.append({"model": model, "rolls": len(errors), "mae": errors.mean(),
                        "worst_error": errors.max(), "mae_minus_baseline": delta.mean(),
                        "wins": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
                        "losses": int((delta > 0).sum())})
        for omitted in errors.index:
            remaining = delta.drop(omitted)
            sensitivity.append({"model": model, "omitted_roll": omitted,
                                "paired_mae_minus_baseline": remaining.mean()})
    return pd.DataFrame(summary), pd.DataFrame(sensitivity)


def write_report(output: Path, predictions: pd.DataFrame, summary: pd.DataFrame,
                 sensitivity: pd.DataFrame, audit: pd.DataFrame) -> None:
    """Write a compact Markdown report and a standalone comparison figure."""
    selected = summary.set_index("model").loc["selected"]
    recommendation = (
        "The nested selection procedure improved on the baseline in this sample; "
        "confirm this on new rolls before adopting it."
        if selected["mae_minus_baseline"] < 0 else
        "Retain the historical-median rule: nested selection did not improve on it."
    )
    lines = ["# Issue #20: Crossover timing comparison", "", recommendation, "",
             "| Model | Test rolls | MAE (panel days) | Worst error | MAE minus baseline |",
             "|---|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.model} | {row.rolls} | {row.mae:.3f} | {row.worst_error:.0f} | {row.mae_minus_baseline:+.3f} |")
    lines += ["", "## Interpretation and limits", "",
              "Only `selected` represents the training-only model selection policy. The fixed "
              "candidates are diagnostics; choosing the best after seeing test scores would "
              "reuse the test set. `ols_asof` uses the original OLS fitting method with "
              "corrected snapshots. `legacy_predictions.csv` separately reproduces the "
              "old notebook's column-wise snapshot selection.", "",
              f"The audit contains {int(audit.sunday_rows.sum())} Sunday-labeled rows. "
              "Offsets count observed upstream dates, not verified exchange business days. "
              "Raw extracts retain dates but discard original event/publication timestamps. "
              "Exchange-session alignment, holidays and OI publication availability therefore "
              "remain unverified. These are provisional comparisons using recorded dates, "
              "not a validated live trading backtest. No dates are silently shifted.", "",
              "Every signal is used no earlier than the next panel observation. Historical "
              "windows must end before the test decision date. Imputation, scaling and inner "
              "model selection use training rolls only. Missing events cannot be scored.", "",
              "Sensitivity results drop one test roll's error at a time without refitting; "
              "they describe dependence on individual outcomes, not confidence intervals. "
              "The sample is small and has already been explored in prior work.", "",
              "This module forecasts a crossover date. The separate `src.crossover_model` "
              "module forecasts 1/3/5-day event probabilities, scored with Brier score and "
              "log loss. Those scores cannot be directly compared with timing MAE. "
              "Neither timing accuracy nor these comparisons establish lower transaction costs.", ""]
    if "offset_zero_matches_supplied_fnd" in audit:
        count = int((~audit["offset_zero_matches_supplied_fnd"]).sum())
        lines += [f"Offset zero differs from the supplied FND for **{count} rolls**. "
                  "See `date_audit.csv` before interpreting these as FND business-day forecasts.", ""]
    lines += ["## Leave-one-roll-out error sensitivity", "",
              "Negative values favor the candidate over the baseline.", ""]
    for model, group in sensitivity.groupby("model", sort=False):
        values = group["paired_mae_minus_baseline"]
        lines.append(f"- {model}: {values.min():+.3f} to {values.max():+.3f} panel days.")
    (output / "report.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = summary["model"].tolist()
    ax.barh(labels, summary["mae"], color=["#4477aa" if x == BASELINE else "#8899aa" for x in labels])
    ax.invert_yaxis()
    ax.set(xlabel="Mean absolute error (upstream panel-day units)",
           title=f"Crossover timing: {predictions.roll_id.nunique()} held-out rolls")
    fig.tight_layout()
    fig.savefig(output / "timing_mae.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=Path("data/processed/zn_roll_panel.parquet"))
    parser.add_argument("--rolls", type=Path, default=Path("data/raw/zn_roll_pairs.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/issue-20"))
    parser.add_argument("--decision-day", type=int, default=-10)
    parser.add_argument("--min-train-rolls", type=int, default=8)
    args = parser.parse_args()
    panel = pd.read_parquet(args.panel)
    decisions = prepare_decisions(panel, args.decision_day)
    legacy = reproduce_legacy(prepare_decisions(panel, args.decision_day, legacy_snapshot=True),
                              args.min_train_rolls)
    audit = date_audit(panel, pd.read_parquet(args.rolls) if args.rolls.exists() else None)
    predictions, tuning = evaluate(decisions, args.min_train_rolls)
    summary, sensitivity = summarize(predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in {"decisions": decisions, "date_audit": audit,
                        "legacy_predictions": legacy,
                        "predictions": predictions, "inner_validation": tuning,
                        "summary": summary, "sensitivity": sensitivity}.items():
        table.to_csv(args.output / f"{name}.csv", index=False)
    config = {"decision_day": args.decision_day, "min_train_rolls": args.min_train_rolls,
              "min_inner_rolls": 6, "candidate_order": list(CANDIDATES),
              "candidates": CANDIDATES,
              "panel": str(args.panel),
              "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
              "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn_version}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_report(args.output, predictions, summary, sensitivity, audit)
    legacy_scores = legacy[["legacy_ols_absolute_error", "historical_median_absolute_error"]].mean()
    with (args.output / "report.md").open("a") as report:
        report.write("\n## Legacy reproduction\n\n" + legacy_scores.to_string() + "\n")
    print("Legacy notebook reproduction (MAE):\n" + legacy_scores.to_string() + "\n")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison, date audit, figure and report to {args.output}")


if __name__ == "__main__":
    main()

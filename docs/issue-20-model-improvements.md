# Issue #20: Improve and validate crossover timing models

The original timing model performed worse than the historical-median crossover
rule. This change reproduces that result, corrects snapshot selection, and
compares a small, predefined set of regularized alternatives. The baseline
remains the preferred forecasting rule on the available data.

## Run the comparison

For Jupyter, open `notebooks/09_improve_crossover_timing.ipynb` and run all
cells. It uses the same implementation and displays the tables, figure,
per-roll predictions, date audit and report inside the notebook. Only the
cached roll panel from notebook 02 is required.

From the repository root, after generating `zn_roll_panel.parquet` with notebook
02:

```bash
pip install -r requirements-model.txt
python -m src.timing_model
python -m unittest discover -s tests -p '*model.py' -v
```

The comparison reads existing local files and makes no network requests. It
does not need notebooks 05–08 to have run and never downloads market data.
Optional flags include `--panel`, `--rolls`, `--output`, `--decision-day` and
`--min-train-rolls`. The default decision cutoff is relative day -10; at least
eight completed earlier rolls are required for each test forecast. Changing
these settings after inspecting test performance is exploratory analysis,
not fresh out-of-sample confirmation.

Generated files go to the ignored `results/issue-20/` directory:

| File | Purpose |
|---|---|
| `legacy_predictions.csv` | Reproduce the original notebook's snapshots and forecasts |
| `decisions.csv` | Corrected decision snapshots and diagnostic outcomes |
| `predictions.csv` | Per-roll predictions and errors for every model |
| `inner_validation.csv` | Training-only candidate scores and selections |
| `summary.csv` | Paired MAE comparisons on identical held-out rolls |
| `sensitivity.csv` | Error differences after omitting each test roll in turn |
| `date_audit.csv` | Weekend labels, offset gaps and supplied FND alignment |
| `config.json` | Evaluation settings and package versions |
| `timing_mae.png`, `report.md` | Comparison figure and readable findings |

## What changed

The old `groupby.first()` snapshot could combine nonmissing values from
different dates. In the local dataset, four decision snapshots have missing
open-interest features that the old code replaced with values from other
dates. The corrected snapshot keeps a single actual row at or before the
cutoff. Missing predictors are filled with medians from historical training
rolls, rather than from later rows of the test roll. Notebook 08 now uses the
same snapshot implementation. Its old outputs were cleared because its cost
tables depend on forecasts that changed.

Standardization, feature eligibility and imputation are fitted using training
rolls only. Training windows must finish before the test decision date.
End-of-day observations cannot select execution on that same panel day.
Unobserved crossover outcomes remain missing and are not scored.

Three ridge candidates are fixed in advance: two volume features with alpha
10, all six liquidity features with alpha 10, and all six with alpha 100.
Ridge limits the size of fitted coefficients, reducing extreme predictions
when there are few training rolls. Corrected unregularized OLS is retained
as a diagnostic rather than a candidate for automatic selection.

For each outer test roll, a second chronological evaluation uses only its
training history to choose among the ridge candidates and the historical
median. Inner validation begins after six training rolls. It selects the
lowest rounded-day MAE; ties favor the historical baseline, then the fixed
candidate order. No outer test outcomes are used to select that roll's model.
The `selected` row is this policy's performance. Picking the best fixed
candidate after seeing the final table would reuse the test set.

## Measured results

Run on October 17, 2026, using the local 837-row panel covering 27 rolls,
with 19 held-out rolls and the default settings:

| Method | MAE in panel-day units | Largest absolute error |
|---|---:|---:|
| Original notebook OLS, legacy snapshots | 1.000 | 7 |
| OLS with corrected snapshots | 0.632 | 3 |
| Ridge, two features, alpha 10 | 0.421 | 2 |
| Ridge, six features, alpha 10 | 0.368 | 2 |
| Ridge, six features, alpha 100 | 0.316 | 1 |
| Historical-median baseline | **0.263** | **1** |
| Training-only model selection | **0.263** | **1** |

The legacy reproduction matches all 19 original notebook model predictions
and all 19 baseline predictions exactly. Snapshot correction and regularization
reduce OLS's instability, but none of the fixed ridge candidates beats the
baseline. The selection policy chooses the historical median for all 19 test
rolls, retaining its accuracy rather than forcing a more complex model.

Dropping one held-out roll's error at a time does not produce a ridge model
that strictly beats the baseline. Strong ridge's MAE disadvantage ranges
from 0.000 to 0.056 panel days; corrected OLS's ranges from 0.222 to 0.444.
These checks omit errors without refitting. They are sensitivity descriptions,
not confidence intervals or proof of future performance.

## Date audit and interpretation

All 27 offset-zero dates match the FND values supplied by the upstream roll
mapping, and there are no gaps in their relative-day counters. However, the
panel contains **140 Sunday-labeled observations**. The counters enumerate
observed raw dates, rather than a verified exchange-session calendar.

Notebook 01 saves dates but discards original event/publication timestamps.
Consequently this audit cannot establish the correct exchange-session label,
holiday treatment, or when open-interest data actually became available.
Matching the supplied FND does not independently validate that supplied
calendar. No dates were automatically shifted or weekend observations dropped.
Until session and publication alignment is verified, these are provisional
comparisons in **panel-day units**, not a validated live trading backtest.

The sample is small and has already informed earlier research choices. New
rolls or additional independently validated history would provide stronger
confirmation than further searching this same test sample.

This module predicts a crossover **date**. Dafu's `src/crossover_model.py`
predicts whether crossover occurs within **1, 3 or 5 days**, using Brier score
and log loss. Those probability scores cannot be directly compared with date
MAE. Neither evaluation establishes that a timing rule reduces transaction
costs; that remains a separate question requiring reliable execution data.

## Recommended next step

Keep the historical-median rule as the default timing forecast. Verify session
labels, the exchange calendar and OI release availability before claiming
tradable improvements. Then evaluate on new rolls or more history, rather
than increasing model complexity to chase this sample's errors.

## Exploratory follow-up: later information and volume trajectories

After inspecting the original evaluation, two focused extensions were tested.
They are available in section 7 of notebook 09, or independently through:

```bash
python -m src.timing_experiments
```

The original comparison and default candidate set are unchanged. The follow-up
compares -10 and -5 panel-day decision cutoffs on their common 19 held-out
rolls. Five-day forecasts have more recent data and give less advance notice;
they should not be presented as solving the identical ten-day-ahead task.

A new trajectory candidate uses the five latest as-of volume shares. It fits
a line to their log-odds, projects when next-contract share reaches 50%,
corrects the typical projection bias using earlier completed rolls, and pulls
the adjusted estimate halfway toward the historical median. The lookback of
five and weight of one-half were fixed for this experiment, not selected by
sweeping test scores. Invalid or decreasing trajectories fall back to the
baseline. The same nested chronological selection can reject this candidate.

| Method | MAE at cutoff -10 | MAE at cutoff -5 |
|---|---:|---:|
| Historical median | 0.263 | 0.263 |
| Strong ridge, alpha 100 | 0.316 | 0.263 |
| Bias-corrected, shrunk trajectory | 1.211 | 0.632 |
| Training-only model selection | 0.263 | 0.263 |

Later information lets strong ridge match the baseline, but does not beat it.
The trajectory candidate adds error and is not recommended. These experiments
provide no evidence to replace the baseline. They were designed after seeing
the original test sample, so even a positive result would need independent
confirmation. Dates and publication availability still require validation.

`src/timing_experiments.py` saves its own predictions, tuning scores, summary,
configuration and report under `results/issue-20/exploratory/`, separate from
the original results. Tests check trajectory recovery on a known logistic
path, as-of feature construction, historical bias adjustment, baseline
fallback, matching test cohorts and protection against future-outcome leakage.

## Exploratory nested grid search

Section 8 of notebook 09 runs `python -m src.timing_grid_search`. The grid is
five ridge strengths (0.1, 1, 10, 100, 1000) crossed with three feature sets:
volume share plus its 3-observation change, OI share plus its 3-observation
change, and all six original liquidity features. This gives 15 learned
configurations, plus the historical-median baseline. The default model
comparison and earlier experimental reports are preserved separately.

At each outer held-out roll, the same chronological inner validation selects
the feature set and strength using earlier rolls only. Preprocessing is
re-fitted inside each split. Ties retain the baseline. The -10 and -5 decision
cutoffs are reported separately on the same 19 test rolls; the search does
not choose a cutoff from its final test performance.

In the measured run, the nested grid selected the historical median for all
19 test rolls at both cutoffs. Its MAE was 0.263 panel days at both cutoffs,
matching the baseline. The search therefore provides no evidence of a better
forecasting policy in this sample.

Results are saved to `results/issue-20/grid-search/`: `summary.csv` contains
diagnostic candidate scores, `predictions.csv` all per-roll forecasts,
`inner_validation.csv` the actual tuning scores, `selected_models.csv` the
chosen configurations, and `config.json` and `report.md` describe the run.
The `selected` rows, rather than the best-looking fixed candidate in hindsight,
measure the grid-search policy's performance.

Because this grid was designed after looking at the historical sample, its
results remain exploratory despite nested validation. The earliest outer
folds have only two inner validation rolls. Broader searching increases the
chance of finding accidental historical patterns; independently validated
additional history or new rolls are needed for stronger confirmation.

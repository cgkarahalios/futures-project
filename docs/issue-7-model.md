# Issue #7 — Predict the liquidity crossover

Uses the feature panel from `notebooks/06_develop_early_indicators.ipynb`.
Run `notebooks/07_predict_crossover.ipynb` for the horizon comparison and chart.
The target is crossover within the next 1, 3, or 5 business days.

Run from the project root:

```bash
pip install -r requirements-model.txt
python -m unittest discover -s tests -v
python -m src.crossover_model --horizon 3 --min-train-rolls 6
```

The model compares a calendar-only logistic baseline against a logistic
model with the lagged liquidity features from notebook 06. Splits are
chronological and grouped by complete quarterly rolls. Brier score and
log loss are averaged equally across held-out rolls.

Features are end-of-day observations; a forecast is actionable **no earlier
than the following trading session**. The diagnostic
`days_until_crossover` is never a predictor. Outputs are saved under the
ignored `data/processed/` directory.

No forecast results are included in the PR: they require the Databento
datasets, which are intentionally not committed.

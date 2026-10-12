"""Regression tests for the roll-level out-of-sample model."""
import unittest

import numpy as np
import pandas as pd

from src.crossover_model import (
    score_predictions,
    walk_forward_predictions,
)


def example_panel(n_rolls=10):
    data = []
    for r in range(n_rolls):
        start = pd.Timestamp("2021-01-04") + pd.Timedelta(days=95 * r)
        for day in range(11):
            share = (day + r * 0.05) / 20
            data.append({
                "roll_id": f"roll-{r:02d}",
                "date": start + pd.Timedelta(days=day),
                "bdays_to_fnd": day - 12,
                "next_volume_share": share,
                "next_volume_share_change_3d": 0.15 if day >= 3 else np.nan,
                "next_volume_share_change_5d": 0.25 if day >= 5 else np.nan,
                "next_oi_share": 0.1 + day / 30,
                "next_oi_share_change_3d": 0.1 if day >= 3 else np.nan,
                "log_spread_volume": np.log1p(30 + 2 * day),
                "days_until_crossover": 11 - day,
                "crossover_within_3d": float(day >= 8),
            })
    return pd.DataFrame(data)


class WalkForwardTests(unittest.TestCase):
    def test_probabilities_and_holdout_rolls(self):
        preds = walk_forward_predictions(example_panel(), min_train_rolls=6)
        self.assertEqual(preds["roll_id"].nunique(), 4)
        self.assertGreaterEqual(preds["train_rolls"].min(), 6)
        self.assertTrue(preds["p_calendar"].between(0, 1).all())
        self.assertTrue(preds["p_model"].between(0, 1).all())
        scores = score_predictions(preds)
        self.assertEqual(set(scores["model"]), {"calendar", "features"})
        self.assertTrue(np.isfinite(scores["brier_roll_mean"]).all())

    def test_future_diagnostic_is_not_used(self):
        panel = example_panel()
        first = walk_forward_predictions(panel)
        panel["days_until_crossover"] = -999
        second = walk_forward_predictions(panel)
        np.testing.assert_allclose(first["p_model"], second["p_model"])

    def test_nonbinary_label_fails(self):
        panel = example_panel()
        panel.loc[0, "crossover_within_3d"] = 2
        with self.assertRaisesRegex(ValueError, "binary"):
            walk_forward_predictions(panel)

    def test_missing_columns_fail(self):
        with self.assertRaisesRegex(ValueError, "Missing columns"):
            walk_forward_predictions(example_panel().drop(columns=["roll_id"]))

    def test_duplicate_days_fail(self):
        panel = example_panel()
        panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            walk_forward_predictions(panel)

    def test_not_enough_prior_rolls_fail(self):
        with self.assertRaisesRegex(ValueError, "No eligible"):
            walk_forward_predictions(example_panel(3), min_train_rolls=6)


if __name__ == "__main__":
    unittest.main()

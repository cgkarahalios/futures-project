"""Leakage and evaluation-contract tests for Issue #20; no market data needed."""
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.timing_model import (
    BASELINE, FEATURES, _forecast, _raw_prediction, date_audit, evaluate, prepare_decisions,
    reproduce_legacy, select_model, summarize,
)
from src.timing_experiments import (
    EXPERIMENTAL_CANDIDATES, run_experiments, trajectory_decisions,
)


def example_panel(n_rolls=12):
    rows = []
    for roll in range(n_rolls):
        start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=90 * roll)
        event = -2 - (roll % 4 == 0)
        for offset in range(-20, 6):
            rows.append({
                "roll_id": f"roll-{roll:02d}",
                "date": start + pd.Timedelta(days=offset + 20),
                "bdays_to_fnd": offset,
                "next_volume_share": 0.6 if offset >= event else (offset + 21) / 100,
                "next_oi_share": (offset + 21) / 30,
                "spread_volume": 100 + (roll + 1) * (offset + 21),
            })
    return pd.DataFrame(rows)


class TimingTests(unittest.TestCase):
    def test_cutoff_never_uses_later_row_or_backfills_missing_value(self):
        panel = example_panel(1)
        panel = panel.loc[panel.bdays_to_fnd != -10].copy()
        panel.loc[panel.bdays_to_fnd == -11, "next_oi_share"] = np.nan
        snapshot = prepare_decisions(panel).iloc[0]
        self.assertEqual(snapshot.decision_bday, -11)
        self.assertTrue(np.isnan(snapshot.next_oi_share))

    def test_legacy_reproduction_is_separate_from_safe_snapshots(self):
        panel = example_panel()
        panel.loc[panel.bdays_to_fnd == -10, "next_oi_share"] = np.nan
        safe = prepare_decisions(panel)
        legacy = prepare_decisions(panel, legacy_snapshot=True)
        self.assertTrue(safe.next_oi_share.isna().all())
        self.assertTrue(legacy.next_oi_share.notna().all())
        result = reproduce_legacy(legacy)
        self.assertEqual(len(result), 4)

    def test_future_observations_do_not_enter_snapshot(self):
        panel = example_panel()
        original = prepare_decisions(panel)
        panel.loc[panel.bdays_to_fnd > -10, "spread_volume"] = 10**9
        panel.loc[panel.bdays_to_fnd > -10, "next_oi_share"] = 1
        changed = prepare_decisions(panel)
        pd.testing.assert_frame_equal(original[list(FEATURES)], changed[list(FEATURES)])

    def test_future_rolls_and_test_outcome_do_not_change_forecasts(self):
        decisions = prepare_decisions(example_panel())
        original, _ = evaluate(decisions)
        test_id = "roll-08"
        changed = decisions.copy()
        mask = changed.roll_id >= test_id
        changed.loc[mask, "actual_crossover_bday"] = -1
        changed.loc[mask, "actual_crossover_date"] += pd.Timedelta(days=1)
        # Test predictors stay fixed; arbitrary future predictors must not leak.
        changed.loc[changed.roll_id > test_id, list(FEATURES)] = 1000
        altered, _ = evaluate(changed)
        columns = ["model", "prediction", "raw_prediction", "selected_model"]
        pd.testing.assert_frame_equal(
            original.loc[original.roll_id == test_id, columns].reset_index(drop=True),
            altered.loc[altered.roll_id == test_id, columns].reset_index(drop=True),
        )

    def test_overlapping_or_incomplete_windows_are_not_training_data(self):
        decisions = prepare_decisions(example_panel())
        cutoff = decisions.loc[8, "decision_date"]
        decisions.loc[0, "roll_end"] = cutoff  # Not strictly before the decision.
        preds, _ = evaluate(decisions, min_train_rolls=7)
        test = preds.loc[preds.roll_id == "roll-08"]
        self.assertEqual(set(test.training_rolls), {7})
        self.assertTrue((preds.latest_training_end < preds.decision_date).all())

    def test_same_heldout_rolls_and_training_only_tuning(self):
        preds, tuning = evaluate(prepare_decisions(example_panel()))
        self.assertEqual(preds.roll_id.nunique(), 4)
        for _, group in preds.groupby("model"):
            self.assertEqual(set(group.roll_id), set(preds.roll_id))
        self.assertEqual(set(tuning.groupby("roll_id").selected.sum()), {1})
        self.assertTrue((preds.prediction > preds.decision_bday).all())
        summary, sensitivity = summarize(preds)
        self.assertTrue(np.isfinite(summary.mae).all())
        self.assertEqual(len(sensitivity), len(summary) * 4)
        baseline = summary.set_index("model").loc[BASELINE]
        self.assertEqual(baseline.mae_minus_baseline, 0)

    def test_ties_keep_baseline_and_missing_predictors_are_safe(self):
        decisions = prepare_decisions(example_panel(8))
        decisions["actual_crossover_bday"] = -2
        decisions[list(FEATURES)] = np.nan
        selected, scores, count = select_model(decisions)
        self.assertEqual(selected, BASELINE)
        self.assertGreater(count, 0)
        self.assertTrue(all(score == 0 for score in scores.values()))

    def test_forecast_is_not_actionable_on_feature_date(self):
        decisions = prepare_decisions(example_panel(9))
        train = decisions.iloc[:8].copy()
        train["actual_crossover_bday"] = -20
        _, prediction = _forecast(train, decisions.iloc[8], BASELINE)
        self.assertEqual(prediction, -9)

    def test_duplicate_or_out_of_order_offsets_fail(self):
        panel = example_panel(1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            prepare_decisions(pd.concat([panel, panel.iloc[[0]]]))
        panel["bdays_to_fnd"] = panel.bdays_to_fnd.astype(float)
        panel.loc[0, "bdays_to_fnd"] = -19.5
        with self.assertRaisesRegex(ValueError, "integers"):
            prepare_decisions(panel)

    def test_no_crossover_is_not_fabricated(self):
        panel = example_panel(1)
        panel["next_volume_share"] = 0.2
        decisions = prepare_decisions(panel)
        self.assertTrue(decisions.actual_crossover_bday.isna().all())
        with self.assertRaisesRegex(ValueError, "No eligible"):
            evaluate(decisions)

    def test_audit_exposes_weekends_and_fnd_mismatch(self):
        panel = example_panel(1)
        panel["front"], panel["next"] = "A", "B"
        panel["roll_id"] = "A->B"
        rolls = pd.DataFrame([{"front": "A", "next": "B", "fnd": "2020-01-22"}])
        audit = date_audit(panel, rolls).iloc[0]
        self.assertGreater(audit.sunday_rows, 0)
        self.assertFalse(audit.offset_zero_matches_supplied_fnd)

    def test_cost_notebook_uses_the_same_corrected_ols_forecasts(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / 'notebooks/08_test_roll_timing_costs.ipynb').read_text())
        scope = {"np": np, "pd": pd, "project_root": root,
                 "standard": example_panel(), "display": lambda *args: None}
        code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        exec(next(c for c in code if 'DECISION_BDAY = -10' in c), scope)
        exec(next(c for c in code if 'prediction_rows = []' in c), scope)
        expected, _ = evaluate(prepare_decisions(example_panel()))
        expected = expected.loc[expected.model == 'ols_asof'].set_index('roll_id')
        actual = scope['predictions'].set_index('roll_id')
        np.testing.assert_array_equal(actual.predicted_crossover_bday,
                                      expected.loc[actual.index, 'prediction'])


class TrajectoryTests(unittest.TestCase):
    def test_logistic_path_extrapolates_known_midpoint(self):
        panel = example_panel(1)
        time = panel.bdays_to_fnd.to_numpy()
        panel['next_volume_share'] = 1 / (1 + np.exp(-0.4 * (time + 2)))
        for cutoff in (-10, -5):
            decision = trajectory_decisions(panel, cutoff).iloc[0]
            self.assertAlmostEqual(decision.volume_logit_trend_bday, -2, places=8)

    def test_future_volume_does_not_change_projection(self):
        panel = example_panel(1)
        original = trajectory_decisions(panel, -5).iloc[0]
        panel.loc[panel.bdays_to_fnd > -5, 'next_volume_share'] = 0.99
        changed = trajectory_decisions(panel, -5).iloc[0]
        self.assertAlmostEqual(original.volume_logit_trend_bday,
                               changed.volume_logit_trend_bday)

    def test_decreasing_path_falls_back_to_baseline(self):
        panel = example_panel(9)
        panel['next_volume_share'] = 0.4 - panel.bdays_to_fnd / 100
        decisions = trajectory_decisions(panel, -5)
        self.assertTrue(decisions.volume_logit_trend_bday.isna().all())
        train = decisions.iloc[:8].copy()
        train['actual_crossover_bday'] = -2
        value = _raw_prediction(train, decisions.iloc[8], 'trend_shrunk',
                                EXPERIMENTAL_CANDIDATES)
        self.assertEqual(value, -2)

    def test_historical_bias_correction_and_fixed_shrinkage(self):
        train = pd.DataFrame({'actual_crossover_bday': [-2] * 6,
                              'volume_logit_trend_bday': [0.] * 6})
        # Historical bias is -2. New projection -2 becomes -4 after correction;
        # halfway between -4 and baseline -2 is -3.
        test = pd.Series({'volume_logit_trend_bday': -2.})
        self.assertEqual(_raw_prediction(train, test, 'trend_shrunk',
                                         EXPERIMENTAL_CANDIDATES), -3)

    def test_both_cutoffs_score_identical_rolls(self):
        predictions, summary, tuning = run_experiments(example_panel())
        cohorts = [set(g.roll_id) for _, g in predictions.groupby('decision_cutoff')]
        self.assertEqual(cohorts[0], cohorts[1])
        self.assertEqual(set(summary.rolls), {4})
        self.assertEqual(set(summary.excluded_test_rolls), {0})
        self.assertTrue((predictions.latest_training_end < predictions.decision_date).all())
        self.assertEqual(set(tuning.groupby(['decision_cutoff', 'roll_id']).selected.sum()), {1})

    def test_test_outcome_and_future_rolls_do_not_select_the_model(self):
        decisions = trajectory_decisions(example_panel(), -5)
        original, _ = evaluate(decisions, candidates=EXPERIMENTAL_CANDIDATES)
        changed = decisions.copy()
        changed.loc[changed.roll_id >= 'roll-08', 'actual_crossover_bday'] = -1
        changed.loc[changed.roll_id > 'roll-08', 'volume_logit_trend_bday'] = 500
        altered, _ = evaluate(changed, candidates=EXPERIMENTAL_CANDIDATES)
        columns = ['model', 'selected_model', 'prediction', 'raw_prediction']
        pd.testing.assert_frame_equal(
            original.loc[original.roll_id == 'roll-08', columns].reset_index(drop=True),
            altered.loc[altered.roll_id == 'roll-08', columns].reset_index(drop=True),
        )


if __name__ == "__main__":
    unittest.main()

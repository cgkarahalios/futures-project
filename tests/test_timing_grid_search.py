"""Chronological selection contracts for the exploratory grid search."""
import unittest

import pandas as pd

from test_timing_model import example_panel
from src.timing_grid_search import ALPHAS, FEATURE_SETS, candidate_grid, run_grid_search
from src.timing_model import BASELINE, evaluate, prepare_decisions


class GridSearchTests(unittest.TestCase):
    def test_grid_size_and_baseline_tie_priority(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 1 + len(ALPHAS) * len(FEATURE_SETS))
        self.assertEqual(next(iter(grid)), BASELINE)
        for features, alpha in list(grid.values())[1:]:
            self.assertIn(features, FEATURE_SETS.values())
            self.assertIn(alpha, ALPHAS)

    def test_test_labels_and_future_rolls_do_not_change_first_forecast(self):
        decisions = prepare_decisions(example_panel(), -5)
        original, _ = evaluate(decisions, candidates=candidate_grid())
        changed = decisions.copy()
        changed.loc[changed.roll_id >= 'roll-08', 'actual_crossover_bday'] = -1
        changed.loc[changed.roll_id > 'roll-08', 'next_volume_share'] = 99
        altered, _ = evaluate(changed, candidates=candidate_grid())
        columns = ['model', 'selected_model', 'prediction', 'raw_prediction']
        pd.testing.assert_frame_equal(
            original.loc[original.roll_id == 'roll-08', columns].reset_index(drop=True),
            altered.loc[altered.roll_id == 'roll-08', columns].reset_index(drop=True),
        )

    def test_each_selected_configuration_minimizes_earlier_validation_error(self):
        predictions, summary, tuning = run_grid_search(example_panel())
        self.assertEqual(set(summary.rolls), {4})
        self.assertEqual(set(summary.excluded_test_rolls), {0})
        self.assertTrue((predictions.latest_training_end < predictions.decision_date).all())
        for (cutoff, roll), scores in tuning.groupby(['decision_cutoff', 'roll_id']):
            selected = scores.loc[scores.selected]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.iloc[0].inner_mae, scores.inner_mae.min())
            forecast = predictions.loc[(predictions.decision_cutoff == cutoff)
                                       & (predictions.roll_id == roll)
                                       & (predictions.model == 'selected')].iloc[0]
            self.assertEqual(forecast.selected_model, selected.iloc[0].candidate)
        cohorts = [set(g.roll_id) for _, g in predictions.groupby('decision_cutoff')]
        self.assertEqual(cohorts[0], cohorts[1])


if __name__ == '__main__':
    unittest.main()

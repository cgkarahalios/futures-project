"""Tests for notebook run-order selection; no Databento access required."""
import unittest

from scripts.run_pipeline import NOTEBOOKS, STAGES, selected_notebooks


class PipelineTests(unittest.TestCase):
    def test_complete_pipeline(self):
        self.assertEqual(selected_notebooks(), [NOTEBOOKS / s for s in STAGES])

    def test_resume_from_raw_data(self):
        self.assertEqual(
            [p.name for p in selected_notebooks(2, 7)],
            list(STAGES[1:]),
        )

    def test_single_stage(self):
        self.assertEqual(selected_notebooks(7, 7)[0].name, STAGES[6])

    def test_bad_stage_range(self):
        with self.assertRaisesRegex(ValueError, "Require"):
            selected_notebooks(5, 3)


if __name__ == "__main__":
    unittest.main()

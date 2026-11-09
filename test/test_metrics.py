"""Tests for multi-label metrics, threshold tuning and experiment ranking (torch-free)."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diagnotech.metrics import (  # noqa: E402
    binarize,
    class_column_slug,
    exact_match_accuracy,
    hamming_accuracy,
    macro_auroc,
    per_class_auroc,
    per_class_report,
    rank_experiments,
    tune_thresholds,
)


class BinarizeAndAccuracyTests(unittest.TestCase):
    def test_binarize_scalar_and_per_class_thresholds(self):
        probs = np.array([[0.2, 0.6], [0.8, 0.4]])
        np.testing.assert_array_equal(binarize(probs, 0.5), [[0, 1], [1, 0]])
        np.testing.assert_array_equal(binarize(probs, [0.1, 0.7]), [[1, 0], [1, 0]])

    def test_hamming_and_exact_match(self):
        y_true = np.array([[1, 0], [1, 1]])
        y_pred = np.array([[1, 0], [1, 0]])
        self.assertAlmostEqual(hamming_accuracy(y_true, y_pred), 0.75)
        self.assertAlmostEqual(exact_match_accuracy(y_true, y_pred), 0.5)


class AurocTests(unittest.TestCase):
    def test_constant_column_is_nan_and_excluded_from_macro(self):
        y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])  # class 1 never positive
        y_prob = np.array([[0.9, 0.2], [0.1, 0.3], [0.8, 0.4], [0.2, 0.1]])
        per_class = per_class_auroc(y_true, y_prob)
        self.assertAlmostEqual(per_class[0], 1.0)
        self.assertTrue(np.isnan(per_class[1]))
        self.assertAlmostEqual(macro_auroc(y_true, y_prob), 1.0)


class PerClassReportTests(unittest.TestCase):
    def test_counts_and_derived_metrics_match_hand_calculation(self):
        classes = ["A", "B"]
        y_true = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
        y_prob = np.array([[0.9, 0.2], [0.4, 0.8], [0.1, 0.1], [0.3, 0.7]])
        report = per_class_report(y_true, y_prob, classes, thresholds=0.5).set_index("class")

        # Class A at 0.5: predicted [1, 0, 0, 0]; truth [1, 1, 0, 0]
        a = report.loc["A"]
        self.assertEqual((a.tp, a.fp, a.fn, a.tn), (1, 0, 1, 2))
        self.assertAlmostEqual(a.precision, 1.0)
        self.assertAlmostEqual(a.recall, 0.5)
        self.assertAlmostEqual(a.f1, 2 / 3)
        self.assertEqual(a.support, 2)

        # Class B at 0.5: predicted [0, 1, 0, 1]; truth [0, 1, 0, 1]
        b = report.loc["B"]
        self.assertEqual((b.tp, b.fp, b.fn, b.tn), (2, 0, 0, 2))
        self.assertAlmostEqual(b.f1, 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            per_class_report(np.zeros((3, 2)), np.zeros((3, 3)), ["A", "B"])


class ThresholdTuningTests(unittest.TestCase):
    def test_perfectly_separable_class_gets_separating_threshold(self):
        y_true = np.array([[0], [0], [1], [1]])
        y_prob = np.array([[0.1], [0.2], [0.8], [0.9]])
        thr = tune_thresholds(y_true, y_prob)
        self.assertGreater(thr[0], 0.2)
        self.assertLessEqual(thr[0], 0.8)
        preds = binarize(y_prob, thr)
        np.testing.assert_array_equal(preds.ravel(), y_true.ravel())

    def test_class_without_positives_keeps_default(self):
        thr = tune_thresholds(np.zeros((4, 1), dtype=int), np.random.default_rng(0).random((4, 1)))
        self.assertEqual(thr[0], 0.5)


class RankingTests(unittest.TestCase):
    def test_degraded_class_is_ineligible_even_with_high_macro_auroc(self):
        results = pd.DataFrame({
            "experiment_id": ["E0", "E1", "E2"],
            "val_macro_auroc": [0.80, 0.95, 0.85],
            "val_macro_f1_tuned": [0.5, 0.6, 0.55],
            "val_loss": [0.4, 0.3, 0.35],
            "auroc_mass": [0.80, 0.70, 0.84],     # E1 drops 0.10 on Mass: ineligible
            "auroc_nodule": [0.80, 0.90, 0.82],
        })
        ranked = rank_experiments(results, baseline_id="E0", max_class_auroc_drop=0.02)
        self.assertEqual(ranked.iloc[0]["experiment_id"], "E2")
        self.assertFalse(bool(ranked.set_index("experiment_id").loc["E1", "eligible"]))
        self.assertEqual(list(ranked["rank"]), [1, 2, 3])

    def test_missing_baseline_raises(self):
        results = pd.DataFrame({"experiment_id": ["E1"], "val_macro_auroc": [0.9],
                                "val_macro_f1_tuned": [0.5], "val_loss": [0.3], "auroc_mass": [0.9]})
        with self.assertRaises(ValueError):
            rank_experiments(results, baseline_id="E0")

    def test_class_column_slug(self):
        self.assertEqual(class_column_slug("No Finding"), "no_finding")
        self.assertEqual(class_column_slug("Pneumothorax"), "pneumothorax")


if __name__ == "__main__":
    unittest.main()

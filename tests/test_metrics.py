"""Tests for evaluation metrics (搂12.2)."""
import unittest

from dea_dd.diagnostics import metrics


class TestHypervolume(unittest.TestCase):
    def test_single_point(self):
        self.assertAlmostEqual(metrics.hypervolume_2d([1.0], [1.0], 2.0, 2.0), 1.0)

    def test_two_point_front(self):
        hv = metrics.hypervolume_2d([1.0, 2.0], [2.0, 1.0], 3.0, 3.0)
        self.assertAlmostEqual(hv, 3.0)

    def test_dominated_point_ignored(self):
        hv = metrics.hypervolume_2d([1.0, 2.0], [1.0, 2.0], 3.0, 3.0)
        self.assertAlmostEqual(hv, (3.0 - 1.0) * (3.0 - 1.0))

    def test_empty(self):
        self.assertEqual(metrics.hypervolume_2d([], [], 2.0, 2.0), 0.0)


class TestCMetric(unittest.TestCase):
    def test_full_coverage(self):
        self.assertAlmostEqual(
            metrics.c_metric([1.0], [1.0], [2.0], [2.0]), 1.0)

    def test_no_coverage(self):
        self.assertAlmostEqual(
            metrics.c_metric([2.0], [2.0], [1.0], [1.0]), 0.0)


class TestNPS(unittest.TestCase):
    def test_nps(self):
        self.assertEqual(metrics.nps([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), 3)
        self.assertEqual(metrics.nps([1.0, 2.0], [1.0, 2.0]), 1)


class TestIGD(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(
            metrics.igd([1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]), 0.0)


class TestRobustnessGap(unittest.TestCase):
    def test_gap(self):
        self.assertAlmostEqual(metrics.robustness_gap(10.0, 5.0), 1.0)
        self.assertAlmostEqual(metrics.robustness_gap(5.0, 10.0), -0.5)
        self.assertAlmostEqual(metrics.robustness_gap(5.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
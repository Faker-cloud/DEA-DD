"""Tests for Pareto machinery: dominance, non-dominated sort, crowding."""
import unittest

from dea_dd.core.pareto import (dominated, fast_non_dominated_sort,
                           crowding_distance, non_dominated_indices)


class TestDominance(unittest.TestCase):
    def test_basic_cases(self):
        self.assertTrue(dominated(1, 1, 2, 2))   # (1,1) dominates (2,2)
        self.assertTrue(dominated(1, 2, 1, 3))   # equal f1, better f2
        self.assertFalse(dominated(1, 2, 2, 1))  # incomparable
        self.assertFalse(dominated(1, 1, 1, 1))  # equal -> no strict dominance
        self.assertFalse(dominated(2, 2, 1, 1))  # worse on both


class TestNonDominatedSort(unittest.TestCase):
    def test_front_structure(self):
        f1 = [1, 2, 3, 2, 4]
        f2 = [1, 2, 1, 3, 2]
        fronts = fast_non_dominated_sort(f1, f2)
        self.assertEqual(fronts[0], [0])       # (1,1) dominates all
        self.assertEqual(sorted(fronts[1]), [1, 2])

    def test_all_incomparable(self):
        fronts = fast_non_dominated_sort([1, 2, 3], [3, 2, 1])
        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), [0, 1, 2])


class TestCrowdingDistance(unittest.TestCase):
    def test_boundaries_infinite(self):
        cd = crowding_distance([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
        self.assertEqual(cd[0], float("inf"))
        self.assertEqual(cd[2], float("inf"))
        self.assertGreater(cd[1], 0.0)

    def test_two_points(self):
        cd = crowding_distance([0.0, 1.0], [1.0, 0.0])
        self.assertEqual(cd[0], float("inf"))
        self.assertEqual(cd[1], float("inf"))


class TestNonDominatedIndices(unittest.TestCase):
    def test_indices(self):
        idx = non_dominated_indices([1, 2, 3], [3, 2, 1])
        self.assertEqual(sorted(idx), [0, 1, 2])

    def test_single_dominator(self):
        idx = non_dominated_indices([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(idx, [0])


if __name__ == "__main__":
    unittest.main()
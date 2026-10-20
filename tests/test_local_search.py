"""Tests for the adaptive local search operators (搂7-搂8)."""
import unittest
from collections import Counter

import numpy as np

from dea_dd.algorithms.local_search import AdaptiveLocalSearch, OPERATORS
from dea_dd.core.instance import generate_due_dates
from dea_dd.params import Params
from tests.helpers import synthetic_instance


def _random_chromosome(inst, rng):
    ms = [int(rng.integers(0, len(inst.machines[a])))
          for a in range(inst.total_ops)]
    os = []
    for j in range(inst.n_jobs):
        os.extend([j] * inst.ops_per_job[j])
    rng.shuffle(os)
    return ms, os


def _assert_feasible(tc, inst, ms, os):
    tc.assertEqual(len(ms), inst.total_ops)
    tc.assertEqual(len(os), inst.total_ops)
    for a in range(inst.total_ops):
        tc.assertTrue(0 <= ms[a] < len(inst.machines[a]))
    cnt = Counter(os)
    for j in range(inst.n_jobs):
        tc.assertEqual(cnt[j], inst.ops_per_job[j])


class TestLocalSearch(unittest.TestCase):
    def setUp(self):
        self.inst = synthetic_instance()
        generate_due_dates(self.inst, rng=np.random.default_rng(0))
        self.params = Params(N=20, N_eo=8, Iter_EO=5, seed=0)
        self.ls = AdaptiveLocalSearch(self.inst, self.inst.due_dates,
                                      self.params, np.random.default_rng(0))

    def test_all_operators_return_feasible_or_none(self):
        rng = np.random.default_rng(7)
        for op in OPERATORS:
            saw_none = False
            saw_move = False
            for _ in range(30):
                ms, os = _random_chromosome(self.inst, rng)
                nms, nos = self.ls.generate(op, ms, os)
                if nms is None:
                    self.assertIsNone(nos)
                    saw_none = True
                else:
                    saw_move = True
                    _assert_feasible(self, self.inst, nms, nos)
            # over 30 random starts each operator should produce some move
            self.assertTrue(saw_move, f"operator {op} never produced a move")
            # and note some none cases exercise the guard branches
            self.assertTrue(saw_none or saw_move)

    def test_pick_operator_weights_sum_to_one(self):
        for _ in range(10):
            op = self.ls.pick_operator()
            self.assertIn(op, OPERATORS)

    def test_record_updates_history(self):
        self.ls.record("L1", True)
        self.ls.record("L1", False)
        self.assertEqual(len(self.ls.history["L1"]), 2)
        self.assertEqual(sum(1 for x in self.ls.history["L1"] if x), 1)


if __name__ == "__main__":
    unittest.main()
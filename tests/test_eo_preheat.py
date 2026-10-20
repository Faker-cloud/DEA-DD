"""Tests for the EO preheat phase (鍒涙柊鈶? 搂5)."""
import unittest

import numpy as np

from dea_dd.core.normalization import Normalizer
from dea_dd.params import Params
from dea_dd.algorithms.eo import EOSolver
from dea_dd.algorithms.preheat import preheat
from dea_dd.core.instance import generate_due_dates
from tests.helpers import synthetic_instance


class TestEO(unittest.TestCase):
    def setUp(self):
        self.inst = synthetic_instance()
        generate_due_dates(self.inst, rng=np.random.default_rng(0))
        self.params = Params(N=20, N_eo=8, Iter_EO=5, seed=0)
        self.params.K = min(self.params.K,
                            max(1, self.params.N // len(self.params.weights)))

    def test_eo_returns_feasible_candidates(self):
        norm = Normalizer()
        solver = EOSolver(self.inst, 0.5, norm, self.params,
                          np.random.default_rng(0))
        cands, conv = solver.run()
        self.assertGreater(len(cands), 0)
        self.assertEqual(len(conv), self.params.Iter_EO + 10 - 9)  # iters+1
        for ms, os, f1, f2 in cands:
            self.assertEqual(len(ms), self.inst.total_ops)
            self.assertEqual(len(os), self.inst.total_ops)
            self.assertTrue(all(0 <= ms[a] < len(self.inst.machines[a])
                                for a in range(self.inst.total_ops)))
            self.assertGreaterEqual(f1, 0.0)
            self.assertGreaterEqual(f2, 0.0)

    def test_convergence_len(self):
        norm = Normalizer()
        solver = EOSolver(self.inst, 0.0, norm, self.params,
                          np.random.default_rng(1))
        _, conv = solver.run()
        self.assertEqual(len(conv), self.params.Iter_EO + 1)


class TestPreheat(unittest.TestCase):
    def test_seed_count_and_due_wired(self):
        inst = synthetic_instance()
        generate_due_dates(inst, rng=np.random.default_rng(2))
        params = Params(N=20, N_eo=8, Iter_EO=5, seed=0)
        params.K = min(params.K, max(1, params.N // len(params.weights)))
        norm = Normalizer()
        seeds = preheat(inst, inst.due_dates, params, norm,
                        np.random.default_rng(3))
        self.assertGreater(len(seeds), 0)
        self.assertLessEqual(len(seeds), params.N)
        self.assertFalse(np.all(inst.due_dates == 0.0))
        for ms, os, f1, f2 in seeds:
            self.assertEqual(len(ms), inst.total_ops)
            self.assertEqual(len(os), inst.total_ops)

    def test_history_populated(self):
        inst = synthetic_instance()
        generate_due_dates(inst, rng=np.random.default_rng(4))
        params = Params(N=16, N_eo=6, Iter_EO=3, seed=0)
        params.K = min(params.K, max(1, params.N // len(params.weights)))
        hist = {}
        preheat(inst, inst.due_dates, params, Normalizer(),
                np.random.default_rng(5), history=hist)
        self.assertIn("preheat_conv", hist)
        self.assertIn("preheat_w_fronts", hist)
        self.assertEqual(set(hist["preheat_conv"]), set(params.weights))

    def test_fill_random_to_N(self):
        """预热去重后不足 N 时，应随机补齐到 N（合成实例搜索空间 ≥ 20）。"""
        inst = synthetic_instance()
        generate_due_dates(inst, rng=np.random.default_rng(6))
        params = Params(N=20, N_eo=8, Iter_EO=5, seed=0)
        # 钳到 K=2 → 7 个 w 最多 14 个种子 < N=20，必触发随机补齐
        params.K = min(params.K, max(1, params.N // len(params.weights)))
        seeds = preheat(inst, inst.due_dates, params, Normalizer(),
                        np.random.default_rng(7))
        self.assertEqual(len(seeds), params.N)
        for ms, os, f1, f2 in seeds:
            self.assertEqual(len(ms), inst.total_ops)
            self.assertEqual(len(os), inst.total_ops)


if __name__ == "__main__":
    unittest.main()
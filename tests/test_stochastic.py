"""Tests for the lognormal stochastic layer (搂9)."""
import unittest

import numpy as np

from dea_dd.simulation.stochastic import LognormalProcessing, mc_evaluate, mc_evaluate_long
from dea_dd.core.decoding import decode_evaluate
from tests.helpers import toy_instance


class TestLognormalProcessing(unittest.TestCase):
    def test_mean_matches_nominal(self):
        inst = toy_instance()
        proc = LognormalProcessing(inst, 0.25, np.random.default_rng(0))
        n = 5000
        # O11 machine 0 has nominal time 3
        mean = float(np.mean([proc.sample_scenario()[0][0] for _ in range(n)]))
        self.assertAlmostEqual(mean, 3.0, delta=0.1)

    def test_ul_zero_is_deterministic(self):
        inst = toy_instance()
        proc = LognormalProcessing(inst, 0.0, np.random.default_rng(0))
        s = proc.sample_scenario()
        self.assertAlmostEqual(s[0][0], 3.0, places=6)

    def test_scenario_shape(self):
        inst = toy_instance()
        proc = LognormalProcessing(inst, 0.25, np.random.default_rng(0))
        s = proc.sample_scenario()
        self.assertEqual(len(s), inst.total_ops)
        for a in range(inst.total_ops):
            self.assertEqual(len(s[a]), len(inst.times[a]))


class TestMCEvaluate(unittest.TestCase):
    def test_mc_matches_decode_when_ul_zero(self):
        inst = toy_instance()
        ms = [0, 1, 0, 0, 0]
        os = [0, 1, 0, 2, 1]          # 0-based toy schedule
        due = inst.due_dates
        proc = LognormalProcessing(inst, 0.0, np.random.default_rng(0))
        scenarios = proc.sample_scenarios(10)
        f1, f2 = mc_evaluate(inst, ms, os, scenarios, due)
        d1, d2 = decode_evaluate(inst, ms, os, due)
        self.assertAlmostEqual(f1, d1, places=6)
        self.assertAlmostEqual(f2, d2, places=6)

    def test_long_sim_std_zero_when_ul_zero(self):
        inst = toy_instance()
        ms = [0, 1, 0, 0, 0]
        os = [0, 1, 0, 2, 1]
        proc = LognormalProcessing(inst, 0.0, np.random.default_rng(0))
        m1, m2, s1, s2 = mc_evaluate_long(inst, ms, os, proc, inst.due_dates, 20)
        self.assertAlmostEqual(s1, 0.0, places=6)
        self.assertAlmostEqual(s2, 0.0, places=6)
        self.assertGreaterEqual(m1, 0.0)
        self.assertGreaterEqual(m2, 0.0)


if __name__ == "__main__":
    unittest.main()
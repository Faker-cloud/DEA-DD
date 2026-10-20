"""Tests for NSGA-II operators: crossover, dynamic mutation, feasibility."""
import unittest
from collections import Counter

import numpy as np

from dea_dd.algorithms.nsga2 import (pox_crossover, ux_crossover, dynamic_mutation_bits,
                          mutate_os, mutate_ms, environmental_selection,
                          Individual)
from dea_dd.core.normalization import Normalizer
from tests.helpers import synthetic_instance


class TestCrossover(unittest.TestCase):
    def setUp(self):
        self.inst = synthetic_instance()
        self.rng = np.random.default_rng(0)

    def test_pox_preserves_multiplicities(self):
        os1 = [0, 1, 0, 1]
        os2 = [1, 0, 1, 0]
        c1, c2 = pox_crossover(os1, os2, self.inst.n_jobs, self.rng)
        for c in (c1, c2):
            cnt = Counter(c)
            self.assertEqual(cnt[0], 2)
            self.assertEqual(cnt[1], 2)

    def test_ux_preserves_feasibility(self):
        ms1 = [0, 0, 1, 0]
        ms2 = [1, 0, 0, 0]
        c1, c2 = ux_crossover(ms1, ms2, 0.5, self.rng)
        for c in (c1, c2):
            for a in range(self.inst.total_ops):
                self.assertTrue(0 <= c[a] < len(self.inst.machines[a]))


class TestMutation(unittest.TestCase):
    def setUp(self):
        self.inst = synthetic_instance()
        self.rng = np.random.default_rng(1)

    def test_dynamic_bits_decaying_and_min(self):
        k_high = dynamic_mutation_bits(1, 100, 100)
        k_low = dynamic_mutation_bits(100, 100, 100)
        self.assertGreaterEqual(k_high, 2)
        self.assertGreaterEqual(k_low, 2)
        self.assertGreaterEqual(k_high, k_low)

    def test_mutate_os_preserves_multiplicities(self):
        os = [0, 1, 0, 1]
        for _ in range(50):
            out = mutate_os(os, 3, self.rng)
            self.assertEqual(Counter(out), Counter({0: 2, 1: 2}))

    def test_mutate_ms_preserves_feasibility(self):
        ms = [0, 0, 1, 0]
        for _ in range(50):
            out = mutate_ms(ms, self.inst, 3, self.rng)
            for a in range(self.inst.total_ops):
                self.assertTrue(0 <= out[a] < len(self.inst.machines[a]))


class TestEnvironmentalSelection(unittest.TestCase):
    def test_truncates_to_N(self):
        parents = [Individual([0, 0, 1, 0], [0, 1, 0, 1], 10.0 + i, 20.0 - i)
                   for i in range(6)]
        offspring = [Individual([1, 0, 0, 0], [1, 0, 1, 0], 5.0, 5.0)
                     for _ in range(6)]
        norm = Normalizer()
        norm.update_many([p.f1 for p in parents] + [o.f1 for o in offspring],
                         [p.f2 for p in parents] + [o.f2 for o in offspring])
        sel = environmental_selection(parents, offspring, norm, 6)
        self.assertEqual(len(sel), 6)
        self.assertIsInstance(sel[0], Individual)


if __name__ == "__main__":
    unittest.main()
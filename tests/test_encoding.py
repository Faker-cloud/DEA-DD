"""Tests for encoding and EO particle decoding (M2)."""

import unittest

import numpy as np

from dea_dd.core.encoding import reflect_coord, reflect_u, particle_to_chromosome

from tests.helpers import toy_instance


class TestReflect(unittest.TestCase):
    def test_inside_unchanged(self):
        self.assertAlmostEqual(reflect_coord(1.2, 3.0), 1.2)

    def test_positive_overshoot_reflects(self):
        # x=3.5 in [0,3): fold 3.5 -> 2.5
        self.assertAlmostEqual(reflect_coord(3.5, 3.0), 2.5)

    def test_negative_reflects(self):
        # x=-1.2 -> 1.2
        self.assertAlmostEqual(reflect_coord(-1.2, 3.0), 1.2)

    def test_exact_boundary_steps_back(self):
        self.assertLess(reflect_coord(3.0, 3.0), 3.0)

    def test_range(self):
        for x in np.linspace(-10, 10, 2001):
            r = reflect_coord(x, 3.0)
            self.assertGreaterEqual(r, 0.0)
            self.assertLess(r, 3.0)


class TestParticleToChromosome(unittest.TestCase):
    def test_feasible_encoding(self):
        inst = toy_instance()
        # bounds = per-op eligible machine count = 2 for all ops
        bounds = np.array([len(m) for m in inst.machines])
        rng = np.random.default_rng(0)
        for _ in range(100):
            u = rng.uniform(-5, 5, size=inst.total_ops)
            v = rng.uniform(-5, 5, size=inst.total_ops)
            u = reflect_u(u, bounds)
            ms, os = particle_to_chromosome(inst, u, v)
            # ms within machine list
            for a in range(inst.total_ops):
                self.assertTrue(0 <= ms[a] < len(inst.machines[a]))
            # os is a topological order of the job precedences
            seen = [0] * inst.n_jobs
            for job in os:
                self.assertEqual(seen[job], inst.op_index_in_job[
                    inst.job_op_ids[job][seen[job]]
                ])
                seen[job] += 1
            self.assertEqual(len(os), inst.total_ops)
            self.assertEqual(seen, inst.ops_per_job)


if __name__ == "__main__":
    unittest.main()
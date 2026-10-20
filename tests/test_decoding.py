"""Tests for dual decoding (M2). Verifies 算法设计.md 搂3.1.1 toy example."""

import unittest

from dea_dd.core.decoding import (
    left_shift_decode,
    right_shift_twte,
    decode_evaluate,
    twte,
)

from tests.helpers import toy_instance, synthetic_instance


class TestToyExample(unittest.TestCase):
    def test_left_shift_makespan_13(self):
        inst = toy_instance()
        ms = [0, 1, 0, 0, 0]
        os = [0, 1, 0, 2, 1]     # 0-based job ids
        S, C, machine_seq, _ = left_shift_decode(inst, ms, os)
        self.assertEqual(max(C), 13)
        # job completion times
        job_comp = [C[last] for last in inst.last_op_of_job]
        self.assertEqual(job_comp, [5, 13, 8])

    def test_left_shift_twte_4(self):
        inst = toy_instance()
        ms = [0, 1, 0, 0, 0]
        os = [0, 1, 0, 2, 1]
        S, C, machine_seq, _ = left_shift_decode(inst, ms, os)
        job_comp = [C[last] for last in inst.last_op_of_job]
        self.assertEqual(twte(job_comp, inst.due_dates), 4.0)

    def test_right_shift_twte_1(self):
        inst = toy_instance()
        ms = [0, 1, 0, 0, 0]
        os = [0, 1, 0, 2, 1]
        S, C, machine_seq, _ = left_shift_decode(inst, ms, os)
        f2, LC = right_shift_twte(inst, ms, C, machine_seq, inst.due_dates)
        self.assertEqual(f2, 1.0)

    def test_decode_evaluate(self):
        inst = toy_instance()
        f1, f2 = decode_evaluate(inst, [0, 1, 0, 0, 0], [0, 1, 0, 2, 1], inst.due_dates)
        self.assertEqual(f1, 13.0)
        self.assertEqual(f2, 1.0)


class TestSynthetic(unittest.TestCase):
    def test_precedence_respected(self):
        inst = synthetic_instance()
        ms = [0, 0, 0, 0]
        os = [0, 1, 0, 1]   # job0 op0, job1 op0, job0 op1, job1 op1
        S, C, machine_seq, _ = left_shift_decode(inst, ms, os)
        # job0 op1 (id 1) cannot start before job0 op0 (id 0) finishes
        self.assertGreaterEqual(S[1], C[0])
        self.assertGreaterEqual(S[3], C[2])

    def test_machine_disjoint(self):
        inst = synthetic_instance()
        ms = [0, 0, 0, 0]
        os = [0, 1, 0, 1]
        S, C, machine_seq, _ = left_shift_decode(inst, ms, os)
        for seq in machine_seq:
            for a, b in zip(seq, seq[1:]):
                self.assertGreaterEqual(S[b], C[a])


if __name__ == "__main__":
    unittest.main()
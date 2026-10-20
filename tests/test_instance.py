"""Instance 解析与交货期生成的测试。"""

import unittest

from dea_dd.core.instance import parse_fjs, generate_due_dates


# §3.1.1 玩具例：3 工件 × 2 机器，5 道工序，每道工序 2 台可选机器。
# O11(M1=3,M2=4) O12(M1=5,M2=2) | O21(M1=2,M2=4) O22(M1=5,M2=1) | O31(M1=3,M2=3)
FJS_SAMPLE = """\
3 2 1.5
2 2 2 3 2 4 2 1 5 2 2
2 2 1 2 2 4 2 1 5 2 1
1 2 1 3 2 3
"""


class TestParseFjs(unittest.TestCase):
    def test_parse_basic(self):
        inst = parse_fjs(FJS_SAMPLE)
        self.assertEqual(inst.n_jobs, 3)
        self.assertEqual(inst.n_machines, 2)
        self.assertEqual(inst.ops_per_job, [2, 2, 1])
        self.assertEqual(inst.total_ops, 5)
        # 机器号 1 基 -> 0 基
        self.assertEqual(inst.machines[0], [1, 1])
        self.assertEqual(inst.times[0], [3, 4])
        # 全部 5 道工序都应是 [机器1, 机器2] 两个可选（§3.1.1）
        self.assertEqual(inst.machines, [[1, 1], [0, 1], [0, 1], [0, 1], [0, 1]])
        self.assertEqual(inst.times, [[3, 4], [5, 2], [2, 4], [5, 1], [3, 3]])
        self.assertEqual(inst.last_op_of_job, [1, 3, 4])
        self.assertEqual(inst.job_of_op[4], 2)
        self.assertEqual(inst.op_index_in_job[4], 0)

    def test_parse_rejects_leftover_tokens(self):
        # 第 2 行结尾多了一个数字 9：两道工序都解析完后仍余下 1 个 token，
        # 应命中「工序行有多余数字」校验（所有机器号均合法）
        bad = (
            "3 2 1.5\n"
            "2 2 1 3 2 4 2 1 5 2 2 9\n"
            "2 2 1 2 2 4 2 1 5 2 1\n"
            "1 2 1 3 2 3\n"
        )
        with self.assertRaises(ValueError) as ctx:
            parse_fjs(bad)
        self.assertIn("多余数字", str(ctx.exception))

    def test_parse_rejects_out_of_range_machine(self):
        # 机器号 5 超出 [1, 2]
        bad = "1 2 1\n1 1 5 2\n"
        with self.assertRaises(ValueError):
            parse_fjs(bad)

    def test_due_dates_shape_and_range(self):
        inst = parse_fjs(FJS_SAMPLE)
        d = generate_due_dates(inst)
        self.assertEqual(len(d), 3)
        self.assertTrue((d > 0).all())


if __name__ == "__main__":
    unittest.main()
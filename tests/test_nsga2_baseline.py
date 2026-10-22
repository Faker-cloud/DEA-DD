"""传统 NSGA-II 基线的单元测试。"""
from __future__ import annotations

import unittest

from dea_dd.baselines.nsga2 import MUT_BITS, run_nsga2, main_nsga2
from dea_dd.params import Params


def _tiny_params(**overrides):
    """构造小规模的共享参数，便于快速测试。

    :param overrides: 覆盖的参数字段
    :return: 小规模 :class:`Params`
    :rtype: Params
    """
    base = dict(N=10, gen_max=3, R_short=4, R_long=30, seed=1, p_m=0.1)
    base.update(overrides)
    return Params(**base)


class TestNSGA2Baseline(unittest.TestCase):
    """测试基线的返回契约、无 EO 预热、确定性运行。"""

    def test_run_nsga2_contract(self):
        """run_nsga2 返回 (history, final_points)，且结构完整。"""
        history, final_points = run_nsga2(
            _tiny_params(), inst_path=None, out_dir="outputs/test_nsga2",
            log_dir="logs")
        # history 含随机初始前沿、收敛序列、最终前沿，且无 EO 相关键
        self.assertIn("initial_front", history)
        self.assertIn("nsga2_conv", history)
        self.assertIn("final_front", history)
        self.assertNotIn("preheat_conv", history)
        self.assertNotIn("preheat_w_fronts", history)
        # final_points 非空，每个点都是 (E_f1, E_f2, std_f1, std_f2) 四元组
        self.assertTrue(final_points)
        for p in final_points:
            self.assertEqual(len(p), 4)
        # 收敛序列长度等于 gen_max
        self.assertEqual(len(history["nsga2_conv"]), 3)

    def test_run_nsga2_deterministic_given_seed(self):
        """相同 seed 下两次运行的最终前沿应完全一致（可复现）。"""
        _, f1 = run_nsga2(_tiny_params(seed=7), inst_path=None,
                          out_dir="outputs/test_nsga2", log_dir="logs")
        _, f2 = run_nsga2(_tiny_params(seed=7), inst_path=None,
                          out_dir="outputs/test_nsga2", log_dir="logs")
        self.assertEqual([p[:2] for p in f1], [p[:2] for p in f2])

    def test_mut_bits_is_fixed_two(self):
        """传统 NSGA-II 固定变异位数为 2（对应消融“固定变异 2 位”）。"""
        self.assertEqual(MUT_BITS, 2)

    def test_main_nsga2_programmatic(self):
        """程序化入口应能直接传入 Params 运行。"""
        history, final_points = main_nsga2(
            params=_tiny_params(), inst_path=None,
            out_dir="outputs/test_nsga2", log_dir="logs")
        self.assertIn("final_front", history)
        self.assertTrue(final_points)


if __name__ == "__main__":
    unittest.main()
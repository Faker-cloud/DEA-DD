"""compare 模块多轮对比的单元测试（run 用 mock 快速替换，避免真跑算法）。"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from dea_dd.compare import _list_fjs_files, _summarize, compare
from dea_dd.params import Params


def _fake_front(seed):
    """返回形如 run() 的 final_points 的小前沿，随 seed 变化以便验证递增。"""
    return [(float(seed + 10), float(seed + 20), 0.0, 0.0),
            (float(seed + 12), float(seed + 18), 0.0, 0.0)]


def _mock_runners():
    """把 compare 里的 run_dea_dd / run_nsga2 换成返回假前沿的桩。"""
    return (mock.patch(
                "dea_dd.compare.run_dea_dd",
                side_effect=lambda p, ip, od, ld: ({}, _fake_front(p.seed))),
            mock.patch(
                "dea_dd.compare.run_nsga2",
                side_effect=lambda p, ip, od, ld: ({}, _fake_front(p.seed))))


class TestSummarize(unittest.TestCase):
    """_summarize 纯函数：mean/std/min/max。"""

    def test_mean_std_extrema(self):
        results = [
            {"NPS_a": 2, "HV_a": 10.0, "NPS_b": 1, "HV_b": 20.0,
             "C_ab": 0.5, "C_ba": 0.0},
            {"NPS_a": 4, "HV_a": 30.0, "NPS_b": 1, "HV_b": 20.0,
             "C_ab": 0.7, "C_ba": 0.0},
        ]
        s = _summarize(results)
        self.assertAlmostEqual(s["HV_a"]["mean"], 20.0)
        self.assertAlmostEqual(s["C_ab"]["mean"], 0.6)
        self.assertEqual(s["NPS_a"]["max"], 4)
        self.assertAlmostEqual(s["NPS_b"]["std"], 0.0)

    def test_single_result_std_zero(self):
        results = [{"NPS_a": 3, "HV_a": 1.0, "NPS_b": 2, "HV_b": 2.0,
                    "C_ab": 1.0, "C_ba": 0.0}]
        s = _summarize(results)
        self.assertAlmostEqual(s["HV_a"]["std"], 0.0)


class TestListFjsFiles(unittest.TestCase):
    """_list_fjs_files：文件直通 + 目录非递归扫描。"""

    def test_file_passthrough(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "x.fjs")
            open(p, "w").close()
            self.assertEqual(_list_fjs_files(p), [p])

    def test_dir_nonrecursive_sorted(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "sub"))
            for n in ("c.fjs", "a.fjs", "b.fjs"):
                open(os.path.join(td, n), "w").close()
            # 子目录里的 .fjs 应被忽略（非递归）；非 .fjs 文件也忽略
            open(os.path.join(td, "sub", "nested.fjs"), "w").close()
            open(os.path.join(td, "notes.txt"), "w").close()
            files = _list_fjs_files(td)
            self.assertEqual(files, [os.path.join(td, "a.fjs"),
                                     os.path.join(td, "b.fjs"),
                                     os.path.join(td, "c.fjs")])

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _list_fjs_files(td)


class TestCompareMultiRun(unittest.TestCase):
    """多轮 compare：seed 递增、目录隔离、汇总表输出、目录/文件双向支持。"""

    def test_rejects_runs_below_one(self):
        with self.assertRaises(ValueError):
            compare(Params(N=10, gen_max=3), runs=0)

    def test_single_instance_nested_and_seed_advances(self):
        out_root = "outputs/test_compare"
        params = Params(N=10, gen_max=3, R_short=2, R_long=12, seed=7)
        m1, m2 = _mock_runners()
        with m1, m2:
            out = compare(params, runs=3, out_root=out_root, log_dir="logs")

        first = out["instances"][0]
        self.assertEqual(first["instance"], "synthetic")   # inst_path=None
        self.assertEqual([r["seed"] for r in first["runs"]], [7, 8, 9])
        inst_dir = os.path.join(out_root, "synthetic")
        for name in ("run_001", "run_002", "run_003"):
            rdir = os.path.join(inst_dir, name)
            self.assertTrue(os.path.isdir(rdir), "缺少目录 " + rdir)
            self.assertTrue(os.path.exists(
                os.path.join(rdir, "front_comparison.png")))
        self.assertTrue(os.path.exists(os.path.join(inst_dir, "summary.csv")))
        self.assertIn("summary", first)
        self.assertIsNone(out["summary_all"])             # 单算例无跨例汇总

    def test_directory_isolates_instances(self):
        out_root = "outputs/test_compare_dir"
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "mk1.fjs"), "w").close()
            open(os.path.join(td, "mk2.fjs"), "w").close()
            params = Params(N=10, gen_max=3, R_short=2, R_long=12, seed=3)
            m1, m2 = _mock_runners()
            with m1, m2:
                out = compare(params, inst_path=td, runs=2,
                              out_root=out_root, log_dir="logs")

        self.assertEqual([i["instance"] for i in out["instances"]],
                         ["mk1", "mk2"])                  # 非递归、按名排序
        for name in ("mk1", "mk2"):
            inst_dir = os.path.join(out_root, name)
            self.assertTrue(os.path.isdir(os.path.join(inst_dir, "run_001")))
            self.assertTrue(os.path.isdir(os.path.join(inst_dir, "run_002")))
            self.assertTrue(os.path.exists(
                os.path.join(inst_dir, "summary.csv")))
        self.assertTrue(os.path.exists(
            os.path.join(out_root, "summary_all.csv")))   # 多算例才有跨例汇总


if __name__ == "__main__":
    unittest.main()
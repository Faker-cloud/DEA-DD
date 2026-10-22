"""对比实验入口：用相同参数跑 DEA-DD 与传统 NSGA-II 基线，输出对比指标与图。

运行两者时共用同一 ``Params``（重叠参数一致），**单次运行内**二者共用同一
``seed``，因此二者在同一算例、同一交货期、同一随机流起点下比较。默认每个算例
重复跑 ``runs=10`` 次，每次 ``seed = 基准 seed + 运行序号``（第 i 次用 seed+i）。

``inst_path`` 既可传单个 ``.fjs`` 文件，也可传目录（**非递归**，只扫该目录顶层
所有 ``.fjs``）：每个算例结果隔离到 ``<out_root>/<算例名>/run_001/``、
``run_002/`` … 下，逐算例汇总表写 ``<out_root>/<算例名>/summary.csv``；多算例时
额外输出跨算例汇总表 ``<out_root>/summary_all.csv``。

每次运行的指标：两个前沿各自的 NPS、超体积 HV（**同一参考点**，否则不可比）、
双向覆盖率 ``C(DEA-DD | NSGA-II)`` 与 ``C(NSGA-II | DEA-DD)``，并保存合并前沿
对比图 ``front_comparison.png``（各次隔离到对应 ``run_XXX/`` 下）。
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import statistics
import sys

if __package__ in ("", None):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from dea_dd.diagnostics import metrics
from dea_dd.diagnostics.logging_setup import setup_logging
from dea_dd.params import Params
from dea_dd.paths import resolve_project_path
from dea_dd.main import run as run_dea_dd
from dea_dd.baselines.nsga2 import run_nsga2

# 参与汇总与 CSV 的数值指标及打印格式
_METRIC_KEYS = ("NPS_a", "HV_a", "NPS_b", "HV_b", "C_ab", "C_ba")
_METRIC_FMT = {"NPS_a": "{:.0f}", "NPS_b": "{:.0f}",
               "HV_a": "{:.4f}", "HV_b": "{:.4f}",
               "C_ab": "{:.3f}", "C_ba": "{:.3f}"}

# inst_path 为 None（合成算例）时的显示名与目录名
_SYNTHETIC_NAME = "synthetic"


def _plot_comparison(front_a, front_b, out_dir):
    """绘制 DEA-DD 与 NSGA-II 两条最终前沿的合并对比图。

    :param front_a: DEA-DD 最终前沿点 ``[(f1, f2), ...]``
    :param front_b: NSGA-II 最终前沿点 ``[(f1, f2), ...]``
    :param out_dir: 输出目录
    :return: 保存路径 ``out_dir/front_comparison.png``
    :rtype: str
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if front_a:
        xa = [p[0] for p in front_a]
        ya = [p[1] for p in front_a]
        order = sorted(range(len(xa)), key=lambda i: xa[i])
        ax.plot([xa[i] for i in order], [ya[i] for i in order],
                marker="o", ms=6, color="tab:purple", label="DEA-DD")
    if front_b:
        xb = [p[0] for p in front_b]
        yb = [p[1] for p in front_b]
        order = sorted(range(len(xb)), key=lambda i: xb[i])
        ax.plot([xb[i] for i in order], [yb[i] for i in order],
                marker="s", ms=6, color="tab:orange", label="传统 NSGA-II")
    ax.set_xlabel("makespan（期望）")
    ax.set_ylabel("TWTE（期望）")
    ax.set_title("最终前沿对比：DEA-DD vs 传统 NSGA-II")
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "front_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _list_fjs_files(inst_path):
    """把 ``inst_path`` 规约为 ``.fjs`` 文件列表。

    文件原样返回；目录则**非递归**列出其顶层所有 ``.fjs``（按名字排序）。

    :param inst_path: `.fjs` 文件或目录路径
    :return: `.fjs` 文件路径列表
    :rtype: list
    :raises ValueError: 目录下没有任何 ``.fjs`` 文件
    """
    if os.path.isdir(inst_path):
        files = sorted(
            os.path.join(inst_path, name)
            for name in os.listdir(inst_path)
            if name.lower().endswith(".fjs")
        )
        if not files:
            raise ValueError("目录 " + inst_path + " 下没有 .fjs 文件")
        return files
    return [inst_path]


def _instance_name(inst_path):
    """算例显示名：文件名去掉 ``.fjs`` 后缀；None 用合成名。"""
    if inst_path is None:
        return _SYNTHETIC_NAME
    return os.path.splitext(os.path.basename(inst_path))[0]


def _compare_once(params, inst_path, run_dir, log_dir, log, inst_name, run_no):
    """执行单算例单次对比：跑 DEA-DD 与 NSGA-II，图隔离写入 ``run_dir``。

    :param params: 本次运行的参数（seed 已被调用方替换为本轮 seed）
    :param inst_path: `.fjs` 文件路径（None 表示合成算例）
    :param run_dir: 本轮的隔离输出目录（DEA-DD 图入 ``run_dir/dea_dd``，
        NSGA-II 图入 ``run_dir/nsga2``，对比图入 ``run_dir`` 根）
    :param log_dir: 日志目录
    :param log: ``compare`` 记录器
    :param inst_name: 算例显示名（仅用于日志）
    :param run_no: 本轮序号（1 起，仅用于打印与返回）
    :return: 含本轮指标、前沿与对比图路径的 dict
    :rtype: dict
    """
    hist_a, front_a = run_dea_dd(params, inst_path,
                                 os.path.join(run_dir, "dea_dd"), log_dir)
    hist_b, front_b = run_nsga2(params, inst_path,
                                os.path.join(run_dir, "nsga2"), log_dir)

    fa1 = [p[0] for p in front_a]
    fa2 = [p[1] for p in front_a]
    fb1 = [p[0] for p in front_b]
    fb2 = [p[1] for p in front_b]

    # 共用参考点：取两前沿合并后的极值放大约 10%，保证 HV 可比
    all_f1 = fa1 + fb1
    all_f2 = fa2 + fb2
    ref1 = max(all_f1) * 1.1 + 1.0 if all_f1 else 1.0
    ref2 = max(all_f2) * 1.1 + 1.0 if all_f2 else 1.0

    hv_a = metrics.hypervolume_2d(fa1, fa2, ref1, ref2)
    hv_b = metrics.hypervolume_2d(fb1, fb2, ref1, ref2)
    nps_a = metrics.nps(fa1, fa2)
    nps_b = metrics.nps(fb1, fb2)
    c_ab = metrics.c_metric(fa1, fa2, fb1, fb2)
    c_ba = metrics.c_metric(fb1, fb2, fa1, fa2)

    log.info("---- [%s] 第 %d 次（seed=%d）----", inst_name, run_no, params.seed)
    log.info("  DEA-DD   ：NPS=%d  HV=%.4f", nps_a, hv_a)
    log.info("  NSGA-II  ：NPS=%d  HV=%.4f", nps_b, hv_b)
    log.info("  C(DEA-DD|NSGA-II)=%.3f   C(NSGA-II|DEA-DD)=%.3f", c_ab, c_ba)

    path = _plot_comparison(front_a, front_b, run_dir)
    return {
        "run": run_no, "seed": params.seed,
        "NPS_a": nps_a, "HV_a": hv_a,
        "NPS_b": nps_b, "HV_b": hv_b,
        "C_ab": c_ab, "C_ba": c_ba,
        "front_a": front_a, "front_b": front_b,
        "comparison_plot": path,
    }


def _summarize(results):
    """对多次运行结果做数值汇总（mean/std/min/max）。

    :param results: :func:`_compare_once` 的返回值列表
    :return: ``{指标名: {"mean"/"std"/"min"/"max": float}}``
    :rtype: dict
    """
    summary = {}
    for k in _METRIC_KEYS:
        vals = [r[k] for r in results]
        summary[k] = {
            "mean": float(statistics.mean(vals)),
            "std": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return summary


def _log_summary(log, summary, runs, inst_name):
    """把单个算例的汇总结果写入 ``compare`` 日志。

    :param log: ``compare`` 记录器
    :param summary: :func:`_summarize` 的返回值
    :param runs: 总运行次数
    :param inst_name: 算例显示名
    """
    log.info("========== [%s] 汇总（%d 次，mean ± std）==========",
             inst_name, runs)
    log.info("  DEA-DD    NPS = %.2f ± %.2f",
             summary["NPS_a"]["mean"], summary["NPS_a"]["std"])
    log.info("  NSGA-II   NPS = %.2f ± %.2f",
             summary["NPS_b"]["mean"], summary["NPS_b"]["std"])
    log.info("  DEA-DD    HV  = %.4f ± %.4f",
             summary["HV_a"]["mean"], summary["HV_a"]["std"])
    log.info("  NSGA-II   HV  = %.4f ± %.4f",
             summary["HV_b"]["mean"], summary["HV_b"]["std"])
    log.info("  C(DEA-DD|NSGA-II) = %.3f ± %.3f",
             summary["C_ab"]["mean"], summary["C_ab"]["std"])
    log.info("  C(NSGA-II|DEA-DD) = %.3f ± %.3f",
             summary["C_ba"]["mean"], summary["C_ba"]["std"])


def _write_summary_csv(results, summary, path):
    """把单个算例的逐次指标与汇总写入 CSV 文件。

    :param results: :func:`_compare_once` 的返回值列表
    :param summary: :func:`_summarize` 的返回值
    :param path: 目标 CSV 路径
    :return: 写入的 CSV 路径
    :rtype: str
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seed"] + list(_METRIC_KEYS))
        for r in results:
            w.writerow([r["run"], r["seed"]] +
                       [_METRIC_FMT[k].format(r[k]) for k in _METRIC_KEYS])
        w.writerow(["mean", ""] +
                   [_METRIC_FMT[k].format(summary[k]["mean"])
                    for k in _METRIC_KEYS])
        w.writerow(["std", ""] +
                   [_METRIC_FMT[k].format(summary[k]["std"])
                    for k in _METRIC_KEYS])
    return path


def _write_summary_all(instances, path):
    """把跨算例汇总（每个算例的均值）写入 CSV 文件。

    :param instances: :func:`_compare_instance` 的返回值列表
    :param path: 目标 CSV 路径
    :return: 写入的 CSV 路径
    :rtype: str
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["instance"] + list(_METRIC_KEYS))
        for inst in instances:
            s = inst["summary"]
            w.writerow([inst["instance"]] +
                       [_METRIC_FMT[k].format(s[k]["mean"])
                        for k in _METRIC_KEYS])
    return path


def _log_summary_all(log, instances):
    """把跨算例汇总打印到日志（每个算例一行，取该算例的均值）。"""
    log.info("========== 跨算例汇总（每个算例的 mean）==========")
    for inst in instances:
        s = inst["summary"]
        log.info("  %-18s HV(DEA-DD)=%.4f  HV(NSGA-II)=%.4f  "
                 "C(DEA-DD|NSGA-II)=%.3f",
                 inst["instance"],
                 s["HV_a"]["mean"], s["HV_b"]["mean"], s["C_ab"]["mean"])


def _compare_instance(params, inst_path, out_dir, log_dir, runs, log):
    """对单个算例重复跑 ``runs`` 次，结果隔离写入 ``out_dir``。

    :param params: 基准参数；第 i 次 seed = ``params.seed + i``
    :param inst_path: `.fjs` 文件路径（None 表示合成算例）
    :param out_dir: 该算例的隔离输出目录（``run_XXX/`` 与 ``summary.csv`` 都写这里）
    :param log_dir: 日志目录
    :param runs: 重复运行次数
    :param log: ``compare`` 记录器
    :return: ``{"instance"/"inst_path"/"runs"/"summary"}`` 的 dict
    :rtype: dict
    """
    inst_name = _instance_name(inst_path)
    log.info("========== 算例 %s：DEA-DD vs 传统 NSGA-II（共 %d 次）==========",
             inst_name, runs)
    results = []
    for i in range(runs):
        seed_i = params.seed + i
        p_i = dataclasses.replace(params, seed=seed_i)
        run_dir = os.path.join(out_dir, "run_{:03d}".format(i + 1))
        results.append(_compare_once(p_i, inst_path, run_dir, log_dir, log,
                                     inst_name, i + 1))

    summary = _summarize(results)
    _log_summary(log, summary, runs, inst_name)
    csv_path = _write_summary_csv(results, summary,
                                  os.path.join(out_dir, "summary.csv"))
    log.info("[%s] 汇总表已保存到 %s", inst_name, csv_path)
    return {"instance": inst_name, "inst_path": inst_path,
            "runs": results, "summary": summary}


def compare(params, inst_path=None, out_root="outputs", log_dir="logs", runs=10):
    """用相同参数对一或多个算例重复跑 DEA-DD 与 NSGA-II 对比。

    :param params: 共享的 :class:`~dea_dd.params.Params`（重叠参数一致）；
        第 i 次运行的 seed 取 ``params.seed + i``（i 从 0 到 runs-1）
    :param inst_path: `.fjs` 文件路径、或含 ``.fjs`` 的目录（**非递归**只扫顶层）、
        或 None（用 10×6 合成算例）
    :param out_root: 对比输出根目录；每个算例隔离到 ``<out_root>/<算例名>/run_XXX/``
        与 ``<out_root>/<算例名>/summary.csv``，多算例另写 ``<out_root>/summary_all.csv``
    :param log_dir: 日志目录
    :param runs: 每个算例重复运行次数（缺省 10，须 ≥ 1）
    :return: ``{"instances": [...], "summary_all": path|None}``；每个 instance
        含 ``instance/inst_path/runs/summary``，summary 为 mean/std 汇总
    :rtype: dict
    """
    if runs < 1:
        raise ValueError("runs 必须 ≥ 1，得到 " + str(runs))
    out_root = resolve_project_path(out_root)
    log_dir = resolve_project_path(log_dir)
    log = setup_logging(log_dir, name="compare")

    # 算例列表：目录→非递归扫描顶层 .fjs；文件→单文件；None→合成
    if inst_path is None:
        files = [None]
    else:
        files = _list_fjs_files(inst_path)

    instances = []
    for fp in files:
        name = _instance_name(fp)
        inst_out = os.path.join(out_root, name)
        instances.append(_compare_instance(params, fp, inst_out, log_dir, runs,
                                           log))

    summary_all = None
    if len(instances) > 1:
        summary_all = _write_summary_all(
            instances, os.path.join(out_root, "summary_all.csv"))
        _log_summary_all(log, instances)
        log.info("跨算例汇总表已保存到 %s", summary_all)

    return {"instances": instances, "summary_all": summary_all}


def main(argv=None, params=None, inst_path=None, runs=None):
    """命令行 / 程序化入口。

    命令行示例（对目录下所有算例各跑 10 次）：
        python -m dea_dd.compare --instance data/Brandimarte --gen 100 --runs 10

    程序化示例：
        from dea_dd.compare import main
        from dea_dd.params import Params
        main(params=Params(N=100, gen=100, R_short=20, R_long=500),
             inst_path="data/Brandimarte", runs=10)

    :param argv: 命令行参数列表
    :param params: 程序化传入的 :class:`~dea_dd.params.Params`
    :param inst_path: `.fjs` 文件或目录路径
    :param runs: 每个算例重复运行次数；缺省 10
    :return: :func:`compare` 的返回值
    :rtype: dict
    """
    if params is None:
        ap = argparse.ArgumentParser(
            description="对比 DEA-DD 与传统 NSGA-II 基线（可重复多次、可扫目录）")
        ap.add_argument("--instance", default=None,
                        help=".fjs 文件或目录（目录则非递归扫描顶层所有 .fjs）；"
                             "缺省用 10x6 合成算例")
        ap.add_argument("--ul", type=float, default=Params.UL)
        ap.add_argument("--pm", type=float, default=Params.p_m)
        ap.add_argument("--N", type=int, default=Params.N)
        ap.add_argument("--gen", type=int, default=Params.gen_max)
        ap.add_argument("--R_short", type=int, default=Params.R_short)
        ap.add_argument("--R_long", type=int, default=Params.R_long)
        ap.add_argument("--seed", type=int, default=Params.seed)
        ap.add_argument("--runs", type=int, default=10,
                        help="每个算例重复运行次数（缺省 10，每次 seed 递增 1）")
        ap.add_argument("--out", default="outputs")
        args = ap.parse_args(argv)
        params = Params(N=args.N, gen_max=args.gen, UL=args.ul,
                        R_short=args.R_short, R_long=args.R_long,
                        seed=args.seed, p_m=args.pm)
        inst_path = inst_path if inst_path is not None else args.instance
        runs = runs if runs is not None else args.runs
        return compare(params, inst_path, out_root=args.out, runs=runs)
    return compare(params, inst_path, runs=runs if runs is not None else 10)


if __name__ == "__main__":
    main(inst_path="../data/Brandimarte", runs=2)
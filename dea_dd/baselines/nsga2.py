"""传统 NSGA-II 基线（用于与 DEA-DD 做性能对比）。

与 DEA-DD 的差异（即去掉 DEA-DD 的创新点）：
  * 无 EO 预热（创新①）——直接随机初始化；
  * 无局部搜索（创新③④）——只用标准遗传算子；
  * 固定变异位数（创新⑤）——不做式(16) 动态变异，固定改 2 位。

与 DEA-DD 完全共享：MS+OS 编码、双解码（创新②）、随机口径蒙特卡洛评估
（对数正态 + CRN）、非支配排序 + 归一化拥挤距离 + 精英保留，以及重叠参数
``N / p_c / p_m / gen_max / UL / R_short / R_long / seed``。父代选择采用标准
二元锦标赛（两个父代均锦标赛，区别于 DEA-DD 的“一锦标赛 + 一随机”）。
"""
from __future__ import annotations

# 支持直接以脚本方式运行（如 `python dea_dd/baselines/nsga2.py`）：
# 此时 __package__ 为空，需把工程根目录加入 sys.path 才能做绝对导入。
import os
import sys

if __package__ in ("", None):
    _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import argparse
import time

import numpy as np

from dea_dd.core.instance import parse_fjs, generate_due_dates
from dea_dd.core.normalization import Normalizer
from dea_dd.core.encoding import random_chromosome
from dea_dd.core.decoding import decode_evaluate
from dea_dd.core.pareto import dominated, non_dominated_indices
from dea_dd.algorithms.nsga2 import (Individual, pox_crossover, ux_crossover,
                                      mutate_os, mutate_ms, environmental_selection)
from dea_dd.simulation.stochastic import (LognormalProcessing, mc_evaluate,
                                          mc_evaluate_long)
from dea_dd.diagnostics import metrics, plots
from dea_dd.diagnostics.logging_setup import setup_logging
from dea_dd.diagnostics.generator import random_instance
from dea_dd.params import Params
from dea_dd.paths import resolve_project_path

# 传统 NSGA-II 的固定变异位数（对应《算法设计.md》§12.3 消融“固定变异 2 位”）：
# DEA-DD 用式(16) 动态变异位数（创新⑤），此基线固定为 2。
MUT_BITS = 2


def _load_instance(path):
    """加载 `.fjs` 算例（与 ``dea_dd.main`` 同款回退逻辑）。

    :param path: `.fjs` 文件路径；为 ``None`` 或空串时回退到 10×6 合成算例
    :return: 解析得到的 FJSP 实例
    :rtype: dea_dd.core.instance.Instance
    """
    if path is None or path == "":
        return random_instance(10, 6, min_ops=5, max_ops=6, seed=1)
    with open(path, "r", encoding="utf-8") as f:
        return parse_fjs(f.read())


def _tournament_select(P, rng):
    """二元锦标赛选择（帕累托支配优先，平局随机）。

    :param P: 候选种群（Individual 列表）
    :param rng: 随机数发生器
    :return: 胜出的个体
    :rtype: dea_dd.algorithms.nsga2.Individual
    """
    a = P[int(rng.integers(0, len(P)))]
    b = P[int(rng.integers(0, len(P)))]
    if dominated(a.f1, a.f2, b.f1, b.f2):
        return a
    if dominated(b.f1, b.f2, a.f1, a.f2):
        return b
    return a if rng.random() < 0.5 else b


def run_nsga2(params, inst_path=None, out_dir="outputs/nsga2", log_dir="logs"):
    """程序化执行入口：运行传统 NSGA-II 基线并返回 ``(history, final_points)``。

    流程与 DEA-DD 阶段二/三对齐：随机初始化 → 逐代交叉/变异 → 环境选择，
    最后对最终种群每个解做长仿真 (R_long)，再按长仿真期望非支配排序得前沿。

    :param params: :class:`~dea_dd.params.Params` 参数对象（重叠参数与 DEA-DD
        保持一致；``N_eo/Iter_EO/K/weights`` 等 EO 专属参数在此不生效）
    :param inst_path: `.fjs` 文件路径；缺省用 10×6 合成算例
    :param out_dir: 图像输出目录（缺省 ``outputs/nsga2``）
    :param log_dir: 日志目录（缺省 ``logs``）
    :return: 元组 ``(history, final_points)``，``final_points`` 为长仿真后的
        ``[(E_f1, E_f2, std_f1, std_f2), ...]``
    :rtype: tuple
    """
    out_dir = resolve_project_path(out_dir)
    log_dir = resolve_project_path(log_dir)
    log = setup_logging(log_dir, name="nsga2")
    rng = np.random.default_rng(params.seed)
    inst = _load_instance(inst_path)
    due = generate_due_dates(inst, rng=rng)
    total_ops = inst.total_ops
    N = params.N

    log.info("[NSGA-II 基线] 实例：%d 工件 × %d 机器，%d 道工序，UL=%.2f",
             inst.n_jobs, inst.n_machines, total_ops, params.UL)
    history = {}

    # ---- 随机初始化（无 EO 预热） ----------------------------------------
    processor = LognormalProcessing(inst, params.UL, rng)
    normalizer = Normalizer()
    scenarios = processor.sample_scenarios(params.R_short)
    P = []
    for _ in range(N):
        ms, os = random_chromosome(inst, rng)
        f1, f2 = mc_evaluate(inst, ms, os, scenarios, due)
        normalizer.update(f1, f2)
        P.append(Individual(ms, os, f1, f2))

    f1s0 = [ind.f1 for ind in P]
    f2s0 = [ind.f2 for ind in P]
    idx0 = non_dominated_indices(f1s0, f2s0)
    history["initial_front"] = [(f1s0[i], f2s0[i]) for i in idx0]
    log.info("[NSGA-II 基线] 随机初始化 %d 个个体，初始非支配前沿 %d 点",
             N, len(idx0))

    # ---- 主循环：标准 NSGA-II（固定变异，无局部搜索） ----------------------
    nsga2_conv = []
    t0 = time.time()
    for gen in range(1, params.gen_max + 1):
        Q = []
        for _ in range(N // 2):
            # 两个父代都采用二元锦标赛选择（传统 NSGA-II 做法）
            p1 = _tournament_select(P, rng)
            p2 = _tournament_select(P, rng)
            if rng.random() < params.p_c:
                c1os, c2os = pox_crossover(p1.os, p2.os, inst.n_jobs, rng)
                c1ms, c2ms = ux_crossover(p1.ms, p2.ms, params.p_c, rng)
            else:
                c1os, c2os = list(p1.os), list(p2.os)
                c1ms, c2ms = list(p1.ms), list(p2.ms)
            # 固定变异位数（非动态），每个子代以概率 p_m 执行变异
            if rng.random() < params.p_m:
                c1os = mutate_os(c1os, MUT_BITS, rng)
                c1ms = mutate_ms(c1ms, inst, MUT_BITS, rng)
            if rng.random() < params.p_m:
                c2os = mutate_os(c2os, MUT_BITS, rng)
                c2ms = mutate_ms(c2ms, inst, MUT_BITS, rng)
            Q.append(Individual(c1ms, c1os, 0.0, 0.0))
            Q.append(Individual(c2ms, c2os, 0.0, 0.0))

        # CRN：同一迭代内 Q 复用同一组场景
        scenarios = processor.sample_scenarios(params.R_short)
        for ind in Q:
            ind.f1, ind.f2 = mc_evaluate(inst, ind.ms, ind.os, scenarios, due)
            normalizer.update(ind.f1, ind.f2)

        P = environmental_selection(P, Q, normalizer, N)
        best_f1 = min(ind.f1 for ind in P)
        best_f2 = min(ind.f2 for ind in P)
        nsga2_conv.append((gen, best_f1, best_f2))
        if gen % 50 == 0 or gen == params.gen_max:
            log.info("[NSGA-II 基线] 第 %d/%d 代：最优 makespan=%.2f，TWTE=%.2f",
                     gen, params.gen_max, best_f1, best_f2)
    history["nsga2_conv"] = nsga2_conv
    log.info("[NSGA-II 基线] 主循环完成，耗时 %.1fs", time.time() - t0)

    # ---- 最终前沿：先长仿真，再按长仿真期望非支配排序 ----------------------
    long_f1, long_f2, long_std = [], [], []
    for ind in P:
        m1, m2, s1, s2 = mc_evaluate_long(inst, ind.ms, ind.os, processor,
                                          due, params.R_long)
        long_f1.append(m1)
        long_f2.append(m2)
        long_std.append((s1, s2))
    front_idx = non_dominated_indices(long_f1, long_f2)
    log.info("[NSGA-II 基线] 最终生成候选 %d，长仿真非支配前沿 %d 点",
             len(P), len(front_idx))
    final_points = []
    for i in front_idx:
        ind = P[i]
        m1, m2 = long_f1[i], long_f2[i]
        s1, s2 = long_std[i]
        final_points.append((m1, m2, s1, s2))
        d1, d2 = decode_evaluate(inst, ind.ms, ind.os, due)
        log.info("[NSGA-II 基线] 前沿解：E[makespan]=%.2f(std %.2f) "
                 "E[TWTE]=%.2f(std %.2f) gap1=%.3f gap2=%.3f",
                 m1, s1, m2, s2,
                 metrics.robustness_gap(m1, d1),
                 metrics.robustness_gap(m2, d2))

    final_f1 = [p[0] for p in final_points]
    final_f2 = [p[1] for p in final_points]
    history["final_front"] = [(a, b) for a, b, _, _ in final_points]

    # ---- 指标与绘图 -------------------------------------------------------
    ref1 = max(final_f1) * 1.1 + 1.0
    ref2 = max(final_f2) * 1.1 + 1.0
    hv = metrics.hypervolume_2d(final_f1, final_f2, ref1, ref2)
    nps_val = metrics.nps(final_f1, final_f2)
    c_init = metrics.c_metric(final_f1, final_f2,
                              [p[0] for p in history["initial_front"]],
                              [p[1] for p in history["initial_front"]])
    log.info("[NSGA-II 基线] 指标：HV=%.4f NPS=%d C(final|initial)=%.3f",
             hv, nps_val, c_init)

    plots.plot_nsga2_convergence(nsga2_conv, out_dir)
    plots.plot_final_front(history["final_front"], out_dir)
    log.info("[NSGA-II 基线] 图像已保存到 %s", out_dir)
    return history, final_points


def main_nsga2(argv=None, params=None, inst_path=None, out_dir="outputs/nsga2",
               log_dir="logs"):
    """命令行 / 程序化入口（与传统 NSGA-II 基线对应的命令行参数）。

    命令行示例：
        python -m dea_dd.baselines.nsga2 --instance data/Brandimarte/BrandimarteMk1.fjs --gen 100

    程序化示例（与 DEA-DD 共用同一套 Params，保证重叠参数一致）：
        from dea_dd.baselines.nsga2 import main_nsga2
        from dea_dd.params import Params
        main_nsga2(params=Params(N=100, gen=400, R_long=10000),
                   inst_path="data/Brandimarte/BrandimarteMk1.fjs")

    :param argv: 命令行参数列表；缺省解析 ``sys.argv[1:]``（仅当 ``params`` 为
        ``None`` 时使用）
    :param params: 程序化传入的 :class:`~dea_dd.params.Params`
    :param inst_path: 算例路径，覆盖命令行 ``--instance``
    :param out_dir: 图像输出目录（程序化调用时使用）
    :param log_dir: 日志目录（程序化调用时使用）
    :return: :func:`run_nsga2` 的返回值 ``(history, final_points)``
    :rtype: tuple
    """
    if params is None:
        ap = argparse.ArgumentParser(
            description="传统 NSGA-II 基线：双目标 FJSP")
        ap.add_argument("--instance", default=None,
                        help=".fjs 文件路径（缺省用 10x6 合成算例）")
        ap.add_argument("--ul", type=float, default=Params.UL)
        ap.add_argument("--pm", type=float, default=Params.p_m,
                        help="变异概率 p_m（缺省 0.1）")
        ap.add_argument("--N", type=int, default=Params.N)
        ap.add_argument("--gen", type=int, default=Params.gen_max)
        ap.add_argument("--R_short", type=int, default=Params.R_short)
        ap.add_argument("--R_long", type=int, default=Params.R_long)
        ap.add_argument("--seed", type=int, default=Params.seed)
        ap.add_argument("--out", default="outputs/nsga2")
        ap.add_argument("--logs", default="logs")
        args = ap.parse_args(argv)
        params = Params(N=args.N, gen_max=args.gen, UL=args.ul,
                        R_short=args.R_short, R_long=args.R_long,
                        seed=args.seed, p_m=args.pm)
        inst_path = inst_path if inst_path is not None else args.instance
        out_dir = args.out
        log_dir = args.logs
    return run_nsga2(params, inst_path, out_dir, log_dir)


if __name__ == "__main__":
    main_nsga2()
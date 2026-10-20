"""DEA-DD 主入口（《算法设计.md》§10 总伪代码）。

三个阶段：(1) 固定加工口径下的多 w EO 预热；(2) 随机加工口径下的改进 NSGA-II
（蒙特卡洛 R_short，CRN），带延迟触发的自适应局部搜索；(3) 对最终前沿做长仿真
(R_long)，随后输出指标与 5 张诊断图。

执行方式（需求 #4）：
  - 命令行：python -m dea_dd.main --instance data/Brandimarte/BrandimarteMk1.fjs --gen 400
  - 程序化：main(params=Params(N=100, gen=400, ...), inst_path="...")
"""
from __future__ import annotations

# 支持直接以脚本方式运行（如 PyCharm 右键 Run / `python dea_dd/main.py`）：
# 此时 __package__ 为空，需把工程根目录加入 sys.path 才能做绝对导入。
if __package__ in ("", None):
    import os
    import sys

    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

import argparse
import time

import numpy as np

from dea_dd.core.instance import parse_fjs, generate_due_dates
from dea_dd.core.normalization import Normalizer
from dea_dd.core.decoding import decode_evaluate
from dea_dd.core.pareto import dominated, non_dominated_indices
from dea_dd.algorithms.preheat import preheat
from dea_dd.algorithms.nsga2 import (Individual, pox_crossover, ux_crossover,
                                      dynamic_mutation_bits, mutate_os, mutate_ms,
                                      environmental_selection)
from dea_dd.algorithms.local_search import AdaptiveLocalSearch
from dea_dd.simulation.stochastic import (LognormalProcessing, mc_evaluate,
                                          mc_evaluate_long)
from dea_dd.diagnostics import metrics, plots
from dea_dd.diagnostics.logging_setup import setup_logging
from dea_dd.diagnostics.generator import random_instance
from dea_dd.params import Params


def _load_instance(path):
    """加载 `.fjs` 算例。

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


def _distinct_count(items):
    """按 ``(ms, os)`` 去重后的个数。

    :param items: ``(ms, os)`` 元组序列
    :return: 不重复解个数
    :rtype: int
    """
    seen = set()
    for ms, os in items:
        seen.add((tuple(ms), tuple(os)))
    return len(seen)


def run(params, inst_path=None, out_dir="output", log_dir="logs"):
    """程序化执行入口：传入 Params 对象运行完整三阶段算法。

    :param params: :class:`~dea_dd.params.Params` 参数对象（K 会在此统一钳制到
        ``≤ N / |W|``）
    :param inst_path: `.fjs` 文件路径；缺省用 10×6 合成算例
    :param out_dir: 图像输出目录（缺省 ``output``）
    :param log_dir: 日志目录（缺省 ``logs``）
    :return: 元组 ``(history, final_points)``，history 含各阶段收敛序列与前沿，
        final_points 为长仿真后的 ``[(E_f1, E_f2, std_f1, std_f2), ...]``
    :rtype: tuple
    """
    # 无论命令行还是程序化调用，都统一钳制 K ≤ N/|W|（§11）
    params.K = min(max(1, params.K), max(1, params.N // len(params.weights)))

    log = setup_logging(log_dir)
    rng = np.random.default_rng(params.seed)
    inst = _load_instance(inst_path)
    due = generate_due_dates(inst, rng=rng)
    total_ops = inst.total_ops

    log.info("实例：%d 工件 × %d 机器，%d 道工序，UL=%.2f",
             inst.n_jobs, inst.n_machines, total_ops, params.UL)
    history = {}

    # ---- 阶段一：多 w EO 预热（固定加工口径） -----------------------------
    normalizer = Normalizer()
    t0 = time.time()
    seeds = preheat(inst, due, params, normalizer, rng, history=history)
    f1s0 = [s[2] for s in seeds]
    f2s0 = [s[3] for s in seeds]
    idx0 = non_dominated_indices(f1s0, f2s0)
    history["initial_front"] = [(f1s0[i], f2s0[i]) for i in idx0]
    history["initial_front_ws"] = [history["seed_ws"][i] for i in idx0]
    log.info("阶段一完成：%d 个种子，耗时 %.1fs", len(seeds), time.time() - t0)

    # ---- 阶段二：改进 NSGA-II（随机口径，蒙特卡洛 R_short） ----------------
    processor = LognormalProcessing(inst, params.UL, rng)
    normalizer = Normalizer()          # §9.3：进入随机口径后重建 min/max 基座
    scenarios = processor.sample_scenarios(params.R_short)
    P = []
    for ms, os, _fd1, _fd2 in seeds:
        f1, f2 = mc_evaluate(inst, ms, os, scenarios, due)
        normalizer.update(f1, f2)
        P.append(Individual(ms, os, f1, f2))

    ls = AdaptiveLocalSearch(inst, due, params, rng)
    nsga2_conv = []
    N = params.N
    if len(P) > N:
        P = P[:N]
    t1 = time.time()
    for gen in range(1, params.gen_max + 1):
        k_mut = dynamic_mutation_bits(gen, params.gen_max, total_ops,
                                      params.delta_max, params.delta_min)
        Q = []
        for _ in range(N // 2):
            p1 = _tournament_select(P, rng)
            p2 = _tournament_select(P, rng)
            if rng.random() < params.p_c:
                c1os, c2os = pox_crossover(p1.os, p2.os, inst.n_jobs, rng)
                c1ms, c2ms = ux_crossover(p1.ms, p2.ms, params.p_c, rng)
            else:
                c1os, c2os = list(p1.os), list(p2.os)
                c1ms, c2ms = list(p1.ms), list(p2.ms)
            c1os = mutate_os(c1os, k_mut, rng)
            c2os = mutate_os(c2os, k_mut, rng)
            c1ms = mutate_ms(c1ms, inst, k_mut, rng)
            c2ms = mutate_ms(c2ms, inst, k_mut, rng)
            Q.append(Individual(c1ms, c1os, 0.0, 0.0))
            Q.append(Individual(c2ms, c2os, 0.0, 0.0))

        # CRN：同一迭代内，Q 与局部搜索复用同一组场景
        scenarios = processor.sample_scenarios(params.R_short)
        for ind in Q:
            ind.f1, ind.f2 = mc_evaluate(inst, ind.ms, ind.os, scenarios, due)
            normalizer.update(ind.f1, ind.f2)

        # 延迟触发的自适应局部搜索（式18：gen ≥ 0.8·gen_max）
        if gen >= params.ls_trigger * params.gen_max:
            for ind in Q:
                op = ls.pick_operator()
                new_ms, new_os = ls.generate(op, ind.ms, ind.os)
                if new_ms is None:
                    ls.record(op, False)
                    continue
                nf1, nf2 = mc_evaluate(inst, new_ms, new_os, scenarios, due)
                # 成功 = 结果不被父代支配（帕累托非支配改进）
                if not dominated(ind.f1, ind.f2, nf1, nf2):
                    ind.ms, ind.os, ind.f1, ind.f2 = new_ms, new_os, nf1, nf2
                    normalizer.update(nf1, nf2)
                    ls.record(op, True)
                else:
                    ls.record(op, False)

        P = environmental_selection(P, Q, normalizer, N)
        best_f1 = min(ind.f1 for ind in P)
        best_f2 = min(ind.f2 for ind in P)
        nsga2_conv.append((gen, best_f1, best_f2))
        if gen % 50 == 0 or gen == params.gen_max:
            log.info("第 %d/%d 代：最优 makespan=%.2f，TWTE=%.2f",
                     gen, params.gen_max, best_f1, best_f2)
    history["nsga2_conv"] = nsga2_conv
    log.info("阶段二完成，耗时 %.1fs", time.time() - t1)

    # ---- 阶段三：非支配前沿长仿真 -----------------------------------------
    f1s = [ind.f1 for ind in P]
    f2s = [ind.f2 for ind in P]
    front_idx = non_dominated_indices(f1s, f2s)
    stars = [P[i] for i in front_idx]
    log.info("NSGA-II 阶段最终生成 %d 个不重复解（非支配前沿）",
             _distinct_count([(s.ms, s.os) for s in stars]))
    final_points = []
    for ind in stars:
        m1, m2, s1, s2 = mc_evaluate_long(inst, ind.ms, ind.os, processor,
                                          due, params.R_long)
        final_points.append((m1, m2, s1, s2))
        d1, d2 = decode_evaluate(inst, ind.ms, ind.os, due)
        log.info("前沿解：E[makespan]=%.2f(std %.2f) E[TWTE]=%.2f(std %.2f) "
                 "gap1=%.3f gap2=%.3f",
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
    log.info("指标：HV=%.4f NPS=%d C(final|initial)=%.3f", hv, nps_val, c_init)

    plots.plot_eo_convergence(history["preheat_conv"], out_dir)
    plots.plot_eo_w_fronts(history["preheat_w_fronts"], out_dir)
    plots.plot_initial_front(history["initial_front"], out_dir,
                             ws=history["initial_front_ws"])
    plots.plot_nsga2_convergence(nsga2_conv, out_dir)
    plots.plot_final_front(history["final_front"], out_dir)
    log.info("图像已保存到 %s", out_dir)
    return history, final_points


def main(argv=None, params=None, inst_path=None, out_dir="output", log_dir="logs"):
    """命令行 / 程序化入口。

    命令行示例：
        python -m dea_dd.main --instance data/Brandimarte/BrandimarteMk1.fjs --gen 100

    程序化示例（需求 #4：直接在代码里传参）：
        from dea_dd.main import main
        from dea_dd.params import Params
        main(params=Params(N=100, gen=400, R_long=10000),
             inst_path="data/Brandimarte/BrandimarteMk1.fjs")

    :param argv: 命令行参数列表；缺省解析 ``sys.argv[1:]``（仅当 ``params`` 为
        ``None`` 时使用）
    :param params: 程序化传入的 :class:`~dea_dd.params.Params`；为 ``None`` 时
        从命令行参数构造
    :param inst_path: 算例路径，覆盖命令行 ``--instance``
    :param out_dir: 图像输出目录（程序化调用时使用）
    :param log_dir: 日志目录（程序化调用时使用）
    :return: :func:`run` 的返回值 ``(history, final_points)``
    :rtype: tuple
    """
    if params is None:
        ap = argparse.ArgumentParser(description="DEA-DD：双目标 FJSP")
        ap.add_argument("--instance", default=None,
                        help=".fjs 文件路径（缺省用 10x6 合成算例）")
        ap.add_argument("--ul", type=float, default=Params.UL)
        ap.add_argument("--N", type=int, default=Params.N)
        ap.add_argument("--N_eo", type=int, default=Params.N_eo)
        ap.add_argument("--Iter_EO", type=int, default=Params.Iter_EO)
        ap.add_argument("--gen", type=int, default=Params.gen_max)
        ap.add_argument("--R_short", type=int, default=Params.R_short)
        ap.add_argument("--R_long", type=int, default=Params.R_long)
        ap.add_argument("--seed", type=int, default=Params.seed)
        ap.add_argument("--out", default="output")
        ap.add_argument("--logs", default="logs")
        args = ap.parse_args(argv)
        params = Params(N=args.N, N_eo=args.N_eo, Iter_EO=args.Iter_EO,
                        gen_max=args.gen, UL=args.ul, R_short=args.R_short,
                        R_long=args.R_long, seed=args.seed)
        inst_path = inst_path if inst_path is not None else args.instance
        out_dir = args.out
        log_dir = args.logs
    return run(params, inst_path, out_dir, log_dir)


if __name__ == "__main__":
    main()
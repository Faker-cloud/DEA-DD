"""阶段一：多 w 切比雪夫分解预热（创新①）。

按指定顺序（两端优先）对每个 w 运行 EO，按归一化目标空间中的拥挤距离保留
每个 w 的前 K 个种子，合并去重，再用非支配排序 + 拥挤截断到 N（§5.3）；
若去重后仍不足 N，则随机生成新解补齐到 N。

额外的日志与统计：记录每个 w 生成的不重复解个数、阶段一最终不重复初始解个数，
并保存各种子来源的权重 w（供初始前沿按 w 连线；随机补齐的解 w 记为 None）。
"""
from __future__ import annotations

import logging
from typing import Dict, List

from dea_dd.algorithms.eo import EOSolver
from dea_dd.core.decoding import decode_evaluate
from dea_dd.core.encoding import random_chromosome
from dea_dd.core.pareto import crowding_distance, fast_non_dominated_sort

logger = logging.getLogger("dea_dd")


def _distinct(pop):
    """按 ``(ms, os)`` 去重。

    :param pop: 候选列表，每项为 ``(ms, os, f1, f2[, ...])``
    :return: 去重后的候选列表（保持首次出现顺序）
    :rtype: list
    """
    seen = set()
    out = []
    for item in pop:
        key = (tuple(item[0]), tuple(item[1]))
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _front_points(pop):
    """返回候选集的非支配前沿点 ``(f1, f2)``。

    :param pop: 候选列表，每项为 ``(ms, os, f1, f2[, ...])``
    :return: 非支配前沿的 ``[(f1, f2), ...]`` 列表
    :rtype: list
    """
    pop = _distinct(pop)
    if not pop:
        return []
    f1s = [c[2] for c in pop]
    f2s = [c[3] for c in pop]
    fronts = fast_non_dominated_sort(f1s, f2s)
    return [(f1s[i], f2s[i]) for i in fronts[0]]


def _select_top_k_by_crowding(candidates, norm, K):
    """按归一化拥挤距离取前 K 个（多样性优先，而非 g 最小）。

    :param candidates: 候选列表，每项为 ``(ms, os, f1, f2[, ...])``
    :param norm: 归一化器
    :param K: 保留个数
    :return: 拥挤距离最大的前 K 个候选
    :rtype: list
    """
    pop = _distinct(candidates)
    if len(pop) <= K:
        return pop
    f1s = [c[2] for c in pop]
    f2s = [c[3] for c in pop]
    n1 = []
    n2 = []
    for a, b in zip(f1s, f2s):
        x, y = norm.normalize(a, b)
        n1.append(x)
        n2.append(y)
    cd = crowding_distance(n1, n2)
    order = sorted(range(len(pop)), key=lambda i: -cd[i])
    return [pop[i] for i in order[:K]]


def _truncate_front(pop, norm, N):
    """非支配排序 + 拥挤距离截断到 N。

    :param pop: 候选列表，每项为 ``(ms, os, f1, f2[, ...])``
    :param norm: 归一化器
    :param N: 目标规模
    :return: 截断后的候选列表（长度不超过 N）
    :rtype: list
    """
    f1s = [c[2] for c in pop]
    f2s = [c[3] for c in pop]
    fronts = fast_non_dominated_sort(f1s, f2s)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= N:
            selected.extend(front)
        else:
            need = N - len(selected)
            n1 = []
            n2 = []
            for i in front:
                a, b = norm.normalize(f1s[i], f2s[i])
                n1.append(a)
                n2.append(b)
            cd = crowding_distance(n1, n2)
            order = sorted(range(len(front)), key=lambda k: -cd[k])
            selected.extend([front[k] for k in order[:need]])
            break
    return [pop[i] for i in selected]


def _fill_random(seeds, inst, due_dates, N, rng):
    """随机生成新解补齐到 N 个（预热去重后不足 N 时调用）。

    随机补齐的解用固定口径 ``decode_evaluate`` 评估（与阶段一一致），来源 w 记为
    ``None``；带重试上限以防极小搜索空间下无法凑满 N。

    :param seeds: 现有种子列表，每项 ``(ms, os, f1, f2[, w])``
    :param inst: FJSP 实例
    :param due_dates: 交货期数组
    :param N: 目标规模
    :param rng: 随机数发生器
    :return: 补齐后的种子列表（随机补齐的解 w 记为 ``None``）
    :rtype: list
    """
    seen = {(tuple(s[0]), tuple(s[1])) for s in seeds}
    filled = []
    max_attempts = max(100, (N - len(seeds)) * 200)
    attempts = 0
    while len(seeds) + len(filled) < N and attempts < max_attempts:
        attempts += 1
        ms, os = random_chromosome(inst, rng)
        key = (tuple(ms), tuple(os))
        if key in seen:
            continue
        seen.add(key)
        f1, f2 = decode_evaluate(inst, ms, os, due_dates)
        filled.append((ms, os, f1, f2, None))
    return seeds + filled


def preheat(inst, due_dates, params, normalizer, rng, history=None):
    """运行阶段一预热。

    :param inst: FJSP 实例
    :param due_dates: 各工件交货期
    :param params: 参数对象（决定权重顺序、N_eo、Iter_EO、K、N）
    :param normalizer: 全局共享的归一化器（阶段一固定口径）
    :param rng: 随机数发生器
    :param history: 可选，字典；传入时写入 ``preheat_conv``/``preheat_w_fronts``/
        ``seed_ws``/``eo_distinct_per_w`` 供日志与绘图使用
    :return: 种子列表，每项为 ``(ms, os, f1, f2)`` 四元组；长度不超过 N，
        去重后不足 N 时随机补齐
    :rtype: list
    """
    seeds = []               # 内部用五元组 (ms, os, f1, f2, w) 以便追踪来源 w
    w_fronts = {}
    conv = {}
    distinct_per_w = {}
    total_candidates = 0
    for w in params.weights:
        solver = EOSolver(inst, w, normalizer, params, rng, due_dates)
        candidates, conv_w = solver.run()
        conv[w] = conv_w
        w_fronts[w] = _front_points(candidates)
        distinct_per_w[w] = len(_distinct(candidates))
        total_candidates += len(candidates)
        logger.info("EO[w=%.2f]：候选 %d 个，其中不重复 %d 个",
                    w, len(candidates), distinct_per_w[w])
        topk = _select_top_k_by_crowding(candidates, normalizer, params.K)
        seeds.extend((m, o, f1, f2, w) for (m, o, f1, f2) in topk)

    seeds = _distinct(seeds)
    logger.info("EO 阶段合并去重后共 %d 个不重复解（候选总数 %d）",
                len(seeds), total_candidates)
    if len(seeds) > params.N:
        seeds = _truncate_front(seeds, normalizer, params.N)
    elif len(seeds) < params.N:
        before = len(seeds)
        seeds = _fill_random(seeds, inst, due_dates, params.N, rng)
        logger.info("EO 阶段去重后 %d 个 < N=%d，随机补齐 %d 个",
                    before, params.N, len(seeds) - before)
    logger.info("EO 阶段最终保留 %d 个初始解（目标 N=%d）",
                len(seeds), params.N)

    seed_ws = [w for (_m, _o, _f1, _f2, w) in seeds]
    final_seeds = [(m, o, f1, f2) for (m, o, f1, f2, w) in seeds]

    if history is not None:
        history["preheat_conv"] = conv
        history["preheat_w_fronts"] = w_fronts
        history["seed_ws"] = seed_ws
        history["eo_distinct_per_w"] = distinct_per_w
    return final_seeds
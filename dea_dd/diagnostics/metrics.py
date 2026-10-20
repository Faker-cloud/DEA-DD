"""评价指标（《算法设计.md》§12.2）：HV、IGD、C-metric、NPS 与稳健性 gap
Delta_o = (f_s - f_d) / f_d。"""
from __future__ import annotations

from typing import List

import numpy as np

from dea_dd.core.pareto import dominated, non_dominated_indices


def hypervolume_2d(f1s, f2s, ref1, ref2):
    """相对参考点 ``(ref1, ref2)`` 的超体积（目标均最小化）。

    :param f1s: 前沿各点 makespan 列表
    :param f2s: 前沿各点 TWTE 列表（与 f1s 对齐）
    :param ref1: 参考点 makespan 分量（应不小于前沿各点的 f1）
    :param ref2: 参考点 TWTE 分量（应不小于前沿各点的 f2）
    :return: 二维超体积（仅统计落在参考点内、且非支配的点）
    :rtype: float
    """
    pts = [(float(f1s[i]), float(f2s[i])) for i in range(len(f1s))
           if f1s[i] <= ref1 and f2s[i] <= ref2]
    if not pts:
        return 0.0
    pts = sorted(set(pts))            # 按 f1 升序
    # 保留非支配前沿（f2 随 f1 升序严格递减）
    front = []
    best_f2 = float("inf")
    for a, b in pts:
        if b < best_f2:
            front.append((a, b))
            best_f2 = b
    hv = 0.0
    for i, (a, b) in enumerate(front):
        next_a = front[i + 1][0] if i + 1 < len(front) else ref1
        hv += (next_a - a) * (ref2 - b)
    return hv


def c_metric(a1, a2, b1, b2):
    """集合 A 对集合 B 的覆盖率：B 中被 A 某点支配的点所占比例。

    :param a1: 集合 A 的 makespan 列表
    :param a2: 集合 A 的 TWTE 列表
    :param b1: 集合 B 的 makespan 列表
    :param b2: 集合 B 的 TWTE 列表
    :return: C(A, B) ∈ [0, 1]，被 A 至少一点支配的 B 点占比
    :rtype: float
    """
    if not b1:
        return 0.0
    cnt = 0
    for j in range(len(b1)):
        if any(dominated(a1[i], a2[i], b1[j], b2[j]) for i in range(len(a1))):
            cnt += 1
    return cnt / len(b1)


def nps(f1s, f2s):
    """非支配解的个数。

    :param f1s: makespan 列表
    :param f2s: TWTE 列表（与 f1s 对齐）
    :return: 非支配前沿（front 0）中解的个数
    :rtype: int
    """
    return len(non_dominated_indices(list(f1s), list(f2s)))


def igd(front1, front2, ref1, ref2):
    """``front`` 相对参考集 ``(ref1, ref2)`` 的 IGD。

    返回每个参考点到最近前沿点欧氏距离的均值。需要外部参考前沿（如基线运行
    合并得到的最优已知前沿）。

    :param front1: 前沿解的 makespan 列表
    :param front2: 前沿解的 TWTE 列表
    :param ref1: 参考前沿的 makespan 列表
    :param ref2: 参考前沿的 TWTE 列表
    :return: IGD 值；参考集为空或前沿为空时返回 +inf
    :rtype: float
    """
    if not ref1 or not front1:
        return float("inf")
    total = 0.0
    for r in range(len(ref1)):
        dmin = float("inf")
        for i in range(len(front1)):
            d = (front1[i] - ref1[r]) ** 2 + (front2[i] - ref2[r]) ** 2
            if d < dmin:
                dmin = d
        total += np.sqrt(dmin)
    return total / len(ref1)


def robustness_gap(f_stochastic, f_deterministic):
    """Delta_o = ``(f_s - f_d) / f_d``（f_d 为 0 时返回 0）。

    :param f_stochastic: 随机口径下的目标值（如 MC 期望）
    :param f_deterministic: 确定性口径下的目标值
    :return: 稳健性相对间隙
    :rtype: float
    """
    if f_deterministic <= 0:
        return 0.0
    return (f_stochastic - f_deterministic) / f_deterministic
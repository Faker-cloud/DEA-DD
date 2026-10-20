"""帕累托工具：支配、快速非支配排序、拥挤距离。

用于阶段一种子截断（§5.3）、阶段二环境选择（§6.3）以及最终前沿（§9.5）。
拥挤距离始终在**归一化**目标坐标上计算（§6.3 备注）。
"""
from __future__ import annotations

from typing import Dict, List


def dominated(a_f1: float, a_f2: float, b_f1: float, b_f2: float) -> bool:
    """判断 a 是否帕累托支配 b（两个目标均最小化）。

    :param a_f1: 点 a 的 f1(makespan)
    :param a_f2: 点 a 的 f2(TWTE)
    :param b_f1: 点 b 的 f1(makespan)
    :param b_f2: 点 b 的 f2(TWTE)
    :return: a 支配 b 则为 True（a 两个目标都不差于 b，且至少一个严格更优）
    :rtype: bool
    """
    return (a_f1 <= b_f1 and a_f2 <= b_f2) and (a_f1 < b_f1 or a_f2 < b_f2)


def fast_non_dominated_sort(f1: List[float], f2: List[float]) -> List[List[int]]:
    """按支配层级返回前沿（front 0 为非支配前沿）。

    :param f1: 各点 makespan 列表
    :param f2: 各点 TWTE 列表（与 f1 对齐）
    :return: 前沿列表，元素为处于同一支配层级的点下标列表；下标从小到大为从优到劣
    :rtype: List[List[int]]
    """
    n = len(f1)
    S = [[] for _ in range(n)]       # S[p]：被 p 支配的点的下标
    n_p = [0] * n                    # 支配 p 的点的个数
    fronts: List[List[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominated(f1[p], f2[p], f1[q], f2[q]):
                S[p].append(q)
            elif dominated(f1[q], f2[q], f1[p], f2[p]):
                n_p[p] += 1
        if n_p[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_p[q] -= 1
                if n_p[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            if i < len(fronts):
                fronts[i] = nxt
            else:
                fronts.append(nxt)
    # 去掉末尾可能出现的空前沿
    return [f for f in fronts if f]


def crowding_distance(f1_norm: List[float], f2_norm: List[float],
                      indices: List[int] = None) -> Dict[int, float]:
    """在归一化 ``(f1, f2)`` 上算拥挤距离。

    :param f1_norm: 归一化 makespan 列表
    :param f2_norm: 归一化 TWTE 列表（与 f1_norm 对齐）
    :param indices: 可选，限定计算的点下标集合（缺省为全部点）
    :return: 下标 -> 拥挤距离 的映射；端点与 ≤2 点情形为 +inf
    :rtype: Dict[int, float]
    """
    idxs = list(range(len(f1_norm))) if indices is None else list(indices)
    n = len(idxs)
    dist = {i: 0.0 for i in idxs}
    if n <= 2:
        for i in idxs:
            dist[i] = float("inf")
        return dist
    for obj in (f1_norm, f2_norm):
        order = sorted(idxs, key=lambda i: obj[i])
        dist[order[0]] += float("inf")
        dist[order[-1]] += float("inf")
        span = obj[order[-1]] - obj[order[0]]
        if span > 0.0:
            for k in range(1, n - 1):
                dist[order[k]] += (obj[order[k + 1]] - obj[order[k - 1]]) / span
    return dist


def non_dominated_indices(f1: List[float], f2: List[float]) -> List[int]:
    """返回第一（非支配）前沿各点的下标。

    :param f1: 各点 makespan 列表
    :param f2: 各点 TWTE 列表（与 f1 对齐）
    :return: 非支配前沿的点下标列表
    :rtype: List[int]
    """
    fronts = fast_non_dominated_sort(f1, f2)
    return fronts[0] if fronts else []
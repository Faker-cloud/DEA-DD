"""阶段二（改进 NSGA-II）：交叉/变异算子与环境选择（§6）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from dea_dd.core.pareto import fast_non_dominated_sort, crowding_distance


@dataclass
class Individual:
    """NSGA-II 种群个体。

    :ivar ms: 机器选择段
    :ivar os: 工序排序段
    :ivar f1: makespan（蒙特卡洛期望）
    :ivar f2: TWTE（蒙特卡洛期望）
    """

    ms: List[int]
    os: List[int]
    f1: float
    f2: float


def pox_crossover(os1, os2, n_jobs, rng):
    """优先工序交叉（POX），作用于 OS 段。

    :param os1: 父代 1 的工序排序段
    :param os2: 父代 2 的工序排序段
    :param n_jobs: 工件数（用于随机划分 J1 集合）
    :param rng: 随机数发生器
    :return: 元组 ``(c1, c2)`` 两个子代 OS 段
    :rtype: tuple
    """
    jobs = np.arange(n_jobs)
    J1 = {int(j) for j in jobs if rng.random() < 0.5}
    # 防止 J1 为空或取满导致交叉退化
    if not J1 or len(J1) == n_jobs:
        split = max(1, n_jobs // 2)
        J1 = set(int(j) for j in rng.choice(jobs, size=split, replace=False))
    return _pox(os1, os2, J1), _pox(os2, os1, J1)


def _pox(keep, fill, J1):
    """保留 J1 集在 keep 中的位置，其余位置按 fill 中非 J1 顺序填入。

    :param keep: 保留位置来源的 OS
    :param fill: 填充顺序来源的 OS
    :param J1: 保留的工件集合
    :return: 一个子代的 OS 段
    :rtype: list
    """
    n = len(keep)
    c = [None] * n
    for i, j in enumerate(keep):
        if j in J1:
            c[i] = j
    it = iter([j for j in fill if j not in J1])
    for i in range(n):
        if c[i] is None:
            c[i] = next(it)
    return c


def ux_crossover(ms1, ms2, pc, rng):
    """均匀交叉（UX），作用于 MS 段（两个子代互补）。

    :param ms1: 父代 1 的机器选择段
    :param ms2: 父代 2 的机器选择段
    :param pc: 交换概率（逐基因位）
    :param rng: 随机数发生器
    :return: 元组 ``(c1, c2)`` 两个子代 MS 段
    :rtype: tuple
    """
    n = len(ms1)
    c1 = [0] * n
    c2 = [0] * n
    for a in range(n):
        if rng.random() < pc:
            c1[a] = ms1[a]
            c2[a] = ms2[a]
        else:
            c1[a] = ms2[a]
            c2[a] = ms1[a]
    return c1, c2


def dynamic_mutation_bits(gen, gen_max, total_ops, dmax=0.05, dmin=0.01):
    """式(16)：按规模缩放、随代数衰减的变异位置数。

    :param gen: 当前代数
    :param gen_max: 总代数
    :param total_ops: 工序总数（用于按比例缩放）
    :param dmax: 变异比例上界（缺省 0.05）
    :param dmin: 变异比例下界（缺省 0.01）
    :return: 本代应执行的变异位置数（下限 2）
    :rtype: int
    """
    delta = dmax * (dmin / dmax) ** (gen / gen_max)
    return max(2, int(np.ceil(delta * total_ops)))


def _swap_diff(os, rng):
    """交换两个不同工序值的位置，保证交换后 OS 一定改变。"""
    n = len(os)
    i = int(rng.integers(0, n))
    j = int(rng.choice([k for k in range(n) if os[k] != os[i]]))
    os[i], os[j] = os[j], os[i]


def _insert_diff(os, rng):
    """移除一个元素并插入到使序列改变的位置，保证结果与原串不同。"""
    before = list(os)
    n = len(os)
    for _ in range(64):                      # 有限次重试，避免无尽头
        i = int(rng.integers(0, n))
        v = before[i]
        j = int(rng.integers(0, n))
        tmp = list(before)
        tmp.pop(i)
        tmp.insert(j, v)                     # pop 后长度 n-1，j∈[0,n-1] 恒为合法插入位
        if tmp != before:
            os[:] = tmp                      # 只提交真正改变的结果
            return
    # 极端情形兜底：做一次必改的异值交换
    t = list(os)
    _swap_diff(t, rng)
    os[:] = t


def _inverse_diff(os, rng):
    """逆序一个非回文段（首尾异值），保证逆序后 OS 一定改变。"""
    n = len(os)
    i = int(rng.integers(0, n))
    j = int(rng.choice([k for k in range(n) if os[k] != os[i]]))
    if i > j:
        i, j = j, i
    os[i:j + 1] = os[i:j + 1][::-1]


def mutate_os(os, k_mut, rng):
    """OS 变异：做 k_mut 次 swap/insert/inverse，保证结果与输入一定不同。

    三种基本操作都做了「防空转」处理——交换/逆序改选两个不同工序值的位置、
    插入按「结果 != 原串」筛选——每次操作都会真实改变当前 OS；最后若 k_mut 次
    操作偶然相互抵消（结果与原串相同），再补一次必改的异值交换兜底。各工件
    出现次数（多重数）不变，排列恒可行。

    :param os: 待变异的工序排序段
    :param k_mut: 变异位置数
    :param rng: 随机数发生器
    :return: 变异后的 OS 段（新列表；长度<2 或全同工序值时原样返回，无法变异）
    :rtype: list
    """
    n = len(os)
    if n < 2 or len(set(os)) < 2:
        return list(os)
    os = list(os)
    orig = list(os)
    for _ in range(k_mut):
        kind = int(rng.integers(0, 3))
        if kind == 0:                        # 交换两不同工序值
            _swap_diff(os, rng)
        elif kind == 1:                      # 移除并插入到改变序列的位置
            _insert_diff(os, rng)
        else:                                # 逆序非回文段
            _inverse_diff(os, rng)
    if os == orig:                           # k_mut 次操作偶然抵消时兜底
        _swap_diff(os, rng)
    return os


def mutate_ms(ms, inst, k_mut, rng):
    """MS 变异：把 k_mut 道随机工序换到另一台可选机器。

    :param ms: 待变异的机器选择段
    :param inst: FJSP 实例（读取各工序可选机器）
    :param k_mut: 变异位置数
    :param rng: 随机数发生器
    :return: 变异后的 MS 段（新列表）
    :rtype: list
    """
    ms = list(ms)
    for _ in range(k_mut):
        a = int(rng.integers(0, len(ms)))
        cand = inst.machines[a]
        if len(cand) <= 1:
            continue
        other = int(rng.choice([k for k in range(len(cand)) if k != ms[a]]))
        ms[a] = other
    return ms


def environmental_selection(parents, offspring, normalizer, N):
    """合并 R = P ∪ Q，非支配排序 → 归一化拥挤 → 截断到 N（精英保留）。

    :param parents: 父代种群（Individual 列表）
    :param offspring: 子代种群（Individual 列表）
    :param normalizer: 归一化器
    :param N: 目标种群规模
    :return: 环境选择后的种群（Individual 列表，长度 N）
    :rtype: list
    """
    R = parents + offspring
    f1s = [x.f1 for x in R]
    f2s = [x.f2 for x in R]
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
                a, b = normalizer.normalize(f1s[i], f2s[i])
                n1.append(a)
                n2.append(b)
            cd = crowding_distance(n1, n2)
            order = sorted(range(len(front)), key=lambda k: -cd[k])
            selected.extend([front[k] for k in order[:need]])
            break
    return [R[i] for i in selected]
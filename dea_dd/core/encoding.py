"""编码：MS+OS 染色体 与 EO 实值双变量粒子。

主编码（阶段二 + 局部搜索）采用经典的 ``MS + OS`` 双段整数编码
（《算法设计.md》§3.1）。阶段一（EO 预热）采用实值双变量粒子 ``(u, v)``（§3.2）：

    u_ij ∈ [0, |M_ij|)   机器坐标 -> floor(u) 选择机器，
    v_ij ∈ R             优先级坐标 -> 仅相对顺序有意义。

机器坐标的越界修复**只用反射**（不做截断），对应需求与 §3.2.1 式(8b)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Chromosome:
    """一个 FJSP 解：机器选择段 + 工序排序段。

    :ivar ms: 长度 total_ops；``ms[a] ∈ [0, |M_a|-1]`` 为工序 a 在其候选机器集内的下标
    :ivar os: 长度 total_ops；工件 i 第 k 次出现表示其第 k 道工序
    """

    ms: List[int]
    os: List[int]

    def __hash__(self) -> int:
        """返回按 ``(ms, os)`` 元组计算的哈希值（用于去重与集合）。

        :return: (ms, os) 的哈希
        :rtype: int
        """
        return hash((tuple(self.ms), tuple(self.os)))

    def __eq__(self, other) -> bool:
        """判定两条染色体是否相同（ms 与 os 均相等）。

        :param other: 待比较的另一条染色体
        :return: 是否为同一解
        :rtype: bool
        """
        return (
            isinstance(other, Chromosome)
            and self.ms == other.ms
            and self.os == other.os
        )


def reflect_coord(x: float, m: float) -> float:
    """把 ``x`` 镜像反射回 ``[0, m)``（不用截断/钳制）。

    用周期 ``2m`` 的环面折叠、在 0 与 m 处镜像，等价于反复应用式(8b) 直到
    回到定义域；恰好落在 ``m`` 上的值用极小量回退一步。

    :param x: 待修复的坐标（可能是越界值）
    :param m: 定义域上界（独占），即该工序的可选机器个数 |M_a|
    :return: 反射回 ``[0, m)`` 之后的坐标
    :rtype: float
    """
    if m <= 0:
        return 0.0
    if x < 0:
        x = -x
    r = x % (2.0 * m)
    if r >= m:
        r = 2.0 * m - r
    if r >= m:                # 精确落在边界 -> 回退一步（式8b 备注）
        r = m - 1e-12
    if r < 0:
        r = 0.0
    return r


def reflect_u(u: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """按工序把机器坐标 ``u`` 反射回 ``[0, |M_a|)``。

    :param u: 机器坐标向量，长度 total_ops
    :param bounds: 各工序的定义域上界，``bounds[a] = len(machines[a])``
    :return: 反射修复后的机器坐标（新数组，输入不被修改）
    :rtype: numpy.ndarray
    """
    out = np.array(u, dtype=float, copy=True)
    for a in range(len(u)):
        out[a] = reflect_coord(float(u[a]), float(bounds[a]))
    return out


def particle_to_chromosome(inst, u: np.ndarray, v: np.ndarray) -> Tuple[List[int], List[int]]:
    """把 EO 实值粒子 ``(u, v)`` 解码为 ``(ms, os)``。

    ``ms[a] = floor(u[a])``（为安全再夹到候选机器下标范围）；``os`` 用贪心
    优先级解码生成：每一步在已释放（前继已排）的未排工序中挑 ``v`` 最大者
    （§3.2.2）。

    :param inst: FJSP 实例（:class:`~dea_dd.core.instance.Instance`）
    :param u: 机器坐标向量，长度 total_ops
    :param v: 优先级坐标向量，长度 total_ops（只比相对大小）
    :return: 元组 ``(ms, os)``，分别为机器选择段与工序排序段
    :rtype: Tuple[List[int], List[int]]
    """
    n_ops = inst.total_ops
    ms = [0] * n_ops
    for a in range(n_ops):
        cand = inst.machines[a]
        k = int(np.floor(u[a]))
        if k < 0:
            k = 0
        elif k >= len(cand):
            k = len(cand) - 1
        ms[a] = k

    scheduled = [False] * n_ops
    os: List[int] = []
    while len(os) < n_ops:
        best = -1
        best_v = -np.inf
        for a in range(n_ops):
            if scheduled[a]:
                continue
            job = inst.job_of_op[a]
            pos = inst.op_index_in_job[a]
            if pos > 0 and not scheduled[inst.job_op_ids[job][pos - 1]]:
                continue
            if v[a] > best_v:
                best = a
                best_v = v[a]
        scheduled[best] = True
        os.append(inst.job_of_op[best])
    return ms, os


def random_chromosome(inst, rng) -> Tuple[List[int], List[int]]:
    """随机生成一条合法 ``(ms, os)`` 染色体（供预热去重后不足 N 时补齐）。

    ``ms``：每道工序在其候选机器集内随机取一个下标；``os``：先由各工件的
    工序数拼出多重集，再随机洗牌（第 k 次出现即第 k 道工序，天然可行）。

    :param inst: FJSP 实例（:class:`~dea_dd.core.instance.Instance`）
    :param rng: 随机数发生器（:class:`numpy.random.Generator`）
    :return: 元组 ``(ms, os)``
    :rtype: Tuple[List[int], List[int]]
    """
    n_ops = inst.total_ops
    ms = [int(rng.integers(0, len(inst.machines[a]))) for a in range(n_ops)]
    os_flat = []
    for j in range(inst.n_jobs):
        os_flat.extend([j] * inst.ops_per_job[j])
    os = [int(x) for x in rng.permutation(os_flat)]
    return ms, os
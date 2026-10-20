"""均衡优化器（EO）作用于实值双变量粒子（《算法设计.md》§3.2、§5.2）。

在固定权重 w 下最小化切比雪夫标量 g(x|w)（式(11)），使用浓度更新公式
式(13)-(15)。仅机器坐标 u 做反射修复，优先级坐标 v 无界。
EO 的固定系数（a1、a2、GP、GCP_coef）统一从 Params 读取，不写死在代码里。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from dea_dd.core.encoding import reflect_u, particle_to_chromosome
from dea_dd.core.decoding import decode_evaluate


class EOSolver:
    """对单一权重向量 w 运行 EO；同时更新**共享的**全局归一化器。"""

    def __init__(self, inst, w, normalizer, params, rng=None, due_dates=None):
        """构造 EO 求解器。

        :param inst: FJSP 实例
        :param w: 权重向量分量（0 表示只关注 TWTE，1 表示只关注 makespan）
        :param normalizer: 共享的 :class:`~dea_dd.core.normalization.Normalizer`
        :param params: :class:`~dea_dd.params.Params` 参数对象
        :param rng: 随机数发生器；缺省新建一个
        :param due_dates: 交货期数组；缺省使用 ``inst.due_dates``
        """
        self.inst = inst
        self.w = float(w)
        self.norm = normalizer
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_ops = inst.total_ops
        # 各工序的机器坐标上界 = 其可选机器个数
        self.bounds = np.array([len(m) for m in inst.machines], dtype=float)
        self.due_dates = (inst.due_dates if due_dates is None
                          else np.asarray(due_dates, dtype=float))
        self.convergence = []  # (iter, best_f1, best_f2)

    def _random_particles(self, N: int) -> np.ndarray:
        """一次性随机初始化 N 个粒子（批量生成）。

        :param N: 粒子数
        :return: 形状 ``(N, 2*total_ops)`` 的粒子矩阵；前 ``n_ops`` 列为机器坐标
            u（``[0, |M_a|)``），后 ``n_ops`` 列为优先级坐标 v（``[0, 1]``）
        :rtype: numpy.ndarray
        """
        U = self.rng.uniform(0.0, 1.0, size=(N, self.n_ops)) * self.bounds
        V = self.rng.uniform(0.0, 1.0, size=(N, self.n_ops))
        return np.concatenate([U, V], axis=1)

    def _decode(self, particle) -> Tuple[List[int], List[int], float, float]:
        """解码一个粒子，只算原始 ``(f1, f2)``，不更新归一化器、不标量化。

        :param particle: ``[u; v]`` 实值向量（长度 2*total_ops）
        :return: 元组 ``(ms, os, f1, f2)``
        :rtype: tuple
        """
        u = particle[: self.n_ops]
        v = particle[self.n_ops:]
        ms, os = particle_to_chromosome(self.inst, u, v)
        f1, f2 = decode_evaluate(self.inst, ms, os, self.due_dates)
        return ms, os, f1, f2

    def _evaluate(self, particle) -> Tuple[List[int], List[int], float, float, float]:
        """评估单个粒子：解码、更新全局归一化器、算切比雪夫标量 g。

        :param particle: ``[u; v]`` 实值向量（长度 2*total_ops）
        :return: 元组 ``(ms, os, f1, f2, g)``，g 为该粒子在 w 下的切比雪夫标量
        :rtype: tuple
        """
        ms, os, f1, f2 = self._decode(particle)
        self.norm.update(f1, f2)       # 全局 min/max（式9）
        g = self.norm.chebyshev(f1, f2, self.w)
        return ms, os, f1, f2, g

    def _evaluate_many(self, particles):
        """一次性评估多个粒子：先统一解码算 ``(f1, f2)``，再用**同一**基座统一标量化。

        与循环调用 :meth:`_evaluate` 的差异：本方法先把整批的原始 ``(f1, f2)``
        全部算出，再用整批一次性更新归一化基座（式9 的逐代口径），最后才统一算
        ``g``。这样保证**同一代内所有粒子**的 g 都基于相同的 min/max，避免「边算边
        扩基座」使先算的粒子拿到偏窄/退化基座而造成的排序漂移。

        :param particles: 形状 ``(M, 2*total_ops)`` 的粒子矩阵
        :return: 元组 ``(ms_list, os_list, f1_list, f2_list, g_list)``，各列表长度为 M
        :rtype: tuple
        """
        ms_list, os_list, f1_list, f2_list = [], [], [], []
        for p in particles:
            ms, os, f1, f2 = self._decode(p)
            ms_list.append(ms)
            os_list.append(os)
            f1_list.append(f1)
            f2_list.append(f2)
        # 用整批 (f1, f2) 一次性更新基座，再统一标量化
        self.norm.update_many(f1_list, f2_list)
        g_list = [self.norm.chebyshev(f1, f2, self.w)
                  for f1, f2 in zip(f1_list, f2_list)]
        return ms_list, os_list, f1_list, f2_list, g_list

    def run(self):
        """运行 EO 主循环并返回候选解与收敛序列。

        :return: 元组 ``(candidates, convergence)``。candidates 为
            ``[(ms, os, f1, f2), ...]``，由最终粒子群 + 历史均衡候选
            （每代前 4 个最优）组成；convergence 为 ``[(iter, best_f1, best_f2), ...]``
        :rtype: tuple
        """
        N = self.params.N_eo
        iters = self.params.Iter_EO
        a1 = self.params.a1
        a2 = self.params.a2
        GP = self.params.GP
        gcp_coef = self.params.GCP_coef

        # 初始化：一次性生成 N 个粒子，再一次评估 + 切比雪夫标量化
        particles = self._random_particles(N)
        ms_list, os_list, f1_list, f2_list, g_list = self._evaluate_many(particles)
        state = [{"p": p, "ms": m, "os": o, "f1": f1, "f2": f2, "g": g}
                 for p, m, o, f1, f2, g in zip(particles,
                                               ms_list, os_list,
                                               f1_list, f2_list, g_list)]

        candidates = []

        def record(it_):
            best = min(state, key=lambda s: s["g"])
            self.convergence.append((it_, best["f1"], best["f2"]))

        record(0)
        for it in range(1, iters + 1):
            # 均衡池：按 g 排序取前 4 个均衡候选 + 其均值
            order = sorted(range(N), key=lambda i: state[i]["g"])
            top4 = [state[i]["p"] for i in order[:4]]
            pool = top4 + [np.mean(top4, axis=0)]
            for i in order[:4]:
                candidates.append(
                    (state[i]["ms"], state[i]["os"], state[i]["f1"], state[i]["f2"])
                )
            t_star = (1.0 - it / iters) ** (a2 * it / iters)
            for q in range(N):
                p = state[q]["p"]
                C_eq = pool[int(self.rng.integers(0, 5))]     # 随机挑一个锚点
                lam = self.rng.uniform(0.0, 1.0, size=2 * self.n_ops)  # 周转率随机向量
                r0 = self.rng.uniform()
                F = a1 * np.sign(r0 - 0.5) * (np.exp(-lam * t_star) - 1.0)
                r1 = self.rng.uniform()
                r2 = self.rng.uniform()
                GCP = gcp_coef * r1 if r2 >= GP else 0.0
                G0 = GCP * (C_eq - lam * p)
                G = G0 * F
                new_p = C_eq + (p - C_eq) * F + (G / (lam + 1e-12)) * (1.0 - F)
                # 仅机器坐标 u 做反射修复，v 不动
                new_u = reflect_u(new_p[: self.n_ops], self.bounds)
                new_p = np.concatenate([new_u, new_p[self.n_ops:]])
                state[q]["p"] = new_p
            # 更新后一次性评估全部粒子
            P = np.array([s["p"] for s in state])
            ms_list, os_list, f1_list, f2_list, g_list = self._evaluate_many(P)
            for q in range(N):
                state[q].update(ms=ms_list[q], os=os_list[q], f1=f1_list[q],
                                f2=f2_list[q], g=g_list[q])
            record(it)

        result = [(s["ms"], s["os"], s["f1"], s["f2"]) for s in state]
        result.extend(candidates)
        return result, self.convergence
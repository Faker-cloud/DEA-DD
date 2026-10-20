"""对数正态随机加工时间（式(19)/(19a)）与蒙特卡洛评估（式(20)），
采用共同随机数（CRN）语义。

E[p_ijk] = p_ijk，Var[p_ijk] = UL * p_ijk，于是
    sigma_ijk = sqrt(ln(1 + UL/p_ijk))，mu_ijk = ln(p_ijk) - 0.5*sigma_ijk^2。
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from dea_dd.core.decoding import left_shift_decode, right_shift_twte


class LognormalProcessing:
    """为固定不确定性水平 UL 预计算每道工序在每台机器上的 (mu, sigma)。"""

    def __init__(self, inst, ul: float, rng: np.random.Generator = None):
        """根据名义加工时间与 UL 预计算对数正态分布的 ``(mu, sigma)``。

        :param inst: FJSP 实例（读取各工序的名义加工时间）
        :param ul: 不确定性水平，``Var = UL * p``
        :param rng: 随机数发生器；缺省新建一个
        """
        self.inst = inst
        self.ul = float(ul)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mu: list = []
        self.sigma: list = []
        for a in range(inst.total_ops):
            mus = []
            sigs = []
            for p in inst.times[a]:
                p = float(p)
                var = self.ul * p
                s2 = math.log(1.0 + var / (p * p))
                sig = math.sqrt(s2)
                mu = math.log(p) - 0.5 * s2
                mus.append(mu)
                sigs.append(sig)
            self.mu.append(mus)
            self.sigma.append(sigs)

    def sample_scenario(self):
        """采样一组随机加工时间。

        :return: 加工时间表，结构与 ``inst.times`` 一致（每工序每可选机器一个采样值）
        :rtype: list
        """
        times = []
        for a in range(self.inst.total_ops):
            times.append([
                self.rng.lognormal(self.mu[a][k], self.sigma[a][k])
                for k in range(len(self.mu[a]))
            ])
        return times

    def sample_scenarios(self, R: int):
        """采样 R 组场景。

        :param R: 场景数
        :return: R 组加工时间表（见 :meth:`sample_scenario`）
        :rtype: list
        """
        return [self.sample_scenario() for _ in range(R)]


def mc_evaluate(inst, ms, os, scenarios, due_dates):
    """在一组共享场景上求均值 ``(f1_hat, f2_hat)``（CRN，式(20)）。

    :param inst: FJSP 实例
    :param ms: 机器选择段
    :param os: 工序排序段
    :param scenarios: 一组加工时间场景（由 :meth:`LognormalProcessing.sample_scenarios` 生成）
    :param due_dates: 各工件交货期
    :return: 元组 ``(f1_hat, f2_hat)``，为各场景解码目标值的算术平均
    :rtype: Tuple[float, float]
    """
    s1 = 0.0
    s2 = 0.0
    for times in scenarios:
        S, C, mseq, _ = left_shift_decode(inst, ms, os, times)
        f1 = float(max(C)) if C else 0.0
        f2, _ = right_shift_twte(inst, ms, C, mseq, due_dates, times)
        s1 += f1
        s2 += f2
    R = len(scenarios)
    return s1 / R, s2 / R


def mc_evaluate_long(inst, ms, os, processor, due_dates, R: int):
    """长仿真：在 R 组独立场景上返回 ``(f1, f2)`` 的均值与标准差。

    :param inst: FJSP 实例
    :param ms: 机器选择段
    :param os: 工序排序段
    :param processor: :class:`LognormalProcessing` 采样器
    :param due_dates: 各工件交货期
    :param R: 场景数
    :return: 元组 ``(m1, m2, std1, std2)``，分别为 makespan 与 TWTE 的均值与标准差
    :rtype: Tuple[float, float, float, float]
    """
    s1 = 0.0
    s2 = 0.0
    q1 = 0.0
    q2 = 0.0
    for _ in range(R):
        times = processor.sample_scenario()
        S, C, mseq, _ = left_shift_decode(inst, ms, os, times)
        f1 = float(max(C)) if C else 0.0
        f2, _ = right_shift_twte(inst, ms, C, mseq, due_dates, times)
        s1 += f1
        s2 += f2
        q1 += f1 * f1
        q2 += f2 * f2
    m1 = s1 / R
    m2 = s2 / R
    v1 = max(0.0, q1 / R - m1 * m1)
    v2 = max(0.0, q2 / R - m2 * m2)
    return m1, m2, math.sqrt(v1), math.sqrt(v2)
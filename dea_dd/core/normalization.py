"""逐代动态 min-max 归一化 + 切比雪夫标量化。

对应《算法设计.md》式(9)-(11)。原始目标值始终存储原始值，比较时用**当前**
极值现场归一化（绝不跨代缓存）。帕累托支配本身不需要归一化。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


class Normalizer:
    """全局共享的逐代动态 min/max 归一化器。"""

    EPS = 1e-6

    def __init__(self, f1min=float("inf"), f1max=float("-inf"),
                 f2min=float("inf"), f2max=float("-inf")):
        """构造归一化器并给定初始极值。

        :param f1min: f1(makespan) 的初始下界（缺省 +inf）
        :param f1max: f1(makespan) 的初始上界（缺省 -inf）
        :param f2min: f2(TWTE) 的初始下界（缺省 +inf）
        :param f2max: f2(TWTE) 的初始上界（缺省 -inf）
        """
        self.f1min = f1min
        self.f1max = f1max
        self.f2min = f2min
        self.f2max = f2max

    def update(self, f1: float, f2: float) -> None:
        """用单个 ``(f1, f2)`` 更新极值（式9）。

        :param f1: makespan 值
        :param f2: TWTE 值
        """
        self.f1min = min(self.f1min, f1)
        self.f1max = max(self.f1max, f1)
        self.f2min = min(self.f2min, f2)
        self.f2max = max(self.f2max, f2)

    def update_many(self, f1s, f2s) -> None:
        """用一批 ``(f1, f2)`` 更新极值。

        :param f1s: makespan 值序列（可迭代）
        :param f2s: TWTE 值序列（与 f1s 对齐）
        """
        if len(f1s) == 0:
            return
        self.f1min = min(self.f1min, float(np.min(f1s)))
        self.f1max = max(self.f1max, float(np.max(f1s)))
        self.f2min = min(self.f2min, float(np.min(f2s)))
        self.f2max = max(self.f2max, float(np.max(f2s)))

    def normalize(self, f1: float, f2: float) -> Tuple[float, float]:
        """返回归一化 ``(f1bar, f2bar) ∈ [0, 1]``（式10）。

        :param f1: makespan 值
        :param f2: TWTE 值
        :return: 归一化后的 (f1bar, f2bar)；若某目标极差小于 EPS 则该分量取 0
        :rtype: Tuple[float, float]
        """
        r1 = (f1 - self.f1min) / (self.f1max - self.f1min + self.EPS)
        r2 = (f2 - self.f2min) / (self.f2max - self.f2min + self.EPS)
        if self.f1max - self.f1min < self.EPS:
            r1 = 0.0
        if self.f2max - self.f2min < self.EPS:
            r2 = 0.0
        return r1, r2

    def chebyshev(self, f1: float, f2: float, w: float) -> float:
        """切比雪夫标量 ``g(x|w) = max(w*f1bar, (1-w)*f2bar)``（式11）。

        :param f1: makespan 值
        :param f2: TWTE 值
        :param w: 权重向量分量（0 表示只关注 TWTE，1 表示只关注 makespan）
        :return: 切比雪夫标量值 g(x|w)
        :rtype: float
        """
        r1, r2 = self.normalize(f1, f2)
        return max(w * r1, (1.0 - w) * r2)
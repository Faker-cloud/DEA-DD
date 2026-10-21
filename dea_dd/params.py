"""算法参数，默认值取自《算法设计.md》§11。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class Params:
    """DEA-DD 算法参数集（dataclass 各字段即构造参数）。

    :ivar N: 种群规模（100）
    :ivar N_eo: 每个 w 子问题的 EO 粒子数（100）
    :ivar Iter_EO: 每个 w 的 EO 迭代次数（60）
    :ivar a1: EO 式(14) 的勘探/开采系数（2.0）
    :ivar a2: EO 式(14) ``t*`` 的指数系数（1.0）
    :ivar GP: EO 式(15) 生成率控制概率阈值，``r2 >= GP`` 时生成 GCP（0.5）
    :ivar GCP_coef: EO 式(15) 生成率控制系统，``GCP = GCP_coef * r1``（0.5）
    :ivar K: 每个 w 保留的种子数，需满足 ``K ≤ N / |W|``（100）
    :ivar p_c: 交叉概率（POX/UX，0.85）
    :ivar delta_max: 变异比例上界（式16，0.05）
    :ivar delta_min: 变异比例下界（式16，0.01）
    :ivar p_m: 变异概率（每个子代以 p_m 执行 OS+MS 变异，否则跳过；0.1）
    :ivar gen_max: 阶段二代数（400）
    :ivar ls_trigger: 延迟局部搜索触发比例（式18，0.8）
    :ivar L: 局部搜索成功率滑动窗口长度（10）
    :ivar eps: 归一化分母防除零（1e-6）
    :ivar mu: 算子选择基线（式17，0.05）
    :ivar UL: 不确定性水平（式19，0.25）
    :ivar R_short: 迭代期蒙特卡洛场景数（20）
    :ivar R_long: 最终长仿真场景数（100）
    :ivar seed: 随机数种子（0）
    :ivar weights: 权重向量序列（两端优先）
    """

    N: int = 100
    N_eo: int = 100
    Iter_EO: int = 60
    a1: float = 2.0
    a2: float = 1.0
    GP: float = 0.5
    GCP_coef: float = 0.5
    K: int = 50
    p_c: float = 0.85
    delta_max: float = 0.05
    delta_min: float = 0.01
    p_m: float = 0.2
    gen_max: int = 400
    ls_trigger: float = 0.65
    L: int = 10
    eps: float = 1e-6
    mu: float = 0.05
    UL: float = 0.25
    R_short: int = 20
    R_long: int = 100
    seed: int = 0
    weights: Tuple[float, ...] = (0.0, 1.0, 0.1, 0.9, 0.25, 0.75, 0.5)
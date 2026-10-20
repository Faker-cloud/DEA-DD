"""DEA-DD：双目标柔性作业车间调度（最小化 makespan + 最小化 TWTE）。

实现《算法设计.md》的 DEA-DD 算法：
  - 阶段一：多 w 切比雪夫分解预热，用均衡优化器（EO）在实值双变量编码上优化；
  - 阶段二：改进 NSGA-II 主搜索，对数正态随机加工时间的蒙特卡洛评估，
    以及延迟触发的自适应局部搜索；
  - 阶段三：对最终前沿做长仿真（R=10000）。

对外主要入口：``dea_dd.main.run(params, ...)``（程序化）与
``dea_dd.main.main(...)``（命令行 / 程序化）。
"""
from __future__ import annotations

__version__ = "0.2.0"

from dea_dd.core.instance import Instance, parse_fjs, generate_due_dates
from dea_dd.core.encoding import Chromosome, reflect_coord, reflect_u, particle_to_chromosome
from dea_dd.core.decoding import (
    left_shift_decode,
    right_shift_twte,
    decode_evaluate,
    twte,
)
from dea_dd.core.normalization import Normalizer

__all__ = [
    "Instance",
    "parse_fjs",
    "generate_due_dates",
    "Chromosome",
    "reflect_coord",
    "reflect_u",
    "particle_to_chromosome",
    "left_shift_decode",
    "right_shift_twte",
    "decode_evaluate",
    "twte",
    "Normalizer",
]
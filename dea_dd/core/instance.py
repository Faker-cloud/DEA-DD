"""FJSP 实例定义、`.fjs` 解析与交货期生成。

`.fjs` 格式（《算法设计.md》§12.1 / Lei-Kun FJSP-benchmarks）：

    <工件数> <机器数> <平均候选机器数>            # 首行
    <工序数 h_i>  <候选数> (机器号 用时)*候选数 ...   # 每工件一行

文件中的机器号从 1 开始，解析时统一转为 0 基。交货期采用 Bagheri &
Zandieh (2011) 的 SW 分支：

    d_i ~ U[ mu_i*(1 - R/2),  mu_i*(1 + R/2) ],
    mu_i = (1 + T*I/M) * sum_j mean(PT_ij)

其中 I 为工件数、M 为机器数，mean(PT_ij) 为工序 O_ij 在其可选机器上的
平均加工时间，T=0.2、R=1.0。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Instance:
    """基于 0 基下标、机器相关的 FJSP 实例。

    :ivar n_jobs: 工件数
    :ivar n_machines: 机器数
    :ivar ops_per_job: 各工件的工序数 h_i
    :ivar job_op_ids: 各工件的全局工序号列表（有序）
    :ivar machines: 各工序的可选机器列表（0 基）
    :ivar times: 各工序在对应可选机器上的加工时间（与 machines 对齐）
    :ivar due_dates: 各工件交货期（长度 n_jobs）
    :ivar job_of_op: 派生：各全局工序号所属的工件
    :ivar op_index_in_job: 派生：各全局工序号在其工件内的序号
    :ivar last_op_of_job: 派生：各工件的末道全局工序号
    """

    n_jobs: int
    n_machines: int
    ops_per_job: List[int]
    job_op_ids: List[List[int]]
    machines: List[List[int]]
    times: List[List[int]]
    due_dates: np.ndarray = None

    def __post_init__(self):
        """派生下标（job_of_op / op_index_in_job / last_op_of_job），并把
        ``due_dates`` 规范为 numpy 数组（未提供时填 0，需调用方用
        :func:`generate_due_dates` 覆盖）。"""
        self.job_of_op: List[int] = []
        self.op_index_in_job: List[int] = []
        self.last_op_of_job: List[int] = []
        for i, ids in enumerate(self.job_op_ids):
            self.last_op_of_job.append(ids[-1])
            for idx in range(len(ids)):
                self.job_of_op.append(i)
                self.op_index_in_job.append(idx)
        if self.due_dates is None:
            self.due_dates = np.full(self.n_jobs, 0.0, dtype=float)
        else:
            self.due_dates = np.asarray(self.due_dates, dtype=float)

    @property
    def total_ops(self) -> int:
        """:return: 工序总数（等于 machines 的长度）。"""
        return len(self.machines)


def parse_fjs(text: str) -> Instance:
    """把 `.fjs` 文本解析为 :class:`Instance`。

    会做格式校验，下列情况抛 ``ValueError``（尽早暴露算例文件错误，避免静默
    产生机器号越界 / 少算工序的非法实例）：
      - 空输入或表头缺少「工件数 机器数」；
      - 工件行数量不足；
      - 候选机器数 ``k`` 非正、或某道工序的 (机器号 用时) 对不完整；
      - 机器号超出 ``[1, n_machines]``；
      - 解析完某工件 ``h`` 道工序后行内还有多余数字。

    :param text: `.fjs` 文件的完整文本
    :return: 解析得到的 FJSP 实例
    :rtype: Instance
    """
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    if not lines:
        raise ValueError(".fjs 输入为空，缺少表头行")
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("表头必须至少包含「工件数 机器数」，得到: %r" % lines[0])
    n_jobs = int(header[0])
    n_machines = int(header[1])
    if n_jobs <= 0 or n_machines <= 0:
        raise ValueError("工件数与机器数必须为正，得到 %d %d" % (n_jobs, n_machines))
    if len(lines) < 1 + n_jobs:
        raise ValueError("期望 %d 个工件行，实际只有 %d 行"
                         % (n_jobs, len(lines) - 1))

    ops_per_job: List[int] = []
    job_op_ids: List[List[int]] = []
    machines: List[List[int]] = []
    times: List[List[int]] = []

    for i in range(n_jobs):
        toks = lines[1 + i].split()
        if not toks:
            raise ValueError("工件 %d 的工序行是空的" % (i + 1))
        h = int(toks[0])               # 该工件的工序数
        if h <= 0:
            raise ValueError("工件 %d 的工序数必须为正，得到 %d" % (i + 1, h))
        ops_per_job.append(h)
        jids: List[int] = []
        pos = 1
        for op in range(h):
            if pos >= len(toks):
                raise ValueError("工件 %d 的第 %d 道工序缺少候选机器数 k"
                                 % (i + 1, op + 1))
            k = int(toks[pos])         # 候选机器数
            pos += 1
            if k <= 0:
                raise ValueError("工件 %d 的第 %d 道工序候选机器数 k 必须为正，得到 %d"
                                 % (i + 1, op + 1, k))
            op_machines: List[int] = []
            op_times: List[int] = []
            for _ in range(k):
                if pos + 1 >= len(toks):
                    raise ValueError("工件 %d 的第 %d 道工序的 (机器号 用时) 对不完整"
                                     % (i + 1, op + 1))
                m = int(toks[pos])     # 机器号（文件里 1 基）
                t = int(toks[pos + 1])  # 加工时间
                pos += 2
                if not 1 <= m <= n_machines:
                    raise ValueError("工件 %d 的第 %d 道工序的机器号 %d 超出范围 [1, %d]"
                                     % (i + 1, op + 1, m, n_machines))
                op_machines.append(m - 1)   # 1 基 -> 0 基
                op_times.append(t)
            machines.append(op_machines)
            times.append(op_times)
            jids.append(len(machines) - 1)
        if pos != len(toks):
            raise ValueError("工件 %d 的工序行有多余数字：解析完 %d 道工序后仍余 %d 个数字"
                             % (i + 1, h, len(toks) - pos))
        job_op_ids.append(jids)

    return Instance(
        n_jobs=n_jobs,
        n_machines=n_machines,
        ops_per_job=ops_per_job,
        job_op_ids=job_op_ids,
        machines=machines,
        times=times,
    )


def generate_due_dates(
    inst: Instance,
    T: float = 0.2,
    R: float = 1.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """为每个工件生成一个交货期（Bagheri & Zandieh 的 SW 分支）。

    注意：本函数会回写 ``inst.due_dates``（这是修复过的关键缺陷，确保阶段一
    EO 与阶段二/三使用同一套交货期），同时返回该数组。

    :param inst: FJSP 实例（用于读取各工序可选机器上的平均加工时间）
    :param T: 完工窗宽度系数（缺省 0.2）
    :param R: 交货期均匀采样区间相对宽度（缺省 1.0，即 ``[0.5μ, 1.5μ]``）
    :param rng: 随机数发生器；缺省新建一个
    :return: 各工件交货期数组（长度 n_jobs），并回写 ``inst.due_dates``
    :rtype: numpy.ndarray
    """
    rng = rng if rng is not None else np.random.default_rng()
    d = np.empty(inst.n_jobs, dtype=float)
    for i in range(inst.n_jobs):
        total_mean = 0.0
        for gid in inst.job_op_ids[i]:
            total_mean += float(np.mean(inst.times[gid]))
        mu = (1.0 + T * inst.n_jobs / inst.n_machines) * total_mean
        lo = mu * (1.0 - R / 2.0)
        hi = mu * (1.0 + R / 2.0)
        d[i] = rng.uniform(lo, hi)
    inst.due_dates = d
    return d
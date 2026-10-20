"""合成 FJSP 算例生成器（用于测试与无 .fjs 文件时的回退）。

结构上与 ``parse_fjs`` 一致：``machines[a]`` 为 0 基可选机器列表，
``times[a]`` 为对齐的加工时间列表。
"""
from __future__ import annotations

import numpy as np

from dea_dd.core.instance import Instance


def random_instance(n_jobs, n_machines, min_ops=2, max_ops=6, seed=0):
    """随机生成一个 ``n_jobs × n_machines`` 的 FJSP 实例。

    :param n_jobs: 工件数
    :param n_machines: 机器数
    :param min_ops: 每个工件的最少工序数（缺省 2）
    :param max_ops: 每个工件的最多工序数（缺省 6）
    :param seed: 随机种子（缺省 0）
    :return: 随机生成的 FJSP 实例
    :rtype: Instance
    """
    rng = np.random.default_rng(seed)
    ops_per_job = [int(rng.integers(min_ops, max_ops + 1))
                   for _ in range(n_jobs)]
    job_op_ids = []
    machines = []
    times = []
    gid = 0
    for h in ops_per_job:
        jids = []
        for _ in range(h):
            nm = int(rng.integers(1, n_machines + 1))
            sel = sorted(int(m) for m in rng.choice(n_machines, size=nm,
                                                   replace=False))
            op_times = [int(rng.integers(1, 100)) for _ in sel]
            machines.append(sel)
            times.append(op_times)
            jids.append(gid)
            gid += 1
        job_op_ids.append(jids)
    return Instance(
        n_jobs=n_jobs,
        n_machines=n_machines,
        ops_per_job=ops_per_job,
        job_op_ids=job_op_ids,
        machines=machines,
        times=times,
    )
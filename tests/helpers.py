"""Shared helpers for tests (no pytest available)."""

import numpy as np

from dea_dd.core.instance import Instance


# Toy example from 算法设计.md 搂3.1.1: 3 jobs x 2 machines, 5 operations.
# Times: O11(M1=3,M2=4) O12(M1=5,M2=2) O21(M1=2,M2=4) O22(M1=5,M2=1) O31(M1=3,M2=3)
def toy_instance() -> Instance:
    return Instance(
        n_jobs=3,
        n_machines=2,
        ops_per_job=[2, 2, 1],
        job_op_ids=[[0, 1], [2, 3], [4]],
        machines=[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
        times=[[3, 4], [5, 2], [2, 4], [5, 1], [3, 3]],
        due_dates=np.array([8.0, 12.0, 8.0]),
    )


# Small synthetic instance: 2 jobs x 2 machines.
def synthetic_instance() -> Instance:
    return Instance(
        n_jobs=2,
        n_machines=2,
        ops_per_job=[2, 2],
        job_op_ids=[[0, 1], [2, 3]],
        machines=[[0, 1], [0], [0, 1], [1]],
        times=[[4, 6], [5], [3, 7], [8]],
        due_dates=np.array([20.0, 25.0]),
    )
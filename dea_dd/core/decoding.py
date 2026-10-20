"""双解码：左移解码（makespan）与右移解码（TWTE）。

同一条染色体解码两次（《算法设计.md》§3.3，创新②）：

  * 左移解码（active，不留空闲） -> f1 = makespan；
  * 右移解码（RightShiftRepair） -> f2 = TWTE，插入空闲把早到工件贴到
    交货期，且绝不增加任何拖期。
"""
from __future__ import annotations

from typing import List, Tuple


def left_shift_decode(inst, ms: List[int], os: List[int], times=None):
    """主动（左移）解码：按 OS 顺序把每道工序安排到其机器可用的最早空档。

    :param inst: FJSP 实例
    :param ms: 机器选择段，长度 total_ops；``ms[a]`` 为工序 a 在候选机器集内的下标
    :param os: 工序排序段，长度 total_ops；工件号序列（第 k 次出现即第 k 道工序）
    :param times: 可选，采样到的随机加工时间（结构同 ``inst.times``）；缺省用名义加工时间
    :return: 元组 ``(S, C, machine_seq, chosen_machine)``：
        ``S``/``C`` 为各工序开始/完工时间列表；
        ``machine_seq[m]`` 为机器 m 上按加工顺序排的工序号列表；
        ``chosen_machine[a]`` 为工序 a 实际分配的 0 基机器号。
    :rtype: tuple
    """
    n_ops = inst.total_ops
    m = inst.n_machines
    S = [0] * n_ops
    C = [0] * n_ops
    machine_avail = [0] * m
    machine_seq: List[List[int]] = [[] for _ in range(m)]
    chosen_machine = [-1] * n_ops
    job_scheduled = [0] * inst.n_jobs
    job_last_C = [0] * inst.n_jobs

    for job in os:
        pos = job_scheduled[job]
        op_id = inst.job_op_ids[job][pos]
        job_scheduled[job] += 1

        k = int(ms[op_id])
        machine = inst.machines[op_id][k]
        # 加工时间：不传 times 时用名义加工时间，否则用采样到的随机加工时间
        p = times[op_id][k] if times is not None else inst.times[op_id][k]
        chosen_machine[op_id] = machine

        pred_C = job_last_C[job] if pos > 0 else 0
        start = max(pred_C, machine_avail[machine])
        end = start + p
        S[op_id] = start
        C[op_id] = end
        job_last_C[job] = end
        machine_avail[machine] = end
        machine_seq[machine].append(op_id)

    return S, C, machine_seq, chosen_machine


def twte(job_completion, due_dates) -> float:
    """无权总加权拖期/提前（WT = WE = 1）。

    :param job_completion: 各工件（末道工序）完工时间列表
    :param due_dates: 各工件交货期列表（对齐）
    :return: TWTE = Σ_i max(0, C_i-d_i) + max(0, d_i-C_i)
    :rtype: float
    """
    total = 0.0
    for c, d in zip(job_completion, due_dates):
        total += max(0.0, c - d) + max(0.0, d - c)
    return total


def right_shift_twte(inst, ms: List[int], C: List[int],
                     machine_seq: List[List[int]], due_dates,
                     times=None) -> Tuple[float, List[float]]:
    """右移（RightShiftRepair）TWTE 评估（§3.3b）。

    从左移调度出发（机器分配与机器顺序固定），逆时间序回推最晚完工时间 LC(o)，
    使早到工件能向交货期方向滑动。保证不增加任何拖期，也不修改输入 ``C``。

    :param inst: FJSP 实例
    :param ms: 机器选择段（用于确定各工序的加工时间）
    :param C: 左移解码得到的各工序完工时间
    :param machine_seq: 左移解码得到的各机器工序顺序（``machine_seq[m]`` 为机器 m 上的工序号列表）
    :param due_dates: 各工件交货期列表
    :param times: 可选，采样到的随机加工时间；缺省用名义加工时间
    :return: 元组 ``(f2, LC)``，f2 为右移后的 TWTE，LC 为各工序的最晚完工时间
    :rtype: Tuple[float, List[float]]
    """
    n_ops = inst.total_ops

    # 由固定调度得到各工序的“机器后继”与“工件后继”。
    machine_succ = [-1] * n_ops
    for seq in machine_seq:
        for a, b in zip(seq, seq[1:]):
            machine_succ[a] = b
    job_succ = [-1] * n_ops
    for ids in inst.job_op_ids:
        for a, b in zip(ids, ids[1:]):
            job_succ[a] = b

    # 各工序在所选机器上的加工时间。
    p = [(times[a][int(ms[a])] if times is not None else inst.times[a][int(ms[a])])
         for a in range(n_ops)]

    LC = [float("inf")] * n_ops
    for i in range(inst.n_jobs):
        last = inst.last_op_of_job[i]
        LC[last] = max(float(C[last]), float(due_dates[i]))

    # 按完工时间降序处理（后继先算）。
    order = sorted(range(n_ops), key=lambda a: -C[a])
    for a in order:
        val = LC[a]
        js = job_succ[a]
        if js >= 0:
            val = min(val, LC[js] - p[js])
        ms2 = machine_succ[a]
        if ms2 >= 0:
            val = min(val, LC[ms2] - p[ms2])
        LC[a] = val

    job_completion = [LC[last] for last in inst.last_op_of_job]
    f2 = twte(job_completion, due_dates)
    return f2, LC


def decode_evaluate(inst, ms: List[int], os: List[int], due_dates,
                    times=None) -> Tuple[float, float]:
    """解码一条染色体并返回 ``(f1, f2)``。

    f1 = 左移解码的 makespan；f2 = 右移解码的 TWTE（解耦评估，创新②）。
    右移解码只在内部新建的 ``LC`` 上回推，不修改左移的 ``C``，故无需
    二次左移，``f1 = max(C)`` 始终有效。

    :param inst: FJSP 实例
    :param ms: 机器选择段
    :param os: 工序排序段
    :param due_dates: 各工件交货期列表
    :param times: 可选，采样到的随机加工时间；缺省用名义加工时间
    :return: 元组 ``(f1, f2)``，f1=makespan、f2=TWTE
    :rtype: Tuple[float, float]
    """
    S, C, machine_seq, _ = left_shift_decode(inst, ms, os, times)
    f1 = float(max(C)) if C else 0.0
    f2, _ = right_shift_twte(inst, ms, C, machine_seq, due_dates, times)
    return f1, f2
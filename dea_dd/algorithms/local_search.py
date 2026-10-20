"""交货期导向的双向邻域 + 自适应算子选择（创新③④，《算法设计.md》§7-§8）。

算子 L1-L3 压缩 makespan；R1a/R1b 把早到工件的某道工序向交货期方向移动
（换更慢机器 / 在机器序列中后移）。一次移动记为**成功**当且仅当结果不被父代
帕累托支配（“帕累托非支配改进”，§8.1）。算子概率按成功率轮盘 式(17) 在滑动
窗口 L=10 上自适应，冷启动为 0.25。
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

from dea_dd.core.decoding import left_shift_decode
from dea_dd.core.pareto import dominated

OPERATORS = ["L1", "L2", "L3", "R1"]


class AdaptiveLocalSearch:
    """纯邻域移动生成器；评估与成功记录留在 main 中完成。"""

    def __init__(self, inst, due_dates, params, rng=None):
        """构造自适应局部搜索器。

        :param inst: FJSP 实例
        :param due_dates: 各工件交货期
        :param params: 参数对象（决定滑动窗口长度 L、基线 μ）
        :param rng: 随机数发生器；缺省新建一个
        """
        self.inst = inst
        self.due = due_dates
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng()
        self.history = {o: deque(maxlen=params.L) for o in OPERATORS}

    # ---- 自适应算子选择（式17） -------------------------------------------
    def pick_operator(self) -> str:
        """按成功率轮盘（式17）选一个算子。

        :return: 算子名（"L1"/"L2"/"L3"/"R1"）
        :rtype: str
        """
        mu = self.params.mu
        raw = []
        for o in OPERATORS:
            h = self.history[o]
            S = sum(1 for x in h if x)
            F = sum(1 for x in h if not x)
            # 冷启动（S+F=0）时成功率记 0，各算子概率退化为均分
            rate = (S / (S + F)) if (S + F) > 0 else 0.0
            raw.append(rate + mu)
        total = sum(raw)
        probs = [r / total for r in raw]
        return OPERATORS[int(self.rng.choice(len(OPERATORS), p=probs))]

    def record(self, op, success) -> None:
        """记录一次算子的成功/失败。

        :param op: 算子名
        :param success: 是否成功（布尔）
        """
        self.history[op].append(bool(success))

    # ---- 移动分发 ---------------------------------------------------------
    def generate(self, op, ms, os):
        """按算子名分发到对应邻域移动。

        :param op: 算子名（"L1"/"L2"/"L3"/"R1"）
        :param ms: 当前机器选择段
        :param os: 当前工序排序段
        :return: 元组 ``(new_ms, new_os)``；无可行移动时返回 ``(None, None)``
        :rtype: tuple
        """
        if op == "L1":
            return self._L1(ms, os)
        if op == "L2":
            return self._L2(ms, os)
        if op == "L3":
            return self._L3(ms, os)
        return self._R1(ms, os)

    # ---- 辅助函数 ---------------------------------------------------------
    def _decode(self, ms, os):
        """左移解码当前解。

        :param ms: 机器选择段
        :param os: 工序排序段
        :return: 左移解码结果 ``(S, C, machine_seq, chosen_machine)``
        :rtype: tuple
        """
        return left_shift_decode(self.inst, ms, os)

    def _op_position(self, os, op_id) -> int:
        """工序 ``op_id`` 在 OS 段中的出现位置。

        :param os: 工序排序段
        :param op_id: 全局工序号
        :return: 该工序在 OS 中的下标；找不到则抛 ValueError
        :rtype: int
        """
        job = self.inst.job_of_op[op_id]
        oi = self.inst.op_index_in_job[op_id]
        cnt = -1
        for p, j in enumerate(os):
            if j == job:
                cnt += 1
                if cnt == oi:
                    return p
        raise ValueError("op not found in os")

    def _critical_ops(self, S, C, mseq, cm):
        """沿 0 等待时间边回溯求出关键路径上的工序集合。

        :param S: 各工序开始时间
        :param C: 各工序完工时间
        :param mseq: 各机器工序顺序（``mseq[m]`` 为机器 m 上的工序号列表）
        :param cm: 各工序实际分配的机器号（0 基）
        :return: 关键路径上的工序号集合
        :rtype: set
        """
        makespan = max(C)
        crit = set()
        stack = [a for a in range(self.inst.total_ops)
                 if abs(C[a] - makespan) < 1e-9]
        while stack:
            a = stack.pop()
            if a in crit:
                continue
            crit.add(a)
            job = self.inst.job_of_op[a]
            oi = self.inst.op_index_in_job[a]
            if oi > 0:
                jp = self.inst.job_op_ids[job][oi - 1]
                if abs(C[jp] - S[a]) < 1e-9:
                    stack.append(jp)
            m = cm[a]
            seq = mseq[m]
            idx = seq.index(a)
            if idx > 0:
                mp = seq[idx - 1]
                if abs(C[mp] - S[a]) < 1e-9:
                    stack.append(mp)
        return crit

    # ---- L1：关键路径工序换更快机器（改 MS） ------------------------------
    def _L1(self, ms, os):
        """:return: (new_ms, new_os) 或 (None, None)。"""
        S, C, mseq, cm = self._decode(ms, os)
        crit = self._critical_ops(S, C, mseq, cm)
        cands = []
        for a in crit:
            fastest = min(range(len(self.inst.times[a])),
                          key=lambda k: self.inst.times[a][k])
            if ms[a] != fastest:
                cands.append((a, fastest))
        if not cands:
            return None, None
        a, fastest = cands[int(self.rng.integers(0, len(cands)))]
        new_ms = list(ms)
        new_ms[a] = fastest
        return new_ms, list(os)

    # ---- L2：瓶颈机上交换相邻两道工序（改 OS） ----------------------------
    def _L2(self, ms, os):
        """:return: (new_ms, new_os) 或 (None, None)。"""
        S, C, mseq, cm = self._decode(ms, os)
        mcomp = [max([C[a] for a in seq]) if seq else -1.0 for seq in mseq]
        bottleneck = int(np.argmax(mcomp))
        seq = mseq[bottleneck]
        pairs = [(o1, o2) for o1, o2 in zip(seq, seq[1:])
                 if self.inst.job_of_op[o1] != self.inst.job_of_op[o2]]
        if not pairs:
            return None, None
        o1, o2 = pairs[int(self.rng.integers(0, len(pairs)))]
        p1 = self._op_position(os, o1)
        p2 = self._op_position(os, o2)
        new_os = list(os)
        new_os[p1], new_os[p2] = new_os[p2], new_os[p1]
        return list(ms), new_os

    # ---- L3：把一道工序前插到更早的空闲间隙（改 OS） ----------------------
    def _L3(self, ms, os):
        """:return: (new_ms, new_os) 或 (None, None)。"""
        S, C, mseq, cm = self._decode(ms, os)
        gap_pairs = []
        for seq in mseq:
            for o1, o2 in zip(seq, seq[1:]):
                if (self.inst.job_of_op[o1] != self.inst.job_of_op[o2]
                        and S[o2] > C[o1] + 1e-9):
                    gap_pairs.append((o1, o2))
        if not gap_pairs:
            return None, None
        o1, o2 = gap_pairs[int(self.rng.integers(0, len(gap_pairs)))]
        p1 = self._op_position(os, o1)
        p2 = self._op_position(os, o2)
        new_os = list(os)
        new_os[p1], new_os[p2] = new_os[p2], new_os[p1]
        return list(ms), new_os

    # ---- R1：早到工件工序换慢机 / 后移（改 MS/OS） ------------------------
    def _R1(self, ms, os):
        """:return: (new_ms, new_os) 或 (None, None)。"""
        S, C, mseq, cm = self._decode(ms, os)
        job_comp = [C[self.inst.last_op_of_job[j]]
                    for j in range(self.inst.n_jobs)]
        # 早到工件：完工时间早于交货期
        early = [j for j in range(self.inst.n_jobs)
                 if job_comp[j] < self.due[j] - 1e-9]
        if not early:
            return None, None
        j = early[int(self.rng.integers(0, len(early)))]
        op_ids = self.inst.job_op_ids[j]
        if self.rng.random() < 0.5:
            # R1a：把某道工序换到最慢可选机器（改 MS）
            cands = []
            for a in op_ids:
                times = self.inst.times[a]
                slowest = max(range(len(times)), key=lambda k: times[k])
                if ms[a] != slowest:
                    cands.append((a, slowest))
            if not cands:
                return None, None
            a, slowest = cands[int(self.rng.integers(0, len(cands)))]
            new_ms = list(ms)
            new_ms[a] = slowest
            return new_ms, list(os)
        # R1b：把某道工序与其机器序列中靠后的工序交换（改 OS，实现后移）
        moves = []
        for a in op_ids:
            m = cm[a]
            seq = mseq[m]
            idx = seq.index(a)
            for later in seq[idx + 1:]:
                if self.inst.job_of_op[later] != self.inst.job_of_op[a]:
                    moves.append((a, later))
        if not moves:
            return None, None
        a, later = moves[int(self.rng.integers(0, len(moves)))]
        p1 = self._op_position(os, a)
        p2 = self._op_position(os, later)
        new_os = list(os)
        new_os[p1], new_os[p2] = new_os[p2], new_os[p1]
        return list(ms), new_os
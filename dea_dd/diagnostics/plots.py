"""matplotlib 绘图：5 张诊断图（用户需求 #4），相近的 w 点连线（需求 #6）。

所有图以 PNG 保存到输出目录；使用 Agg 后端，无头环境也可用。

    ① EO makespan 与 TWTE 随迭代变化     ② EO 各 w 的帕累托前沿
    ③ EO 初始种群前沿                     ④ NSGA-II makespan 与 TWTE 随代数变化
    ⑤ 最终帕累托前沿
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# 中文字体：默认 DejaVu 无 CJK 字形，中文标签会显示为缺字方框；
# 优先微软雅黑，其次黑体，最后回退默认字体。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _save(fig, out_dir, name):
    """保存 figure 为 PNG 并释放句柄。

    :param fig: matplotlib Figure 对象
    :param out_dir: 输出目录（自动创建）
    :param name: 文件名（含 .png 后缀）
    :return: 保存后的完整文件路径
    :rtype: str
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_eo_convergence(conv, out_dir):
    """绘制 EO 阶段各 w 的 makespan/TWTE 随迭代变化（图①）。

    :param conv: 字典 ``{w: [(iter, f1, f2), ...]}``
    :param out_dir: 输出目录
    :return: 保存路径 ``out_dir/eo_convergence.png``
    :rtype: str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for w, seq in conv.items():
        iters = [s[0] for s in seq]
        axes[0].plot(iters, [s[1] for s in seq], marker=".", label=f"w={w:.2f}")
        axes[1].plot(iters, [s[2] for s in seq], marker=".", label=f"w={w:.2f}")
    axes[0].set_xlabel("EO 迭代次数")
    axes[0].set_ylabel("makespan")
    axes[0].set_title("EO：makespan 随迭代变化")
    axes[1].set_xlabel("EO 迭代次数")
    axes[1].set_ylabel("TWTE")
    axes[1].set_title("EO：TWTE 随迭代变化")
    axes[1].legend(fontsize="small")
    return _save(fig, out_dir, "eo_convergence.png")


def plot_eo_w_fronts(w_fronts, out_dir):
    """绘制 EO 各 w 的帕累托前沿（图②）；同一 w 的点按 f1 升序连线（需求 #6）。

    :param w_fronts: 字典 ``{w: [(f1, f2), ...]}``
    :param out_dir: 输出目录
    :return: 保存路径 ``out_dir/eo_w_fronts.png``
    :rtype: str
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for w, pts in w_fronts.items():
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: p[0])   # 按 makespan 升序，便于连前沿
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", markersize=4, label=f"w={w:.2f}")
    ax.set_xlabel("makespan")
    ax.set_ylabel("TWTE")
    ax.set_title("EO 各 w 的帕累托前沿")
    ax.legend(fontsize="small")
    return _save(fig, out_dir, "eo_w_fronts.png")


def plot_initial_front(points, out_dir, ws=None):
    """绘制 EO 初始种群前沿（图③）。

    :param points: 前沿点 ``[(f1, f2), ...]``
    :param out_dir: 输出目录
    :param ws: 可选，与 points 等长的来源权重列表；给定时按 w 分组连线
    :return: 保存路径 ``out_dir/eo_initial_front.png``
    :rtype: str
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if ws is not None and len(ws) == len(points) and points:
        groups = {}
        for p, w in zip(points, ws):
            groups.setdefault(w, []).append(p)
        for w, pts in groups.items():
            pts = sorted(pts, key=lambda p: p[0])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            label = "随机补齐" if w is None else f"w={w:.2f}"
            ax.plot(xs, ys, marker="o", markersize=4, label=label)
        ax.legend(fontsize="small")
    elif points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=40, color="tab:blue")
    ax.set_xlabel("makespan")
    ax.set_ylabel("TWTE")
    ax.set_title("EO 初始种群帕累托前沿")
    return _save(fig, out_dir, "eo_initial_front.png")


def plot_nsga2_convergence(conv, out_dir):
    """绘制 NSGA-II 阶段最优 makespan/TWTE 随代数变化（图④）。

    :param conv: 列表 ``[(gen, best_f1, best_f2), ...]``
    :param out_dir: 输出目录
    :return: 保存路径 ``out_dir/nsga2_convergence.png``
    :rtype: str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gens = [c[0] for c in conv]
    axes[0].plot(gens, [c[1] for c in conv], marker=".", color="tab:red")
    axes[1].plot(gens, [c[2] for c in conv], marker=".", color="tab:green")
    axes[0].set_xlabel("代数")
    axes[0].set_ylabel("最优 makespan")
    axes[0].set_title("NSGA-II：最优 makespan 随代数变化")
    axes[1].set_xlabel("代数")
    axes[1].set_ylabel("最优 TWTE")
    axes[1].set_title("NSGA-II：最优 TWTE 随代数变化")
    return _save(fig, out_dir, "nsga2_convergence.png")


def plot_final_front(points, out_dir):
    """绘制最终帕累托前沿（图⑤）。

    :param points: 前沿点 ``[(f1_hat, f2_hat), ...]``
    :param out_dir: 输出目录
    :return: 保存路径 ``out_dir/final_front.png``
    :rtype: str
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.scatter(xs, ys, s=50, color="tab:purple")
        # 按 makespan 排序连出前沿形状
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        ax.plot([xs[i] for i in order], [ys[i] for i in order],
                color="tab:purple", alpha=0.3, linewidth=1)
    ax.set_xlabel("makespan（期望）")
    ax.set_ylabel("TWTE（期望）")
    ax.set_title("最终帕累托前沿（R=10000 长仿真）")
    return _save(fig, out_dir, "final_front.png")
"""项目根目录与输出路径解析（需求：控制台与 IDE 执行落到同一路径）。

用 ``__file__`` 推导 DEA-DD 项目根目录，把相对输出路径锚定到项目根，
这样无论从控制台（任意 cwd）还是代码编译器（PyCharm）启动，日志与图片
都写到同一绝对路径，不受当前工作目录 cwd 的影响。
"""
from __future__ import annotations

from pathlib import Path

# DEA-DD 项目根目录 = 本文件所在 dea_dd/ 目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_project_path(path: str) -> str:
    """把路径解析为绝对路径字符串。

    :param path: 输出目录路径（相对或绝对）
    :return: 绝对路径字符串；相对路径按 ``PROJECT_ROOT / path`` 锚定到项目根
    :rtype: str
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((PROJECT_ROOT / p).resolve())
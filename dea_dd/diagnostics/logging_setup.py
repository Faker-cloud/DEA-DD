"""日志配置：控制台 + 文件双输出，全程不使用 ``print``（用户需求）。"""
from __future__ import annotations

import logging
import os


def setup_logging(log_dir="logs", name="dea_dd", level=logging.INFO):
    """配置名为 ``name`` 的全局 logger，同时输出到控制台与日志文件。

    每次调用移除旧 handler 再重建，避免重复追加。不同 ``name`` 得到互不
    干扰的 logger 与日志文件，文件名固定为 ``<name>.log``（需求：不同算法
    日志写到不同文件）。

    :param log_dir: 日志文件目录（缺省 ``logs``），自动创建
    :param name: 日志文件名与记录器名（缺省 ``dea_dd``）
    :param level: 日志级别（缺省 ``logging.INFO``）
    :return: 配置好的全局 logger（名 ``name``）
    :rtype: logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{name}.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
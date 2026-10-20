"""日志配置：控制台 + 文件双输出，全程不使用 ``print``（用户需求）。"""
from __future__ import annotations

import logging
import os


def setup_logging(log_dir="logs", level=logging.INFO):
    """配置名为 ``dea_dd`` 的全局 logger，同时输出到控制台与日志文件。

    每次调用移除旧 handler 再重建，避免重复追加。

    :param log_dir: 日志文件目录（缺省 ``logs``），自动创建
    :param level: 日志级别（缺省 ``logging.INFO``）
    :return: 配置好的全局 logger（名 ``dea_dd``）
    :rtype: logging.Logger
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("dea_dd")
    logger.setLevel(level)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(
        os.path.join(log_dir, "dea_dd.log"), encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
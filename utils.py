import os
import sys
import torch
import logging
import warnings
from rdkit import rdBase
from typing import List, Tuple, Any


def setup_logging(log_filename="monitor/monitor.log"):
    # 1. Tạo Logger riêng biệt (không dùng root logger để tránh xung đột với RDKit/Torch)
    logger = logging.getLogger("ExperimentLogger")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Quan trọng: Ngăn không cho log trôi về root (nơi RDKit hay chiếm dụng)

    # Xóa các handler cũ nếu có (tránh bị log đôi hoặc lỗi handler chết)
    if logger.hasHandlers():
        logger.handlers.clear()

    # 2. Định dạng log
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 3. Handler ghi ra màn hình (Terminal)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    # 4. Handler ghi vào File (QUAN TRỌNG: mode='a' và buffering=1)
    # buffering=1 nghĩa là cứ 1 dòng là ghi xuống đĩa ngay lập tức (Line Buffering)
    try:
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {e}")

    return logger


def configure_warnings_and_logs(
    ignore_warnings: bool = False, disable_rdkit_logs: bool = False
) -> None:
    """
    Configures warning and logging behaviors for Python and RDKit. This function allows
    selective suppression of Python warnings and disabling of RDKit error and warning logs
    based on the parameters provided.

    Parameters
    ----------
    ignore_warnings : bool, optional
        If True, suppresses all Python warnings. If False, normal operation of warnings.
        Default is False.
    disable_rdkit_logs : bool, optional
        If True, disables RDKit error and warning logs. If False, logs operate normally.
        Default is False.

    Usage
    -----
    This function should be used at the start of scripts where control over warning and
    logging verbosity is needed. It is particularly useful in production or testing phases
    to reduce output clutter but should be used cautiously during development to
    avoid overlooking important warnings or errors.
    """
    if ignore_warnings:
        # Suppress all Python warnings (e.g., DeprecationWarning, RuntimeWarning)
        warnings.filterwarnings("ignore")
    else:
        # Reset the warnings to default behavior (i.e., printing all warnings)
        warnings.resetwarnings()

    if disable_rdkit_logs:
        # Disable RDKit error and warning logs
        rdBase.DisableLog("rdApp.error")
        rdBase.DisableLog("rdApp.warning")
    else:
        # Enable RDKit error and warning logs if they were previously disabled
        rdBase.EnableLog("rdApp.error")
        rdBase.EnableLog("rdApp.warning")



"""安全落盘与可暂停的 Windows 共享冲突重试。

每次写入使用唯一临时文件，完成后原子替换。只对 Windows 暂时拒绝访问、
共享冲突重试；其他错误直接向调用方报告。进程异常退出时旧文件仍可用。
"""
from __future__ import annotations
import logging
import os
import threading
from pathlib import Path
from typing import Callable
from uuid import uuid4

local = threading.local()

class PauseRequested(Exception):
    """用户请求暂停；调用方在保存的区域边界退出。"""


def install_context(event: threading.Event, notify: Callable[[str], None]) -> None:
    local.event, local.notify = event, notify


def check_pause() -> None:
    event = getattr(local, 'event', None)
    if event is not None and event.is_set():
        raise PauseRequested('用户请求暂停')


def retryable(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and getattr(exc, 'winerror', None) in (5, 32, 33)


def wait_retry(exc: BaseException) -> None:
    """重试间隔 5 秒；期间用户请求暂停会立即唤醒。"""
    logging.warning('文件暂时不可访问，5 秒后重试: %s', exc)
    notify = getattr(local, 'notify', None)
    if notify: notify(f'文件暂时不可访问，5 秒后重试：{exc}')
    event = getattr(local, 'event', None)
    if event is not None:
        if event.wait(5): raise PauseRequested('用户请求暂停')
    else:
        threading.Event().wait(5)


def atomic_bytes(path: Path, data: bytes) -> None:
    """写入新临时文件并原子替换；异常前不破坏旧文件。"""
    temp = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    ready = False
    try:
        while True:
            check_pause()
            try:
                if not ready:
                    with temp.open('wb') as f:
                        f.write(data)
                        f.flush()
                    ready = True
                os.replace(temp, path)
                return
            except OSError as exc:
                if not retryable(exc): raise
                wait_retry(exc)
    finally:
        # 删除本次生成的临时文件；不会删除已有正式结果。
        try: temp.unlink(missing_ok=True)
        except OSError: logging.warning('保留未提交的临时文件 %s', temp)


def atomic_text(path: Path, text: str) -> None:
    atomic_bytes(path, text.encode('utf-8'))


def read_text(path: Path) -> str:
    """读取关键状态；共享锁冲突时按同一策略等待，其他错误原样抛出。"""
    while True:
        check_pause()
        try:
            return path.read_text(encoding='utf-8')
        except OSError as exc:
            if not retryable(exc): raise
            wait_retry(exc)

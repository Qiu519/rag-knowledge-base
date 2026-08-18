# -*- coding: utf-8 -*-
"""隐藏启动 launcher。

目的：完全绕开 cmd `start "" pythonw.exe ...` 在本机上的边角问题
（曾弹 "Windows 找不到 '\\\\' 文件" 对话框）。

机制：
  1) bat 同步调用 `pythonw.exe app\\launcher.py`（约 1-2 秒后 launcher 退出，bat 继续）
  2) launcher 自己的 stdout/stderr 重定向到 outputs\\launcher.log
  3) launcher 用 subprocess.Popen + DETACHED_PROCESS + CREATE_NO_WINDOW 拉起 web_ui
     （不把 launcher 的日志句柄传过去，避免 Windows 上句柄继承竞态；
      web_ui 自己接管 outputs\\server.log）
  4) launcher 立即退出，web_ui 完全独立在后台跑

全程不调用 cmd 的 `start`，避免一切 cmd `start` + pythonw 的解析怪癖。
"""
from __future__ import annotations

import datetime
import subprocess
import sys
import traceback
from pathlib import Path

# --- 自身日志兜底（launcher 跑在 pythonw 下，stdout/stderr 原本为 None） ---
# 关键：写横幅时用 file=_lf 显式指定文件，不依赖 sys.stdout。
# 不同父 std handle 继承场景下 PYW 的 sys.stdout 可能是 None / NUL 包装 /
# PIPE 包装，写横幅不能用 print()，否则日志会丢。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_LOG = PROJECT_ROOT / "outputs" / "launcher.log"
LAUNCHER_LOG.parent.mkdir(parents=True, exist_ok=True)
LAUNCHER_LOG.write_text("", encoding="utf-8")  # 截断
_lf = open(LAUNCHER_LOG, "a", encoding="utf-8", buffering=1)
if sys.stdout is None:
    sys.stdout = _lf
if sys.stderr is None:
    sys.stderr = _lf

def _log(msg: str) -> None:
    # 显式 file=_lf：不依赖 sys.stdout 当前指向哪里
    print(msg, flush=True, file=_lf)

_log(
    f"=== launcher.py started "
    f"{datetime.datetime.now().isoformat(timespec='seconds')} ==="
)

WEB_UI = PROJECT_ROOT / "app" / "web_ui.py"
PYW = r"S:\anaconda3\envs\rag\pythonw.exe"

try:
    _log(f"=== launching: {PYW} {WEB_UI} ===")
    p = subprocess.Popen(
        [PYW, str(WEB_UI)],
        # DO NOT pass stdout/stderr/stdin: launcher is pythonw (GUI, std handles
        # are None/invalid), and Popen with DEVNULL would wrap NUL into a
        # TextIOWrapper — making web_ui's sys.stdout non-None, so its top-of-file
        # redirect `if sys.stdout is None` would NOT trigger and logs would be
        # silently swallowed into NUL. Inheriting launcher (None) handles keeps
        # web_ui's sys.stdout truly None so its own log redirect fires.
        cwd=str(PROJECT_ROOT),
        close_fds=True,
        creationflags=(
            subprocess.DETACHED_PROCESS  # 0x00000008: 脱离父进程
            | subprocess.CREATE_NO_WINDOW  # 0x08000000: 不创建新控制台窗口
        ),
    )
    _log(f"=== web_ui.py launched, pid={p.pid} ===")
except Exception:
    _log("=== launcher failed ===")
    _log(traceback.format_exc())
    raise
finally:
    _lf.close()
# launcher 退出，web_ui 在后台独立运行

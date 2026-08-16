# -*- coding: utf-8 -*-
"""生成 启动/停止 bat（ASCII 内容 + CRLF 行尾，raw 字符串杜绝转义事故）。"""

START = r"""@echo off
chcp 65001 >nul
cd /d S:\WorkBuddy-Workspace\rag-knowledge-base

rem ---- local LLM: start on demand if not running ----
curl -s -o nul --max-time 3 http://127.0.0.1:11434/api/version
if errorlevel 1 (
    echo [start] local LLM server - Ollama ...
    set OLLAMA_MODELS=S:\Ollama\models
    start "ollama-serve" /min S:\Ollama\ollama.exe serve
    ping -n 5 127.0.0.1 >nul
)

echo [start] RAG Knowledge Base QA - loading models, browser will open ...
S:\anaconda3\envs\rag\python.exe app\web_ui.py
pause
"""

STOP = r"""@echo off
chcp 65001 >nul
echo [stop] closing QA app - web_ui ...
wmic process where "commandline like '%%web_ui.py%%'" delete >nul 2>&1
echo [stop] closing local LLM server - ollama ...
taskkill /IM ollama.exe /F >nul 2>&1
echo [done] all stopped. GPU memory released.
pause
"""

for name, content in [("启动知识库问答.bat", START), ("停止知识库问答.bat", STOP)]:
    # ASCII 自检：bat 内容必须纯 ASCII，任何非 ASCII 字节立即报错
    content.encode("ascii")
    with open(name, "w", encoding="ascii", newline="") as f:
        f.write(content.replace("\n", "\r\n"))
    print(name, "OK", len(content), "chars")

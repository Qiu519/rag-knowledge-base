@echo off
chcp 65001 >nul
cd /d S:\WorkBuddy-Workspace\rag-knowledge-base

rem ---- 1) start local LLM (Ollama) on demand ----
curl -s -o nul --max-time 3 http://127.0.0.1:11434/api/version
if errorlevel 1 (
    if exist S:\Ollama\ollama.exe (
        echo [start] local LLM server - Ollama ...
        set OLLAMA_MODELS=S:\Ollama\models
        start /min S:\Ollama\ollama.exe serve
        ping -n 5 127.0.0.1 >nul
    ) else (
        echo [skip] Ollama not installed at S:\Ollama\ollama.exe - skipping
    )
)

rem ---- 2) launch the RAG server in fully hidden mode ----
rem Strategy: bat calls pythonw.exe launcher.py synchronously (no cmd "start").
rem launcher.py internally uses subprocess.Popen + DETACHED_PROCESS to launch
rem web_ui.py and redirects stdout/stderr to outputs\server.log. Then launcher
rem exits within ~1-2s and the bat continues to the port poll.
rem This avoids cmd's "start" + pythonw edge cases that previously raised a
rem "Windows cannot find '\\\' file" dialog.
echo [start] RAG Knowledge Base QA - launching server ...
"S:\anaconda3\envs\rag\pythonw.exe" "S:\WorkBuddy-Workspace\rag-knowledge-base\app\launcher.py"

rem ---- 3) wait for the port to be ready (up to ~90s) ----
echo [wait] waiting for server to be ready ...
set /a "tries=0"
:waitloop
curl -s -o nul --max-time 2 http://127.0.0.1:7860/ >nul 2>&1
if not errorlevel 1 goto server_up
set /a "tries+=1"
if %tries% geq 30 (
    echo [warn] server not responding within 60s, check outputs\server.log
    echo [warn] if no server.log was written, the launch failed before logging
    echo [warn]  - try running this in a console to see the real error:
    echo [warn]    S:\anaconda3\envs\rag\python.exe app\launcher.py
    goto end
)
ping -n 2 127.0.0.1 >nul
goto waitloop

:server_up
echo [open] server is up - opening browser ...
start "" http://127.0.0.1:7860

:end
echo [done] server started. Browser should be open now.
echo         Hidden: no window, no taskbar; logs in outputs\server.log
echo         Stop: 停止知识库问答.bat
timeout /t 3 /nobreak >nul

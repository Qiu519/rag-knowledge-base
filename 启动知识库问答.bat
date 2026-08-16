@echo off
chcp 65001 >nul
cd /d S:\WorkBuddy-Workspace\rag-knowledge-base

rem ---- 本地 LLM（Ollama）按需拉起：没在跑就静默启动 ----
curl -s -o nul --max-time 3 http://127.0.0.1:11434/api/version
if errorlevel 1 (
    echo [start] local LLM server (Ollama)...
    set OLLAMA_MODELS=S:\Ollama\models
    start "ollama-serve" /min S:\Ollama\ollama.exe serve
    timeout /t 4 /nobreak >nul
)

echo [start] RAG Knowledge Base QA - loading models, browser will open...
S:\anaconda3\envs\rag\python.exe app\web_ui.py
pause

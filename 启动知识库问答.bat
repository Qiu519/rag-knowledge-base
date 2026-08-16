@echo off
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

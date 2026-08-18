@echo off
chcp 65001 >nul
echo [stop] closing QA app - web_ui ...
wmic process where "commandline like '%%web_ui.py%%'" delete >nul 2>&1
echo [stop] closing local LLM server - ollama ...
taskkill /IM ollama.exe /F >nul 2>&1
echo [done] all stopped. GPU memory released.
timeout /t 2 /nobreak >nul

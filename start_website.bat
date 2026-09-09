@echo off
title Economic Crash Early-Warning System - 1-Click Server & Public Tunnel Launcher
echo =======================================================================
echo 🛡️ Starting Economic Crash Early-Warning System...
echo =======================================================================
cd /d "%~dp0"

echo 1. Starting Local Web Engine Server...
start /b .venv\Scripts\streamlit.exe run app\dashboard\main.py --server.port 8501 --server.headless true

timeout /t 3 /nobreak >nul

echo 2. Launching Live HTTPS Global Public Tunnel...
.\cloudflared.exe tunnel --url http://localhost:8501

pause

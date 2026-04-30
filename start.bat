@echo off
setlocal
title Support Bot v1
cd /d "%~dp0"

set "SUPPORT_BOT_PORT=8009"
set "PYTHONUTF8=1"
set "GEMMA_REWRITE_ENABLED=1"
set "GEMMA_REWRITE_TIMEOUT_SECONDS=180"
set "SUPPORT_BOT_ARTIFACT_DIR=%~dp0artifacts\tr_bert_uncased_epoch9"

if exist "%~dp0.venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

echo Eski 8009 server surecleri kapatiliyor...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$connections = Get-NetTCPConnection -LocalPort %SUPPORT_BOT_PORT% -ErrorAction SilentlyContinue; foreach ($connection in $connections) { Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue }"

echo Support Bot v1 baslatiliyor...
echo Python: %PYTHON_EXE%
echo Model : %SUPPORT_BOT_ARTIFACT_DIR%
echo Adres : http://127.0.0.1:%SUPPORT_BOT_PORT%

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%SUPPORT_BOT_PORT%'"

"%PYTHON_EXE%" -u server.py

echo.
echo Server kapandi veya hata olustu.
pause


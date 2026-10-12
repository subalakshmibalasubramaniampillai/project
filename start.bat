@echo off
rem StayTrace for Windows: double-click this file.
rem It stops old copies of the site, rebuilds it, starts it and opens it.
setlocal
cd /d "%~dp0"
title StayTrace

echo.
echo [1/4] Stopping old copies of the site...
for %%P in (8000 5173 5174 5175 5176 5177 5178 5179 4173) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr LISTENING ^| findstr /C:":%%P "') do (
        taskkill /PID %%I /F >nul 2>&1
    )
)

echo.
echo [2/4] Checking Python...
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 from https://www.python.org and run this file again.
    goto fail
)
python -c "import fastapi, uvicorn, torch, lightgbm, xgboost, sklearn, pandas" >nul 2>&1
if errorlevel 1 (
    echo Installing Python packages. This can take a few minutes the first time...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto fail
)

echo.
echo [3/4] Building the website...
where npm >nul 2>&1
if errorlevel 1 (
    echo Node.js was not found. Install it from https://nodejs.org and run this file again.
    goto fail
)
pushd web
call npm install --no-audit --no-fund
if errorlevel 1 (
    popd
    goto fail
)
call npm run build
if errorlevel 1 (
    popd
    goto fail
)
popd

echo.
echo [4/4] Starting StayTrace at http://127.0.0.1:8000
echo The site opens in your browser in a few seconds.
echo Keep this window open while you use it. Close it to stop StayTrace.
echo.
start "" /min powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -Command "$ProgressPreference = 'SilentlyContinue'; for ($i = 0; $i -lt 120; $i++) { try { Invoke-WebRequest 'http://127.0.0.1:8000/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; break } catch { Start-Sleep -Seconds 1 } }; Start-Process ('http://127.0.0.1:8000/?v=' + (Get-Date -Format 'HHmmss'))"
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
goto end

:fail
echo.
echo StayTrace could not start. The message above says why.

:end
echo.
pause

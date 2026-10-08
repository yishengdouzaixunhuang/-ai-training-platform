@echo off
rem ============================================================
rem  Farasis 3-view segmentation toolkit launcher
rem  Kept PURE ASCII on purpose: a .bat containing Chinese must be
rem  saved as GBK to display correctly in cmd, which breaks editing.
rem  All Chinese UI lives in tools\menu.py (UTF-8 native).
rem ============================================================
setlocal
set "ROOT=%~dp0"

rem auto-detect layout: portable bundle (app\ + python\) vs dev checkout (sources at root)
if exist "%ROOT%app\main.py" (
  set "APPDIR=%ROOT%app"
  set "PY=%ROOT%python\python.exe"
) else (
  set "APPDIR=%ROOT%"
  set "PY=C:\aitp_venv\Scripts\python.exe"
)
if not exist "%PY%" set "PY=python.exe"

pushd "%APPDIR%"
if errorlevel 1 (
  echo [ERROR] cannot enter program directory: %APPDIR%
  pause
  exit /b 1
)
"%PY%" "tools\menu.py"
popd
endlocal

@echo off
cd /d "%~dp0"
echo =============================================
echo    AI Training Platform - Setup and Launch
echo =============================================
echo.

set "BASE_PY=C:\Users\34928\AppData\Local\Programs\Python\Python310\python.exe"
if not exist "%BASE_PY%" set "BASE_PY=C:\Python310\python.exe"
if not exist "%BASE_PY%" set "BASE_PY=C:\Users\34928\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe"

echo Base Python: %BASE_PY%
if not exist "%BASE_PY%" echo [ERROR] No Python 3.10+ found. Install from https://www.python.org/downloads/ (tick "Add python.exe to PATH") and run again.
echo.

echo [1/4] Creating a fresh virtual environment .venv310 ...
"%BASE_PY%" -m venv .venv310

set "PY=.venv310\Scripts\python.exe"
echo Venv Python: %PY%
echo.

echo [2/4] Upgrading pip ...
"%PY%" -m pip install --upgrade pip

echo [3/4] Installing dependencies (large download ~1GB, please wait) ...
"%PY%" -m pip install -r requirements.txt
"%PY%" -m pip install onnxruntime rapidocr-onnxruntime ultralytics scikit-learn openpyxl

echo.
echo [4/4] Launching platform ...
"%PY%" main.py

echo.
echo =============================================
echo    Done. Exit code: %errorlevel%
echo.
echo    If the window did not open, open a CMD in this
echo    folder and run:  .venv310\Scripts\python.exe main.py
echo    then send me the error text.
echo =============================================
pause

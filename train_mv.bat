@echo off
REM ---------------------------------------------------------------
REM Double-click to run 3-view segmentation training
REM (diffuse + shape1 + shape2 -> 9 channels).
REM
REM The project path and the training params live in _run_train_mv.py,
REM NOT here: cmd.exe parses .bat with the OEM codepage (GBK), so a
REM literal non-ASCII path in this file would be mangled into "?".
REM
REM   train_mv.bat                 default project / 1 epoch / batch 4
REM   train_mv.bat --dry           data precheck only, no training
REM   train_mv.bat "" 3 8          default project / 3 epochs / batch 8
REM ---------------------------------------------------------------
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo.
echo === 3-view training starting ===
echo === log file: E:\05-TrainingProject\_train_logs ===
echo.
"C:\aitp_venv\Scripts\python.exe" -u _run_train_mv.py %*
echo.
echo === finished, exit code %ERRORLEVEL% ===
pause

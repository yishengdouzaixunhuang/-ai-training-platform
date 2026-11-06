@echo off
chcp 936 >nul 2>nul
cd /d "%~dp0"
title AI训练平台

rem ---------------------------------------------------------------
rem 自动找可用的运行环境。只找纯英文路径：
rem   中文路径下 Qt 加载插件会卡死，项目内的 .venv 不能用。
rem 优先级：
rem   1. C:\aitp_venv                     <- 本机现成环境
rem   2. %LOCALAPPDATA%\aitp_venv        <- 离线安装的备选位置
rem   3. PATH 里的 pythonw                <- 兜底
rem ---------------------------------------------------------------

set "PYW="
set "FROM="

if exist "C:\aitp_venv\Scripts\pythonw.exe" (
    set "PYW=C:\aitp_venv\Scripts\pythonw.exe"
    set "FROM=C:\aitp_venv"
)
if not defined PYW if exist "%LOCALAPPDATA%\aitp_venv\Scripts\pythonw.exe" (
    set "PYW=%LOCALAPPDATA%\aitp_venv\Scripts\pythonw.exe"
    set "FROM=%LOCALAPPDATA%\aitp_venv"
)
if not defined PYW (
    for %%P in (pythonw.exe) do if exist "%%~$PATH:P" (
        set "PYW=pythonw.exe"
        set "FROM=PATH 里的 Python"
    )
)

if not defined PYW (
    echo ==================================================
    echo   [X] 没找到可用的 Python 运行环境
    echo.
    echo   双击本目录下的「离线安装.bat」，
    echo   它会用 wheels\ 里的包装一个到 C:\aitp_venv
    echo ==================================================
    echo.
    pause
    exit /b 1
)

echo 使用环境: %FROM%
start "" "%PYW%" "%~dp0main.py"

@echo off
chcp 936 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 离线安装运行环境

echo ==================================================
echo   AI训练平台 - 离线安装 Python 环境
echo   安装源: 本目录 wheels\   约 2.8GB
echo   全程不需要联网
echo ==================================================
echo.
echo 为什么不能装在项目目录里：
echo   项目路径含中文和 ^& 符号，Qt 在那种路径下加载插件
echo   会直接卡死（已实测）。所以环境必须装到纯英文路径。
echo.

if not exist "wheels" (
    echo [X] 没有 wheels\ 目录，项目拷贝可能不完整
    pause
    exit /b 1
)
dir /b "wheels\torch-*.whl" >nul 2>nul
if errorlevel 1 (
    echo [X] wheels\ 里没有 torch 的 wheel，拷贝不完整
    pause
    exit /b 1
)

rem ---------------------------------------------------------------
rem 已经有能用的环境就不用再装
rem ---------------------------------------------------------------
if exist "C:\aitp_venv\Scripts\python.exe" (
    "C:\aitp_venv\Scripts\python.exe" -c "import torch, PyQt5" >nul 2>nul
    if not errorlevel 1 (
        echo [OK] 已存在可用环境: C:\aitp_venv
        echo      直接双击「启动_便携.bat」运行即可，无需安装
        echo.
        pause
        exit /b 0
    )
)

rem ---------------------------------------------------------------
rem wheels 里全是 cp312，必须 Python 3.12
rem ---------------------------------------------------------------
set "BASE="
if exist "C:\Python312\python.exe" set "BASE=C:\Python312\python.exe"
if not defined BASE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "BASE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined BASE (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "BASE=py -3.12"
)
if not defined BASE (
    echo [X] 没找到 Python 3.12
    echo.
    echo     wheels\ 里的包是 cp312，必须用 Python 3.12
    echo     下载: https://www.python.org/downloads/release/python-3127/
    echo     安装时勾选 "Add python.exe to PATH"，装完再运行本脚本
    echo.
    pause
    exit /b 1
)

echo 基础解释器: %BASE%
%BASE% -c "import sys;print('版本:', sys.version.split()[0])"
echo.

rem ---------------------------------------------------------------
rem 选目标目录：优先 C:\aitp_venv，冲突则退到 LOCALAPPDATA
rem ---------------------------------------------------------------
set "TARGET=C:\aitp_venv"
if exist "%TARGET%\Scripts\python.exe" (
    set "TARGET=%LOCALAPPDATA%\aitp_venv"
    echo [!] C:\aitp_venv 存在但不可用，改装到: !TARGET!
)
echo 目标目录: !TARGET!
echo.

if exist "!TARGET!\Scripts\python.exe" (
    echo [1/2] 目录已存在，跳过创建
) else (
    echo [1/2] 创建虚拟环境 ...
    %BASE% -m venv "!TARGET!"
    if errorlevel 1 (
        echo [X] 创建虚拟环境失败
        pause
        exit /b 1
    )
)

echo [2/2] 从 wheels\ 安装依赖，约 2.8GB，请耐心等待 ...
echo.
"!TARGET!\Scripts\python.exe" -m pip install --no-index --find-links "%~dp0wheels" --no-input -r "%~dp0requirements-lock.txt"

if errorlevel 1 (
    echo.
    echo [X] 安装失败。常见原因：
    echo     1. Python 版本不是 3.12（wheel 是 cp312）
    echo     2. wheels\ 里有文件拷贝损坏
    echo.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   [OK] 环境装好了: !TARGET!
echo   双击「启动_便携.bat」运行
echo ==================================================
echo.
pause
endlocal

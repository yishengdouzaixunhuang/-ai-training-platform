@echo off
chcp 936 >nul 2>nul
cd /d "%~dp0"
title 安装 vendor 依赖

echo ==================================================
echo   把 OCR / ONNX 相关依赖装到项目内的 vendor\
echo   装完这个文件夹就能整体拷走，不用再 pip install
echo ==================================================
echo.

set "PY="
if exist "C:\aitp_venv\Scripts\python.exe" set "PY=C:\aitp_venv\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%~dp0.venv310\Scripts\python.exe" set "PY=%~dp0.venv310\Scripts\python.exe"
if not defined PY (
    for %%P in (python.exe) do if exist "%%~$PATH:P" set "PY=python"
)
if not defined PY (
    echo [X] 没找到 Python，先装 Python 3.10+ 或建好虚拟环境再跑本脚本
    echo.
    pause
    exit /b 1
)

echo 使用解释器: %PY%
%PY% -c "import sys;print('版本:', sys.version.split()[0])"
echo.

echo 正在安装到 vendor\ ...
%PY% -m pip install --target "%~dp0vendor" --no-deps --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple rapidocr_onnxruntime onnxruntime shapely pyclipper pyyaml

if errorlevel 1 (
    echo.
    echo [X] 安装失败，检查网络后重试。
    echo     也可以手动执行：
    echo     %PY% -m pip install --target "%~dp0vendor" --no-deps rapidocr_onnxruntime onnxruntime shapely pyclipper pyyaml
    echo.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo   [OK] 已安装到 vendor\
echo   验证方式：启动平台 -^> Tools 菜单 -^> 截图 OCR
echo ==================================================
echo.
pause

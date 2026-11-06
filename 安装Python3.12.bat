@echo off
chcp 936 >nul 2>nul
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 安装 Python 3.12

echo ==================================================
echo   安装 Python 3.12
echo   优先用项目内自带的安装包，不需要联网
echo ==================================================
echo.

call :find312
if defined FOUND goto :already

echo 当前没有检测到 Python 3.12，开始安装。
echo.

set "INSTALLER=%~dp0installers\python-3.12.7-amd64.exe"
if exist "%INSTALLER%" goto :use_local

where winget >nul 2>nul
if errorlevel 1 goto :no_source

echo [1/2] 项目内没有安装包，改用 winget 联网安装 ...
winget install --id Python.Python.3.12 -e --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :no_source
goto :verify

:use_local
echo [1/2] 使用项目自带安装包，全程不联网
echo       %INSTALLER%
echo       静默安装到当前用户目录，约 1-2 分钟，请不要关闭窗口 ...
echo.
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1 Shortcuts=0
goto :verify

:verify
echo [2/2] 等待安装完成 ...
set "N=0"
:waitloop
call :find312
if defined FOUND goto :installed
set /a N+=1
if !N! GEQ 18 goto :timeout
timeout /t 5 /nobreak >nul
goto :waitloop

:installed
echo.
echo ==================================================
echo   [OK] Python 3.12 装好了
echo   路径: !FOUND!
!FOUND! -V
echo ==================================================
echo.
echo   下一步：双击「离线安装.bat」装平台依赖
echo.
pause
exit /b 0

:already
echo [OK] 已经装过了: !FOUND!
!FOUND! -V
echo      不需要重复安装
echo.
echo   下一步：双击「离线安装.bat」
echo.
pause
exit /b 0

:timeout
echo.
echo [!] 安装程序返回了，但 90 秒内没找到 python.exe
echo     可能被权限拦了，或装到了非常规位置。
echo     手动安装: https://www.python.org/downloads/release/python-3127/
echo     安装时勾选 "Add python.exe to PATH"
echo.
pause
exit /b 1

:no_source
echo [X] 既没有项目内安装包，也没有 winget
echo.
echo     手动下载: https://www.python.org/downloads/release/python-3127/
echo     选 "Windows installer (64-bit)"，安装时勾 "Add python.exe to PATH"
echo     装完再运行本脚本即可
echo.
pause
exit /b 1

rem ============================================================
rem  查找 Python 3.12（只用无空格路径，方便直接执行）
rem ============================================================
:find312
set "FOUND="
if exist "C:\Python312\python.exe" set "FOUND=C:\Python312\python.exe"
if not defined FOUND if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "FOUND=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined FOUND (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "FOUND=py -3.12"
)
goto :eof

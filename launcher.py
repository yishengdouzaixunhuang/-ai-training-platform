# -*- coding: utf-8 -*-
"""AI训练平台 启动器 —— 打包成小 exe 放桌面，双击就跑软件

它只做一件事：用 C:\\aitp_venv 里的 pythonw 启动源码目录的 main.py。
pythonw 没有控制台窗口，所以不会闪黑框。
"""
import os
import subprocess
import sys

# 源码目录（改这里就能换版本）
源码 = r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程\ai_training_platform_src_20261021\ai_training_platform"
# 专用运行环境
PY = r"C:\aitp_venv\Scripts\pythonw.exe"


def 弹窗(标题, 内容):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, 内容, 标题, 0x10)
    except Exception:
        pass


def main():
    if not os.path.isdir(源码):
        return 弹窗("AI训练平台 启动失败", "找不到源码目录：\n%s\n\n可能是挪了位置，请联系 Arthur。" % 源码)
    if not os.path.isfile(PY):
        return 弹窗("AI训练平台 启动失败",
                    "找不到运行环境：\n%s\n\n需要先建 venv：\n"
                    "  C:\\Python312\\python.exe -m venv C:\\aitp_venv\n"
                    "  C:\\aitp_venv\\Scripts\\pip install -r requirements.txt" % PY)
    主程序 = os.path.join(源码, "main.py")
    if not os.path.isfile(主程序):
        return 弹窗("AI训练平台 启动失败", "找不到 main.py：\n%s" % 主程序)

    日志 = os.path.join(源码, "启动日志.txt")
    with open(日志, "w", encoding="utf-8") as f:
        f.write("启动: %s\n用: %s\n时间: %s\n" % (
            主程序, PY, __import__("time").strftime("%Y-%m-%d %H:%M:%S")))
    try:
        # ⚠️ 这个文件句柄要**一直开着**：子进程的 stdout 指着它，
        #    启动器先退出/关掉句柄的话，软件下一次 print 就会崩（曾经就是这样"闪退"）。
        f = open(日志, "a", encoding="utf-8")
        p = subprocess.Popen([PY, 主程序], cwd=源码,
                             stdout=f, stderr=subprocess.STDOUT,
                             creationflags=0x00000008)   # DETACHED_PROCESS：跟启动器脱钩
        # 等 6 秒，看它是不是刚起来就死了
        try:
            p.wait(timeout=6)
            死了 = True
        except subprocess.TimeoutExpired:
            死了 = False
    except Exception as e:
        return 弹窗("AI训练平台 启动失败", "启动时出错：\n%s" % e)

    if 死了:
        try:
            tail = open(日志, encoding="utf-8", errors="replace").read()[-1500:]
        except Exception:
            tail = "（读不到日志）"
        return 弹窗("AI训练平台 启动失败", "程序刚启动就退出了（退出码 %s）。\n\n日志末尾：\n%s\n\n完整日志：%s"
                    % (p.returncode, tail, 日志))
    # 不关 f：子进程还在用这个句柄（见上面的注释）


if __name__ == "__main__":
    main()

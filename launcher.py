# -*- coding: utf-8 -*-
r"""AI训练平台 启动器 —— 打包成小 exe 放桌面，双击就跑软件

它只做一件事：用专用 venv 里的 pythonw 启动源码目录的 main.py。
pythonw 没有控制台窗口，所以不会闪黑框。

## 路径解析（2026-10-29 重写）

原来源码目录和 python 路径是**写死**的：

    源码 = r"D:\01_D盘\...\ai_training_platform_src_20261021\ai_training_platform"

问题是这个路径里带日期后缀。每归档一个新版本（`..._src_20261105`），
旧 exe 就再也找不到源码，双击只弹「找不到源码目录」。
更麻烦的是它只会弹窗，不会自己找。

现在改成按优先级**自动探测**，写死的那份降级为最后的兜底：

    源码目录：
      1. 环境变量 AITP_HOME
      2. exe / 本脚本所在目录（main.py 就在旁边的情况）
      3. 上面目录的 ai_training_platform 子目录
      4. 上一级目录的 ai_training_platform
      5. 默认路径（老位置，兜底）

    Python：
      1. 环境变量 AITP_PYTHON
      2. 源码目录下的 .venv / .venv312 / venv
      3. C:\aitp_venv（现场约定位置）

想固定某一套环境，设两个环境变量即可，不用重新打包：
    setx AITP_HOME   "D:\\...\\ai_training_platform"
    setx AITP_PYTHON "C:\\aitp_venv\\Scripts\\pythonw.exe"
"""
import os
import subprocess
import sys

# 老位置：仅作最后一档兜底，不再作为唯一来源
默认源码 = r"D:\01_D盘\01_Arthur\20_软件\05-VibeCoding软件&工程\ai_training_platform_src_20261021\ai_training_platform"
默认PY = r"C:\aitp_venv\Scripts\pythonw.exe"


def 弹窗(标题, 内容):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, 内容, 标题, 0x10)
    except Exception:
        pass


def _自身所在目录():
    """打包成 exe 时取 exe 目录；源码运行时取本文件目录。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _是有效源码目录(path):
    """判定标准：main.py 和 ui 包都在，才算一个能跑的源码目录。"""
    if not path or not os.path.isdir(path):
        return False
    return (os.path.isfile(os.path.join(path, "main.py"))
            and os.path.isdir(os.path.join(path, "ui")))


def 找源码目录():
    """按优先级返回 (目录, 候选清单)。找不到有效目录时返回 (None, 清单)。"""
    base = _自身所在目录()
    parent = os.path.dirname(base)
    候选 = []

    env = os.environ.get("AITP_HOME", "").strip()
    if env:
        候选.append(env)

    候选 += [
        base,                                             # exe 和源码同目录
        os.path.join(base, "ai_training_platform"),        # exe 旁边一层
        os.path.join(parent, "ai_training_platform"),      # 上一级
        默认源码,                                          # 老位置兜底
    ]

    去重 = []
    for p in 候选:
        p = os.path.normpath(p)
        if p not in 去重:
            去重.append(p)

    for p in 去重:
        if _是有效源码目录(p):
            return p, 去重
    return None, 去重


def 找python(源码):
    """按优先级找 pythonw.exe。找不到返回 (None, 候选清单)。"""
    候选 = []

    env = os.environ.get("AITP_PYTHON", "").strip()
    if env:
        候选.append(env)

    # 源码目录里自带的 venv（项目里确实有过 .venv312）
    for venv名 in (".venv", ".venv312", "venv"):
        for exe名 in ("pythonw.exe", "python.exe"):
            候选.append(os.path.join(源码 or "", venv名, "Scripts", exe名))

    候选.append(默认PY)

    去重 = []
    for p in 候选:
        p = os.path.normpath(p) if p else p
        if p and p not in 去重:
            去重.append(p)

    for p in 去重:
        # 优先 pythonw（无黑框）；退而求其次 python 也能跑，只是会带控制台
        if os.path.isfile(p):
            return p, 去重
    return None, 去重


def main():
    源码, 源码候选 = 找源码目录()
    if not 源码:
        return 弹窗(
            "AI训练平台 启动失败",
            "找不到源码目录（需要同时含 main.py 和 ui 目录）。\n\n已尝试：\n  %s\n\n"
            "如果源码挪了位置，设一个环境变量即可：\n"
            "  setx AITP_HOME \"新的源码目录\"\n\n"
            "建议把本 exe 直接放在源码目录里，就永远不用改。"
            % "\n  ".join(源码候选))

    PY, py候选 = 找python(源码)
    if not PY:
        return 弹窗(
            "AI训练平台 启动失败",
            "找不到运行环境 pythonw.exe。\n\n已尝试：\n  %s\n\n"
            "需要先建 venv：\n"
            "  C:\\Python312\\python.exe -m venv C:\\aitp_venv\n"
            "  C:\\aitp_venv\\Scripts\\pip install -r requirements.txt\n\n"
            "或直接指定：\n  setx AITP_PYTHON \"...\\pythonw.exe\""
            % "\n  ".join(py候选))

    主程序 = os.path.join(源码, "main.py")
    日志 = os.path.join(源码, "启动日志.txt")
    try:
        with open(日志, "w", encoding="utf-8") as f:
            f.write("启动: %s\n用: %s\n时间: %s\n" % (
                主程序, PY, __import__("time").strftime("%Y-%m-%d %H:%M:%S")))
    except OSError:
        pass   # 日志写不了不影响启动

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

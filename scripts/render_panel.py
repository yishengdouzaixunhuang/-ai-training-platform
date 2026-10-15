# -*- coding: utf-8 -*-
"""render_panel.py —— DL Studio 右栏面板**离屏渲染**（改布局前/后都要先看一眼）

为什么要离屏：
  · 本机不能靠 COM/Office 渲染（会弹窗打扰人）；
  · 直接启动 GUI 自己去截图，又会抢小千手头的鼠标/焦点。
  `grab()` 只把控件渲染进一张 QPixmap，**不开真窗口**，安静且可重复。

用法（**必须**用 aitp_venv 的解释器 + 关掉沙箱）：
    C:/aitp_venv/Scripts/python.exe scripts/render_panel.py            # 全部 4 个页签
    C:/aitp_venv/Scripts/python.exe scripts/render_panel.py params     # 只渲染指定页签
    （页签名 params/annot/model/image —— 见下方 ALL_TABS 的注释，旧名字已废弃）

字体不用管：脚本自己往 offscreen 里挂系统字体（不挂就是一堆空白块）。

⚠️ 为什么必须关沙箱：`import torch` 在沙箱内会
    OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败 —— c10.dll。
⚠️ 为什么路径写 E:/...（不是 /e/...）：Python 不认 MSYS 风格路径。
⚠️ 为什么 pump 要等够 900ms：面板高度是靠 QTimer 延迟算出来的
    （`_fit_ti_split` 自己会重试 5 次 × 80ms）。不等够，量到的是 0 或上一页的旧值，
    于是"看起来布局崩了"其实是自己量早了 —— 这是这个脚本最容易自欺的地方。

输出：`_ui_render/<tab>.png`（右栏）+ `_ui_render/<tab>_full.png`（整窗），
     并在 stdout 打印各栈的 实际高度 vs sizeHint（判断"留白爆炸"的量化依据）。
     `*.png` 已在 .gitignore 里，不会污染仓库。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_ui_render"

# ⚠️ 平台用 offscreen，但**必须手动挂字体**（见下面 _load_system_fonts）：
#    offscreen 完全没有字体库 —— 不挂字体时抓出来是**空白色块、一个字都没有**，
#    极容易被误判成"布局崩了/控件没渲染"（气泡与本脚本上各踩过一次）。
#    为什么不用 windows 平台：它会真的把窗口映射到桌面，还会把 QMessageBox 之类的
#    模态对话框弹出来**阻塞住**（实测直接卡死被 SIGTERM）—— 既打扰小千又不稳定。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ---- 1) 先 torch、再碰 sys.path（严格照 main.py 的顺序）----
# ⚠️⚠️ 实测（2026-10-13）：**绝不能**在这两行之前 `import PyQt5`。
#    只要 PyQt5 先把 Qt 的 DLL 加载进来，随后的 `import torch` 就必然失败：
#        OSError: [WinError 1114] c10.dll 初始化例程失败
#    同一个解释器、同一个 PATH，**只调换这两行的先后**就能稳定复现/消除；
#    而 main.py 正是"先 torch 再 PyQt5"，所以它没这个毛病。
#    下面取插件路径改用 find_spec —— 它只查 spec、**不执行** PyQt5 的 __init__，
#    不会把 Qt DLL 拉进来，于是既拿到路径又不踩雷。
os.chdir(ROOT)
import torch  # noqa: F401,E402

sys.path.insert(0, str(ROOT))

# ---- 2) 平台插件路径：只要在 QApplication 创建之前设好即可 ----
_spec = importlib.util.find_spec("PyQt5")
if _spec is not None and _spec.origin:
    os.environ.setdefault(
        "QT_PLUGIN_PATH", str(Path(_spec.origin).parent / "Qt5" / "plugins")
    )

from PyQt5.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

#: 右栏页签名 —— **以 MainWindow._right_tab_index 为准**：
#:     {"params": 0, "annot": 1, "model": 2}
#:  ⚠️ 旧名字（annot/ver/train/det/hm/aug 那 6 个）已经废弃：右栏参照友商 友商平台
#:     重构成模块页了，训练/推理参数挪进了 params 页。
#:     2026-10-14 又砍掉 "image"（右栏「图像管理」与左栏新页重名且内容重复，整页删除）。
#:     用旧名字调 show_right_tab() 会**静默 return 不报错**，于是你会对着 annot 页
#:     的截图分析 train 的布局 —— 这坑真踩过（2026-10-13）。
ALL_TABS = ["params", "annot", "model"]


def pump(ms: int) -> None:
    """空转事件循环 ms 毫秒（不进主循环）。"""
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec_()


def _h(w) -> str:
    if w is None:
        return "-"
    try:
        return f"{w.height()}/{w.sizeHint().height()}"
    except RuntimeError:
        # wrapped C/C++ object deleted —— 正好暴露"延迟回调打到已销毁对象"这类隐患
        return "已销毁"


#: offscreen 平台字体库为空，得把系统字体文件手动塞进去，否则一个字都画不出来。
_FONT_FILES = (
    r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑（界面正文）
    r"C:\Windows\Fonts\msyhbd.ttc",    # 微软雅黑 Bold
    r"C:\Windows\Fonts\simhei.ttf",    # 黑体（兜底）
    r"C:\Windows\Fonts\segoeui.ttf",   # 西文兜底
)


def _load_system_fonts() -> int:
    """给 offscreen 平台挂上系统字体，返回成功加载的个数（必须 >0，否则截图是空白块）。"""
    from PyQt5.QtGui import QFontDatabase

    n = 0
    for fp in _FONT_FILES:
        if os.path.exists(fp) and QFontDatabase.addApplicationFont(fp) != -1:
            n += 1
    return n


def main(argv: list[str]) -> int:
    keys = [k for k in argv if not k.startswith("-")] or ALL_TABS
    bad = [k for k in keys if k not in ALL_TABS]
    if bad:
        print(f"未知页签 {bad}；可选：{', '.join(ALL_TABS)}")
        return 2

    OUT.mkdir(exist_ok=True)

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])  # noqa: F841

    nf = _load_system_fonts()
    print(f"系统字体挂载 {nf} 个" + ("" if nf else "  ⚠️ 一个都没挂上 → 截图会是空白块！"))

    from PyQt5.QtCore import QT_VERSION_STR
    print(f"Qt {QT_VERSION_STR} / QPA={os.environ['QT_QPA_PLATFORM']} / torch {torch.__version__}")

    from ui.main_window import MainWindow

    win = MainWindow()
    win.resize(1680, 1000)
    win.show()
    pump(1200)  # 构造完再等一手，让启动期的一次性布局跑完

    scr = app.primaryScreen().size()
    print(f"虚拟屏幕 {scr.width()}x{scr.height()} → 窗口实到 {win.width()}x{win.height()}"
          f"（sizeHint {win.sizeHint().width()}x{win.sizeHint().height()}，"
          f"minimumSize {win.minimumWidth()}x{win.minimumHeight()}）")
    if win.width() != 1680:
        print("  ⚠️ resize 没生效 —— 别拿这个尺寸的截图判断排版（右栏会被挤成细条）")

    stack = getattr(win, "_right_stack", None)
    if stack is None:
        print("!! 找不到 _right_stack，主窗口结构变了？")
        return 3
    idx_map = getattr(win, "_right_tab_index", {})

    print(f"{'页签':<8}{'右栏栈 实际/sizeHint':<24}{'当前页 实际/sizeHint':<24}"
          f"{'训练栈 实际/sizeHint':<24}{'推理栈 实际/sizeHint':<24}")
    for key in keys:
        win.show_right_tab(key)
        pump(900)  # ← _fit_ti_split 最多重试 400ms，这里留一倍余量

        # ⚠️ show_right_tab() 查不到 key 时是**静默 return** 的（不抛错、不报警）。
        #    不校验的话，会拿着上一个页签的截图去分析这个页签的布局 —— 已踩过一次。
        expect = idx_map.get(key)
        if stack.currentIndex() != expect:
            print(f"!! {key}: 没有切过去（currentIndex={stack.currentIndex()}, 期望={expect}）"
                  f"；该 key 不在 _right_tab_index={idx_map} 里？")
            continue

        cur = stack.currentWidget()
        print(f"{key:<9}{_h(stack):<26}{_h(cur):<26}"
              f"{_h(getattr(win, '_train_stack', None)):<26}"
              f"{_h(getattr(win, '_infer_stack', None)):<26}")

        stack.grab().save(str(OUT / f"{key}.png"))
        win.grab().save(str(OUT / f"{key}_full.png"))

    win.close()
    pump(100)
    print(f"\n已输出到 {OUT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

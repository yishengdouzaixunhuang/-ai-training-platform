"""验证 `_fit_ti_split` 在 QSplitter 的 C++ 对象被销毁后**不再杀进程**。

背景（crash_log.txt 2026-10-13 两次实测崩溃）：
  `_fit_ti_split` 是 `QTimer.singleShot` 的回调。PyQt5 里槽函数抛异常会**直接终止进程**。
  而 `sp.count()` 原先写在 try **外面** —— QSplitter 被销毁后 Python 这层 wrapper 还在
  （`sp is None` 为 False），一碰就 `RuntimeError: wrapped C/C++ object ... has been deleted`。

为什么不用真 MainWindow：这里要验的是"**方法本身**对已销毁控件的行为"，
  与方法绑在哪个实例上无关。用未初始化的实例调未绑定方法即可，
  顺便避开"MainWindow 构造后进程退不出去"这个本机坑。

能红也能绿：同一个"坏 self"分别喂给**旧形态**和**现网代码**，
  旧形态必须抛、现网代码必须不抛。断言如果两边都过，等于没测。
"""
import os
import sys

import torch  # noqa: F401  —— 先导 torch，PyQt5 在前会撞 c10.dll（本机铁律）

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # 不需要字体，纯逻辑验证
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QSplitter, QWidget  # noqa: E402
from PyQt5 import sip  # noqa: E402

RESULTS = []


def ck(name, cond, detail=""):
    RESULTS.append((bool(cond), name))
    print("  %-4s %s%s" % ("PASS" if cond else "FAIL", name,
                           ("  << %s" % detail) if (not cond and detail) else ""), flush=True)


def old_form(self):
    """修复前的形态：`count()` 在 try 外面（crash_log 里崩的就是这一行）。"""
    sp = getattr(self, "_ti_split", None)
    if sp is None or sp.count() != 2:      # ← 裸调用，一碰就炸
        return
    avail = sp.height()
    return avail


def main():
    app = QApplication(sys.argv)  # noqa: F841
    from ui.main_window import MainWindow

    # 造一个未初始化的实例：只喂 _ti_split / _ti_fit_tries，不跑 __init__
    w = MainWindow.__new__(MainWindow)

    # ── A. 先证明"坏控件"确实能造出来 ──
    sp = QSplitter()
    sp.addWidget(QWidget())
    sp.addWidget(QWidget())
    ck("A1 活着的 QSplitter.count() == 2", sp.count() == 2, sp.count())
    sip.delete(sp)
    w._ti_split = sp                       # 引用的 C++ 对象已死
    ck("A2 sip.delete 后 sp is None 为 False（wrapper 还活着）", sp is not None)
    raised = None
    try:
        sp.count()
    except RuntimeError as e:
        raised = e
    ck("A3 复现成立：已销毁的 splitter 一碰就 RuntimeError",
       raised is not None and "has been deleted" in str(raised), raised)

    # ── B. 旧形态必须抛（证明断言"能红"）──
    old_raised = None
    try:
        old_form(w)
    except Exception as e:
        old_raised = e
    ck("B1 旧形态在同一条件下确实抛异常（所以这条断言不是空跑）",
       isinstance(old_raised, RuntimeError) and "has been deleted" in str(old_raised),
       old_raised)

    # ── C. 现网代码必须不抛（证明"变绿"）──
    w._ti_fit_tries = 0
    now_raised = None
    try:
        MainWindow._fit_ti_split(w)        # 直接调方法，绕开 __init__
    except Exception as e:
        now_raised = e
    ck("C1 现网 _fit_ti_split() 对已销毁的 splitter 不抛（槽函数异常不会再带走进程）",
       now_raised is None, now_raised)

    # ── D. 正常路径没被 try 吞成静默失败 ──
    # ⚠️ 夹具必须**带 layout**：产品里两段是 `_make_ti_section()` 造的、一定挂 QVBoxLayout；
    #    第一版我直接 addWidget(QWidget())，`box.layout()` 返回 None，
    #    `_ti_section_need` 就崩在 `lay.count()` —— 那是夹具不真实，不是产品的问题。
    from PyQt5.QtWidgets import QVBoxLayout

    w2 = MainWindow.__new__(MainWindow)
    live = QSplitter()
    for _ in range(2):
        box = QWidget()
        QVBoxLayout(box)
        live.addWidget(box)
    live.resize(600, 400)
    w2._ti_split = live
    w2._ti_fit_tries = 0
    got = None
    try:
        got = MainWindow._ti_section_need(w2, 0)
    except Exception as e:
        got = "异常 %s: %s" % (type(e).__name__, e)
    ck("D1 正常 splitter 下 _ti_section_need(0) 仍返回数值（≥90）",
       isinstance(got, int) and got >= 90, got)
    need_raised = None
    try:
        MainWindow._fit_ti_split(w2)       # 活着时应能走到 setSizes
    except Exception as e:
        need_raised = e
    ck("D2 正常 splitter 下 _fit_ti_split() 正常执行", need_raised is None, need_raised)

    n_pass = sum(1 for ok_, _ in RESULTS if ok_)
    print()
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (n_pass, len(RESULTS) - n_pass, len(RESULTS)), flush=True)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())

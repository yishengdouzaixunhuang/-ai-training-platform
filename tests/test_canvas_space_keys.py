# -*- coding: utf-8 -*-
"""画布的 Space / Ctrl+Space：显示开关、快捷键入口一致性、全局按键不抢输入框。

## 这一版修的四件事（都先复现再修）

① 空格只在画布有焦点时有效：分类模式有全局兜底，检测模式没有
   ⇒ 点完图片列表按空格"没反应"。
② 检测模式 Ctrl+Space 关掉了框，画布状态行仍显示 `Annot:ON`
   —— `status_info_text()` 读 `_show_annotation`，检测模式画框看的是
   `_show_annotation_boxes`（分割模式两者才是同一个标志）。
③ 键盘入口和按钮入口走的不是同一条路：`keyPressEvent` 里是内联副本，
   末尾只有 `update()`，**没有 `view_changed`** ⇒ 多视图参考窗不跟着刷
   （就是"窗口 2/3 慢一拍"那个老毛病）。
④ 按住空格/ Ctrl+Space 会连翻 —— 两个分支都没判 `isAutoRepeat()`。

## ⚠️ 全局兜底必须带焦点守卫

`QApplication.installEventFilter(self)` 等于接管**全应用**的按键。老版本只判
`_cls_mode`、不看焦点 ⇒ 分类工程开着时，新建工程对话框里 `'NG part'` 会变成
`'NGpart'`，还顺手切了背后的热力图。这是本文件第 5、6 条用例拦的东西，
**别为了"让空格更好用"把这层守卫去掉**。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_canvas_space_keys.py
"""
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ⚠️ 与 main.py 一致：torch 必须先于 PyQt5（否则 WinError 1114 c10.dll）
try:
    import torch  # noqa: F401
except OSError as _e:  # pragma: no cover
    print("[warn] torch 加载失败：%s" % _e)

from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLineEdit, QWidget  # noqa: E402

from annotation.mask_editor import AnnotationCanvas  # noqa: E402

failures = []
_APP = None
_WIN = None


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        failures.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc()


def app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def canvas(det=True):
    """一块干净的画布；det=True 时假装是检测模式（键位与状态行都按它分流）。"""
    app()
    c = AnnotationCanvas()
    c._det_overlay = object() if det else None
    return c


def key(k, mods=Qt.NoModifier, autorepeat=False):
    ev = QKeyEvent(QEvent.KeyPress, k, mods, "")
    ev.isAutoRepeat = lambda: autorepeat          # Qt 没有公开的 setter
    return ev


def window():
    """真主窗（eventFilter 装在 QApplication 上，只有真窗口才测得到）。"""
    global _WIN
    if _WIN is not None:
        return _WIN
    import ui.task_queue as tq

    fd, p = tempfile.mkstemp(prefix="aitp_space_", suffix=".json")
    os.close(fd)
    os.unlink(p)
    tq.DEFAULT_HISTORY_PATH = p                   # 别写真实的 .task_history.json

    from ui.main_window import MainWindow
    from ui.qt_thread import init_ui_invoker

    init_ui_invoker()
    _WIN = MainWindow()
    _WIN.show()
    app().processEvents()
    return _WIN


class _Recorder(QWidget):
    """接住 KeyPress 的哑控件：用它判断"这一下到底有没有被全局过滤器吃掉"。"""

    def __init__(self):
        super().__init__()
        self.got = []

    def keyPressEvent(self, ev):
        self.got.append(ev.key())
        super().keyPressEvent(ev)


def give_focus(widget):
    """把应用焦点明确挪到这个控件上。

    ⚠️ 只 `setFocus()` 不够：上一条用例删掉自己的控件后，焦点会落回主窗的搜索框
       （QLineEdit），于是"输入框守卫"替别的守卫挡枪，用例**红得不对**。
       这里先 clearFocus 再 setFocus，并断言真的挪到位。
    """
    fw = QApplication.focusWidget()
    if fw is not None and fw is not widget:
        fw.clearFocus()
    widget.show()
    widget.setFocus()
    app().processEvents()
    assert QApplication.focusWidget() is widget, \
        "测试环境没把焦点挪到 %r（当前 %r）" % (widget, QApplication.focusWidget())


# ------------------------------------------------------------------ 画布自身

def test_autorepeat_space_toggles_once():
    """按住空格：系统连发 KeyPress，只能翻一次。"""
    c = canvas()
    before = c._show_pred_boxes
    c.keyPressEvent(key(Qt.Key_Space))                       # 第一下：真按键
    mid = c._show_pred_boxes
    assert mid != before, "第一下就没翻，用例前提不成立"
    for _ in range(5):
        c.keyPressEvent(key(Qt.Key_Space, autorepeat=True))  # 按住不放
    assert c._show_pred_boxes == mid, \
        "★ 按住空格连翻了（%s -> %s）" % (mid, c._show_pred_boxes)


def test_autorepeat_ctrl_space_toggles_once():
    """Ctrl+Space 同理 —— 上一轮只报了空格，其实两个分支都漏。"""
    c = canvas()
    before = c._show_annotation_boxes
    c.keyPressEvent(key(Qt.Key_Space, Qt.ControlModifier))
    mid = c._show_annotation_boxes
    assert mid != before, "第一下就没翻，用例前提不成立"
    for _ in range(3):
        c.keyPressEvent(key(Qt.Key_Space, Qt.ControlModifier, autorepeat=True))
    assert c._show_annotation_boxes == mid, \
        "★ 按住 Ctrl+Space 连翻了（%s -> %s）" % (mid, c._show_annotation_boxes)


def test_keyboard_and_method_emit_same_view_changed():
    """键盘入口必须和方法入口一样发 `view_changed`（多视图参考窗靠它刷新）。"""
    c = canvas()
    got = []
    c.view_changed.connect(lambda: got.append(1))

    c.keyPressEvent(key(Qt.Key_Space))
    n_key = len(got)
    got.clear()
    c.toggle_prediction_visibility()
    n_method = len(got)

    assert n_key == n_method == 1, \
        "★ 两个入口不一致：键盘 %d 次 / 方法 %d 次（键盘少发＝参考窗不刷新）" % (n_key, n_method)


def test_det_status_line_follows_boxes_flag():
    """② 检测模式下状态行要跟 `_show_annotation_boxes` 走。"""
    c = canvas()
    assert "Annot:ON" in c.status_info_text(), c.status_info_text()

    c.keyPressEvent(key(Qt.Key_Space, Qt.ControlModifier))
    assert c._show_annotation_boxes is False, "前提：框应该被关掉了"
    s = c.status_info_text()
    assert "Annot:OFF" in s, "★ 框已关掉，状态行还显示 Annot:ON —— 假显示：%s" % s

    c.keyPressEvent(key(Qt.Key_Space, Qt.ControlModifier))
    assert "Annot:ON" in c.status_info_text(), c.status_info_text()


def test_seg_status_line_still_uses_seg_flag():
    """分割模式不能被上面的修复带偏：它读的仍是 `_show_annotation`。"""
    c = canvas(det=False)
    c._show_annotation = False
    c._show_annotation_boxes = True          # 故意相反，验证读的是分割那个
    assert "Annot:OFF" in c.status_info_text(), c.status_info_text()


# ------------------------------------------------------------------ 全局兜底

def test_text_input_keeps_space_in_detection():
    """★ 最要紧的一条：检测模式下在输入框里打 'NG part'，空格不能丢。"""
    w = window()
    w._detection_mode = True
    w._cls_mode = False
    edit = QLineEdit()
    give_focus(edit)
    QTest.keyClicks(edit, "NG part")
    app().processEvents()
    assert edit.text() == "NG part", "★ 空格被全局过滤器吃了：%r" % edit.text()
    edit.deleteLater()


def test_text_input_keeps_space_in_cls():
    """分类模式（老 bug 所在）同样要能打空格。"""
    w = window()
    w._cls_mode = True
    w._detection_mode = False
    edit = QLineEdit()
    give_focus(edit)
    QTest.keyClicks(edit, "NG part")
    app().processEvents()
    assert edit.text() == "NG part", "★ 分类模式下空格被吃了：%r" % edit.text()
    edit.deleteLater()


def test_global_space_works_when_focus_elsewhere():
    """① 的收益：焦点不在画布上（比如刚点完图片列表），空格照样切预测框。"""
    w = window()
    w._detection_mode = True
    w._cls_mode = False
    w.canvas._det_overlay = object()
    rec = _Recorder()
    give_focus(rec)

    before = w.canvas._show_pred_boxes
    QTest.keyClick(rec, Qt.Key_Space)
    app().processEvents()
    assert w.canvas._show_pred_boxes != before, "★ 焦点不在画布时空格无效（① 没修好）"
    assert Qt.Key_Space not in rec.got, "空格不该同时漏给下面的控件"
    rec.deleteLater()


def test_cls_ctrl_space_does_not_toggle_heatmap():
    """⑥ 的另一半：分类模式下 Ctrl+Space 不许被当成"切热力图"吃掉。"""
    w = window()
    w._cls_mode = True
    w._detection_mode = False
    w.canvas._show_heatmap = True
    rec = _Recorder()
    give_focus(rec)

    QTest.keyClick(rec, Qt.Key_Space, Qt.ControlModifier)
    app().processEvents()
    assert w.canvas._show_heatmap is True, "★ Ctrl+Space 被分类分支吃了（切了热力图）"
    # 不只是"没切热力图"，是**整个放行**给下层控件
    assert Qt.Key_Space in rec.got, "★ Ctrl+Space 仍被全局过滤器吃掉了"
    rec.deleteLater()


def test_global_ctrl_space_works_when_focus_elsewhere():
    """检测模式：焦点不在画布时 Ctrl+Space 也能切标注框（与画布内同一个方法）。"""
    w = window()
    w._detection_mode = True
    w._cls_mode = False
    w.canvas._det_overlay = object()
    rec = _Recorder()
    give_focus(rec)

    before = w.canvas._show_annotation_boxes
    QTest.keyClick(rec, Qt.Key_Space, Qt.ControlModifier)
    app().processEvents()
    assert w.canvas._show_annotation_boxes != before, "★ 焦点不在画布时 Ctrl+Space 无效"
    assert Qt.Key_Space not in rec.got, "Ctrl+Space 不该同时漏给下面的控件"
    rec.deleteLater()


def test_modal_dialog_keeps_space():
    """模态对话框里（新建工程就是）空格一律不抢 —— 哪怕焦点不在输入框上。

    ⚠️ 先把焦点放到一个**非输入控件**上并断言这一点。否则焦点残留在上一条用例的
       输入框里时，"输入框守卫"会替"模态守卫"挡枪：把模态那段删掉测试照样绿
       （变异测试实拍，第一版就这么漏的）。
    """
    from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout

    w = window()
    w._detection_mode = True
    rec = _Recorder()
    give_focus(rec)

    dlg = QDialog()
    dlg.setModal(True)
    lay = QVBoxLayout(dlg)
    # ⚠️ 先放一个**按钮**：模态框一显示，Qt 就把焦点给第一个可聚焦子控件。
    #    如果第一个是输入框，"输入框守卫"会替"模态守卫"挡枪 —— 把模态那段删掉
    #    测试照样绿（变异实拍）。所以先让默认焦点落在按钮上。
    btn = QPushButton("OK")
    lay.addWidget(btn)
    edit = QLineEdit()
    lay.addWidget(edit)
    dlg.show()
    app().processEvents()
    assert QApplication.activeModalWidget() is dlg, "前提：模态框没生效"
    btn.setFocus()
    app().processEvents()

    # 前提：焦点不在需要空格的控件上 ⇒ 此刻只有"模态守卫"能返回 False
    assert w._focus_is_text_input() is False, \
        "前提不成立：焦点还在输入框上（%r），这条用例会测到别的守卫" % QApplication.focusWidget()
    assert w._space_should_be_ours(key(Qt.Key_Space)) is False, \
        "★ 模态对话框开着还去抢空格（新建工程里会打不出空格）"

    give_focus(edit)
    QTest.keyClicks(edit, "NG part")
    app().processEvents()
    assert edit.text() == "NG part", "★ 模态框里空格被吃了：%r" % edit.text()
    dlg.close()
    dlg.deleteLater()
    rec.deleteLater()


def test_seg_mode_space_not_grabbed_globally():
    """分割模式保持原语义：空格只在画布内响应，全局不抢。"""
    w = window()
    w._cls_mode = False
    w._detection_mode = False
    rec = _Recorder()
    give_focus(rec)

    QTest.keyClick(rec, Qt.Key_Space)
    app().processEvents()
    assert Qt.Key_Space in rec.got, "★ 分割模式下空格被全局抢走了（超出本次范围）"
    rec.deleteLater()


def main():
    checks = [
        ("★ 按住空格只翻一次", test_autorepeat_space_toggles_once),
        ("★ 按住 Ctrl+Space 只翻一次", test_autorepeat_ctrl_space_toggles_once),
        ("★ 键盘/按钮两个入口 view_changed 一致", test_keyboard_and_method_emit_same_view_changed),
        ("★ 检测模式状态行跟标注框标志走", test_det_status_line_follows_boxes_flag),
        ("分割模式状态行仍读分割标志", test_seg_status_line_still_uses_seg_flag),
        ("★ 检测模式输入框能打 'NG part'", test_text_input_keeps_space_in_detection),
        ("★ 分类模式输入框能打 'NG part'", test_text_input_keeps_space_in_cls),
        ("★ 焦点不在画布时空格仍生效", test_global_space_works_when_focus_elsewhere),
        ("★ 焦点不在画布时 Ctrl+Space 也生效", test_global_ctrl_space_works_when_focus_elsewhere),
        ("★ 模态对话框里空格不抢", test_modal_dialog_keeps_space),
        ("★ 分类模式 Ctrl+Space 不切热力图", test_cls_ctrl_space_does_not_toggle_heatmap),
        ("分割模式空格仍不全局抢", test_seg_mode_space_not_grabbed_globally),
    ]
    for name, fn in checks:
        check(name, fn)
    time.sleep(0.05)

    print()
    if failures:
        print("❌ 有失败：%d / %d" % (len(failures), len(checks)))
        for name, exc in failures:
            print("   - %s: %s" % (name, exc))
        return 1
    print("✅ 全部通过  %d/%d 项通过" % (len(checks), len(checks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

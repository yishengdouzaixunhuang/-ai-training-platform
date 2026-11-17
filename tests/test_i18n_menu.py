# -*- coding: utf-8 -*-
"""P0 集成：英文模式下 Settings → Language 菜单必须已翻译（不是硬编码双语）。

运行：C:\\aitp_venv\\Scripts\\python.exe tests\\test_i18n_menu.py
"""
import _testenv  # noqa: F401  无显示器 + 配置目录隔离
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: F401  必须先于 PyQt5（WinError 1114）


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from core import i18n
    from core import ui_prefs
    # 这条用例不许读仓库根真实偏好：必须走 tests/_testenv.py 的隔离目录。
    # 否则用户在应用里点了 English，门禁第 2 步就会红（实测已发生）。
    cfg_dir = os.environ.get("AITP_TEST_CFG_DIR")
    assert cfg_dir, "AITP_TEST_CFG_DIR 未设置（_testenv 没生效）"
    assert os.path.abspath(ui_prefs.default_path()) == os.path.abspath(
        os.path.join(cfg_dir, ".ui_settings.json")), \
        "ui_prefs 没认 AITP_TEST_CFG_DIR：%r" % ui_prefs.default_path()
    # 用例自己写死临时配置，不依赖仓库根、也不依赖用户当前选了什么
    ui_prefs.save({"lang": "zh"})
    i18n.reset_for_tests()
    assert i18n.load_dictionary() >= 1, "resources/i18n/en.json 没读到"
    i18n.set_language("en")

    from ui.qt_thread import init_ui_invoker
    init_ui_invoker()
    from ui.main_window import MainWindow

    w = MainWindow()
    try:
        settings = [a for a in w.menuBar().actions() if a.text() == "Settings(&S)"]
        assert settings, "找不到 Settings 菜单"
        subs = [a.text() for a in settings[0].menu().actions()]
        assert subs[0] == "Workspace...", "Workspace 之后才是 Language，实际 %r" % subs
        assert subs[1] == "Language", "语言菜单没翻译：%r" % subs
        lang = settings[0].menu().actions()[1].menu()
        assert [a.text() for a in lang.actions()] == ["Chinese", "English"], \
            "语言子项不对：%r" % [a.text() for a in lang.actions()]
        assert lang.actions()[0].isChecked(), "磁盘 lang=zh，应勾中文"
    finally:
        w.close()
        app.processEvents()

    print("✅ 全部通过  1/1 通过（Settings → Language 已走 tr）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

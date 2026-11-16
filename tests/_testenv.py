# -*- coding: utf-8 -*-
"""测试公共环境：无显示器 + **配置目录隔离**。

## 为什么必须有这个文件

有几次跑完测试 `git status` 里冒出 `M .train_settings.json` ——
`MainWindow.closeEvent()` → `_save_panel_train_settings()` 把**仓库根**的
真实配置改成了测试窗口的面板状态（`flip: true → false`）。

前两次是给个别测试 stub `_train_settings_path`，第三次又漏了。
靠"每个新测试都记得 stub"不可靠，所以：

1. `core.settings.cfg_path()` / `Scope.default_path()` 认环境变量
   `AITP_TEST_CFG_DIR`，设了就不写仓库根；
2. 这里把它指到一个 **临时目录**；
3. `tests/test_test_isolation.py` 静态检查"建 MainWindow 的测试都 import 了本模块"。

## 用法

在**创建 QApplication / MainWindow 之前**：

    import _testenv  # noqa: F401

（脚本方式 `python tests/xxx.py` 时 `tests/` 就在 `sys.path[0]`，直接 import 即可。）
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("AITP_TEST_CFG_DIR", tempfile.mkdtemp(prefix="aitp_testcfg_"))

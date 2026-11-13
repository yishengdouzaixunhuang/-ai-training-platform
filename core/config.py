"""全局配置"""
import os
import json
import tempfile
from pathlib import Path

APP_NAME = "AI训练平台"
APP_VERSION = "0.1.0"

# 默认路径
DEFAULT_WORKSPACE = str(Path.home() / "Documents" / "AI_Training_Platform")
CONFIG_FILE = str(Path.home() / ".ai_training_config.json")

#: 读不到配置时用的兜底值。**必须包含所有"调用方直接下标取"的键** ——
#: `main_window.__init__` 第一行就是 `self.config["workspace"]`，缺了就是 KeyError。
DEFAULT_CONFIG = {"workspace": DEFAULT_WORKSPACE, "recent_projects": []}

# 类别颜色表（20种预定义颜色）
CLASS_COLORS = [
    (128, 0, 0),   (0, 128, 0),   (128, 128, 0), (0, 0, 128),
    (128, 0, 128), (0, 128, 128), (128, 128, 128),(64, 0, 0),
    (192, 0, 0),   (64, 128, 0),  (192, 128, 0), (64, 0, 128),
    (192, 0, 128), (64, 128, 128),(192, 128, 128),(0, 64, 0),
    (128, 64, 0),  (0, 192, 0),   (128, 192, 0), (0, 64, 128),
]

# Special ignore class
IGNORE_CLASS_ID = 255
IGNORE_CLASS_COLOR = (128, 128, 128)  # Gray for ignore regions

def load_config():
    """读全局配置。**读不出来就给默认值，绝不抛** —— 这是平台启动的第一步。

    ★ 2026-11-13：改前是裸 `json.load`，没有 try。而 `MainWindow.__init__` 第 176 行
      是 `self.config = load_config()`，外面也没有兜 —— 文件半截（`save_config`
      写到一半断电就会这样）时抛 JSONDecodeError ⇒ **整个平台起不来**，
      用户看到的只是"双击图标没反应"。
      实测：配置写成 `{"workspace": "C:/x", ` → `MainWindow()` 直接抛。

    现在的处理：
    - 解析失败 / 不是对象 / 读不了 → 把坏文件改名成 `*.bad` 留档（不静默删掉），
      返回默认配置；
    - 能读 → 与 `DEFAULT_CONFIG` **合并**（老文件缺 `workspace` 也不会 KeyError），
      未知键原样保留。
    """
    if not os.path.exists(CONFIG_FILE):
        return _copy_defaults()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("配置内容不是对象：%s" % type(data).__name__)
    except (OSError, ValueError):
        try:
            os.replace(CONFIG_FILE, CONFIG_FILE + ".bad")   # 留档，别静默丢
        except OSError:
            pass
        return _copy_defaults()

    merged = _copy_defaults()
    merged.update(data)
    if not isinstance(merged.get("recent_projects"), list):
        merged["recent_projects"] = []
    return merged


def _copy_defaults():
    return {k: (list(v) if isinstance(v, list) else v)
            for k, v in DEFAULT_CONFIG.items()}


def save_config(config):
    """原子写全局配置（临时文件 + `os.replace`）。

    ★ 2026-11-13：改前是 `open(CONFIG_FILE, "w")` 直接覆盖 —— 写到一半断电/崩溃
      就留下半个 JSON，而 `load_config` 读到半个 JSON 会抛（见上）⇒
      **保存时崩一次，平台就再也起不来**。这条链子两头都得堵：
      写要原子，读要容错。
    """
    if not isinstance(config, dict):
        raise TypeError("config 必须是 dict，实际 %s" % type(config).__name__)

    d = os.path.dirname(os.path.abspath(CONFIG_FILE))
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(prefix=".ai_training_config.", suffix=".tmp", dir=d)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())      # 真落盘再替换，别只信页缓存
            except OSError:
                pass
        os.replace(tmp, CONFIG_FILE)
    except OSError:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise

"""全局配置"""
import os
import json
from pathlib import Path

APP_NAME = "AI训练平台"
APP_VERSION = "0.1.0"

# 默认路径
DEFAULT_WORKSPACE = str(Path.home() / "Documents" / "AI_Training_Platform")
CONFIG_FILE = str(Path.home() / ".ai_training_config.json")

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
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"workspace": DEFAULT_WORKSPACE, "recent_projects": []}

def save_config(config):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

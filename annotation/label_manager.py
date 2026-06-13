"""类别/标签管理"""
import json
import os
from core.config import CLASS_COLORS

class LabelManager:
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.meta_path = os.path.join(project_dir, "project.json")
        self.classes = []
        self.load()
    
    def load(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.classes = meta.get("classes", ["background"])
        # 确保 background 在第一位
        if "background" not in self.classes:
            self.classes.insert(0, "background")
    
    def save(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["classes"] = self.classes
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    def add_class(self, name):
        if name in self.classes:
            return False
        self.classes.append(name)
        self.save()
        return True
    
    def remove_class(self, name):
        if name == "background":
            return False
        if name in self.classes:
            self.classes.remove(name)
            self.save()
            return True
        return False
    
    def get_color(self, class_id):
        return CLASS_COLORS[class_id % len(CLASS_COLORS)]
    
    def class_to_id(self, name):
        try:
            return self.classes.index(name)
        except ValueError:
            return 0

# -*- coding: utf-8 -*-
"""工程管理"""
import os
import json
from datetime import datetime
from pathlib import Path

class ProjectManager:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    MIXED_CLS_LABEL = "混合分类"
    VALID_TASK_TYPES = ("语义分割", "目标检测", "图像分类", "OCR文字识别", "OCV字符质检")
    VALID_TASK_TYPES_FULL = ("语义分割", "目标检测", "图像分类", "混合分类", "OCR文字识别", "OCV字符质检")

    def create_project(self, name, task_type="语义分割"):
        allowed = self.VALID_TASK_TYPES_FULL
        if task_type not in allowed:
            raise ValueError(f"不支持的任务类型: {task_type}，可选: {allowed}")
        project_dir = self.workspace / name
        if project_dir.exists():
            raise FileExistsError(f"工程 '{name}' 已存在")
        project_dir.mkdir(parents=True)
        (project_dir / "images").mkdir()
        (project_dir / "annotations").mkdir()
        (project_dir / "models").mkdir()
        (project_dir / "outputs").mkdir()
        
        meta = {
            "name": name,
            "task_type": task_type,
            "created": datetime.now().isoformat(),
            "classes": [],
        }
        with open(project_dir / "project.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta
    
    def open_project(self, name):
        path = self.workspace / name / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"工程 '{name}' 不存在")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_project_meta(self, name, meta):
        path = self.workspace / name / "project.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    def list_projects(self):
        projects = []
        for d in self.workspace.iterdir():
            if d.is_dir() and (d / "project.json").exists():
                stat = (d / "project.json").stat()
                projects.append({
                    "name": d.name,
                    "modified": stat.st_mtime,
                })
        projects.sort(key=lambda x: x["modified"], reverse=True)
        return projects
    
    def get_project_dir(self, name):
        return self.workspace / name
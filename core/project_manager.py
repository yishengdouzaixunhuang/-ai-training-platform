# -*- coding: utf-8 -*-
"""????"""
import os
import json
import tempfile
from datetime import datetime
from pathlib import Path

class ProjectManager:
    def __init__(self, workspace):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._custom_dirs = self._load_custom_dirs()
    
    MIXED_CLS_LABEL = "????"
    VALID_TASK_TYPES = ("语义分割", "目标检测", "图像分类", "OCR文字识别", "OCV字符质检")
    VALID_TASK_TYPES_FULL = ("语义分割", "目标检测", "图像分类", "混合分类", "OCR文字识别", "OCV字符质检")

    def _custom_registry_path(self):
        return self.workspace / ".custom_projects.json"

    def _load_custom_dirs(self):
        p = self._custom_registry_path()
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_custom_dirs(self):
        """原子写外部工程注册表（`.custom_projects.json`）。

        ★ 2026-11-13：改前是 `open(w)` 直接覆盖。这份文件登记的是**所有外部工程**
          的路径，写坏之后 `_load_custom_dirs()` 静默返回 {} —— 用户在「工程管理」里
          看到的是"我的一堆外部工程全没了"（文件还在磁盘上，但平台不认）。
          改成临时文件 + os.replace，写一半崩了也只是这次没写成。
        """
        path = self._custom_registry_path()
        tmp = None
        try:
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=".custom_projects.", suffix=".tmp", dir=d)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._custom_dirs, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
        except OSError:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _resolve_project_dir(self, name):
        """Return the actual project directory, checking custom dirs first."""
        if name in self._custom_dirs:
            return Path(self._custom_dirs[name])
        return self.workspace / name

    def create_project(self, name, task_type="语义分割", parent_dir=None,
                       views=None, view_names=None):
        """创建工程。

        views/view_names：多视图（如客户三视图 漫反射/形状图1/形状图2）声明。
        声明后会写进 project.json，训练端据此决定输入通道数：
            单张图像 → 3 通道；三视图 → 9 通道。
        不声明则按单张图像处理（与既有行为一致）。
        """
        allowed = self.VALID_TASK_TYPES_FULL
        if task_type not in allowed:
            raise ValueError(f"不支持的任务类型: {task_type}；可选: {allowed}")
        if parent_dir:
            parent = Path(parent_dir)
            project_dir = parent / name
        else:
            project_dir = self.workspace / name
        if project_dir.exists():
            raise FileExistsError(f"工程 '{name}' 已存在")
        project_dir.mkdir(parents=True)
        (project_dir / "images").mkdir()
        (project_dir / "annotations").mkdir()
        (project_dir / "models").mkdir()
        (project_dir / "outputs").mkdir()
        if parent_dir:
            self._custom_dirs[name] = str(project_dir.resolve())
            self._save_custom_dirs()
        
        meta = {
            "name": name,
            "task_type": task_type,
            "created": datetime.now().isoformat(),
            "classes": [],
        }
        # 多视图声明：写进 project.json 后，训练端（is_multiview_project）会自动识别，
        # 无需再靠扫描文件名去猜。
        if views:
            vlist = [str(v) for v in views]
            meta["views"] = vlist
            meta["view_count"] = len(vlist)
            meta["in_channels"] = 3 * len(vlist)
            if view_names:
                meta["view_names"] = {str(k): str(v) for k, v in dict(view_names).items()}
        if parent_dir:
            meta["parent_dir"] = str(parent_dir)
        with open(project_dir / "project.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return meta
    
    def open_project(self, name):
        path = self._resolve_project_dir(name) / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"?? '{name}' ???")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def register_external_project(self, dirpath):
        """注册（导入）工作区之外的已有工程目录，返回工程名。

        用于「Open Project」选择目录后，把包含 project.json 的文件夹登记到
        custom-dirs 注册表，使 list_projects() 能列出、open_project() 能解析。
        同时把 meta 里从其它机器拷贝过来后失效的 parent_dir 修正为当前实际父目录。
        """
        d = Path(dirpath).resolve()
        meta_path = d / "project.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"所选目录不含 project.json: {d}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        name = meta.get("name") or d.name
        meta["parent_dir"] = str(d.parent)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        self._custom_dirs[name] = str(d)
        self._save_custom_dirs()
        return name

    def save_project_meta(self, name, meta):
        path = self._resolve_project_dir(name) / "project.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    def list_projects(self):
        projects = []
        seen = set()
        # Workspace projects
        for d in self.workspace.iterdir():
            pj = d / "project.json"
            if d.is_dir() and pj.exists() and not d.name.startswith("."):
                stat = pj.stat()
                projects.append({
                    "name": d.name,
                    "modified": stat.st_mtime,
                })
                seen.add(d.name)
        # Custom-dir projects
        for name, dirpath in self._custom_dirs.items():
            if name in seen:
                continue
            p = Path(dirpath)
            pj = p / "project.json"
            if p.exists() and pj.exists():
                stat = pj.stat()
                projects.append({
                    "name": name,
                    "modified": stat.st_mtime,
                })
        projects.sort(key=lambda x: x["modified"], reverse=True)
        return projects
    
    def delete_project(self, name):
        import shutil
        project_dir = self._resolve_project_dir(name)
        if not project_dir.exists():
            raise FileNotFoundError(f"?? '{name}' ???")
        shutil.rmtree(str(project_dir))
        if name in self._custom_dirs:
            del self._custom_dirs[name]
            self._save_custom_dirs()

    def get_project_dir(self, name):
        return self._resolve_project_dir(name)


# 工程里的 task_type 有中文（新建时写）和英文（旧版/外部导入）两套写法。
# 界面原来在 `_open_project` 里内联了一张同样的表，还在 try 里赋值 —— 读不到
# project.json 时 `internal` 直接未绑定，紧接着的分支就 NameError。收到这里一份，
# 谁要判断工程类型就调用它，别再各抄一张表。
TASK_TYPE_ALIASES = {
    "语义分割": "segmentation",
    "目标检测": "detection",
    "图像分类": "classification",
    "混合分类": "mixed_classification",
    "OCR文字识别": "ocr",
    "OCV字符质检": "ocv",
    # 兼容旧版英文 task_type
    "semantic_segmentation": "segmentation",
    "object_detection": "detection",
    "image_classification": "classification",
    "mixed_classification": "mixed_classification",
    "ocr_recognition": "ocr",
    "ocv": "ocv",
}


def read_task_type(project_dir, default="segmentation"):
    """读工程 project.json 的 task_type，统一转成内部英文键。

    读不到文件 / 字段不认识 → 返回 default（**不抛异常**，也不返回 None：
    调用方拿它去分支，None 会变成"哪种都不是"的静默错）。
    """
    if not project_dir:
        return default
    try:
        with open(os.path.join(str(project_dir), "project.json"),
                  "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return default
    raw = meta.get("task_type") or ""
    return TASK_TYPE_ALIASES.get(raw, default)

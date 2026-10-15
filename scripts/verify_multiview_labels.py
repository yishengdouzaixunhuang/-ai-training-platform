# -*- coding: utf-8 -*-
"""三视图标注标签一致性验证 —— 主画布 vs 参考窗（回归测试）

守住的 bug（2026-10-13 维护者 报）：
    三视图中**只有漫反射图（主画布）**显示标注标签（`IGNORE` / 类别名如 `Ys`），
    形状图1 / 形状图2 只有掩膜斜纹色块、一个字的标签都没有。

根因：标签是主画布 paintEvent 里 `_draw_class_labels()` 画的；参考窗
    (`_RefPanel.render_view`) 只叠加"仅掩膜"的透明层 —— **从来没有画标签这段逻辑**。
修法：标签绘制抽到 `core/label_overlay.paint_shape_labels()`，两处共用同一份；
    参考窗新增 `set_labels()`，数据源与掩膜同源（主画布当前图的 json）。

本脚本两条腿都要过：
  ① 程序核验（防假绿）：**走真实产品路径** `MultiViewPreview._sync_from_canvas()`，
     然后对每个参考窗做"有标签 / 无标签"逐像素 diff > 阈值。
     —— 只断言 `panel._shapes is not None` 是不够的（变量被赋了值不等于像素画上去了，
        这个项目已经吃过一次"假绿"：断言命中注释、形式 PASS）。
  ② 肉眼核验：输出三窗并排截图 `_ui_render/mv_labels[_real].png`。

对照组的做法：把标注 json 临时改名（`os.replace`，本机 safe-delete 钩子会拦 unlink）
→ 参考窗拿不到 shapes → 渲染出的就是"改前那种只有色块、没有标签"的图。
—— 也就是说这个 diff 正是"修复前 vs 修复后"的像素差。

用法（必须 aitp_venv + 关沙箱，理由同 render_panel.py：沙箱里 import torch 会 WinError 1114）：
    C:/aitp_venv/Scripts/python.exe scripts/verify_multiview_labels.py
    C:/aitp_venv/Scripts/python.exe scripts/verify_multiview_labels.py --real "17_31_凹点"

⚠️ 真实模式只**读**你的标注/原图，复制副本到 `tmp/mvtest_real/` 里跑，原文件一个字不动。
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_ui_render"
# 造的假工程放 `tmp/`（已在 .gitignore 里）：里面的 jpg / json / BMP **不会被**忽略，
# 放 _ui_render 会脏了仓库（*.png 有忽略，*.json 没有）。
FIX_SYN = ROOT / "tmp" / "mvtest"
FIX_REAL = ROOT / "tmp" / "mvtest_real"

# 真实 labelme 数据（只读来源；副本进 FIX_REAL）
REAL_IMG_DIR = Path(r"D:/01_Arthur/孚能/20260816-100张labelMe/原图_2026816212540")
REAL_JSON_DIR = Path(r"D:/01_Arthur/孚能/20260816-100张labelMe/Label_2026816212153")

# offscreen + 手动挂字体（不挂就是一堆空白块）—— 同 render_panel.py
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ⚠️ 先 torch、再碰 sys.path，且**绝不能**先 import PyQt5（否则 torch 报 WinError 1114）
os.chdir(ROOT)
import torch  # noqa: F401,E402

sys.path.insert(0, str(ROOT))

_spec = importlib.util.find_spec("PyQt5")
if _spec is not None and _spec.origin:
    os.environ.setdefault("QT_PLUGIN_PATH", str(Path(_spec.origin).parent / "Qt5" / "plugins"))

import numpy as np  # noqa: E402
from PyQt5.QtCore import QEventLoop, Qt, QTimer  # noqa: E402
from PyQt5.QtGui import QFontDatabase, QImage  # noqa: E402
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget  # noqa: E402

_FONT_FILES = (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyhbd.ttc",
               r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\segoeui.ttf")

W, H = 400, 1200            # 合成图尺寸（真图用原尺寸）
IMG_WIN = (1680, 1000)      # 宿主窗口尺寸


def pump(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(ms), loop.quit)
    loop.exec_()


def load_fonts() -> int:
    return sum(1 for fp in _FONT_FILES
               if os.path.exists(fp) and QFontDatabase.addApplicationFont(fp) != -1)


def write_project_json(root: Path, labels) -> None:
    """类别表唯一真源：project.json（非 ignore 的 label 必须在里面，否则被打成背景）。"""
    classes = ["background"]
    for lb in labels:
        if lb and "ignore" not in lb.lower() and "gnore" not in lb.lower() and lb not in classes:
            classes.append(lb)
    (root / "project.json").write_text(json.dumps({"classes": classes}, ensure_ascii=False,
                                                  indent=2), encoding="utf-8")
    return classes


# ─────────────── fixture: 合成 ───────────────
def make_synthetic(root: Path) -> tuple:
    """造最小三视图工程：三张灰阶图 + 一份标注 json + project.json。"""
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    # 三张图底色拉开差距：漫反射浅灰、形状图1 中灰、形状图2 近黑
    # （形状图2 近黑时白字标签最显眼；也顺带覆盖"深底上看不看得见标签"）
    base = {"chk_te_0": 200, "chk_te_1": 140, "chk_te_2": 30}
    rng = np.random.RandomState(7)
    for stem, lv in base.items():
        arr = np.full((H, W), lv, np.uint8)
        for (cy, cx, r) in ((180, 120, 34), (520, 300, 26), (980, 150, 30)):
            yy, xx = np.mgrid[0:H, 0:W]
            arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = min(255, lv + 90)
        arr = np.clip(arr.astype(np.int16) + rng.randint(-6, 7, arr.shape), 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").convert("RGB").save(root / f"{stem}.jpg", quality=95)

    json_path = root / "chk_te_0.json"
    shapes = [
        {"label": "ignore", "points": [[44, 60], [356, 60], [356, 150], [44, 150]],
         "shape_type": "polygon", "group_id": None, "flags": {}},
        {"label": "ignore", "points": [[44, 900], [356, 900], [356, 990], [44, 990]],
         "shape_type": "polygon", "group_id": None, "flags": {}},
        {"label": "Ys", "points": [[70, 320], [240, 320], [240, 400], [70, 400]],
         "shape_type": "polygon", "group_id": None, "flags": {}},
    ]
    json_path.write_text(json.dumps({
        "version": "5.0", "flags": {}, "imagePath": "chk_te_0.jpg",
        "imageData": None, "imageWidth": W, "imageHeight": H, "shapes": shapes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    write_project_json(root, [s["label"] for s in shapes])
    return (root / "chk_te_0.jpg", root / "chk_te_1.jpg", root / "chk_te_2.jpg", json_path)


# ─────────────── fixture: 真实 labelme 数据 ───────────────
def make_real(root: Path, group: str) -> tuple:
    """从真实 labelme 导出目录取一组三视图（图 + te_0.json）复制进 root。

    原文件只读、绝不修改；`load_mask_from_json` 要求 json 与图**同目录同名**，
    所以必须复制副本（源目录里图在一个文件夹、json 在另一个）。
    """
    root.mkdir(parents=True, exist_ok=True)
    src = None
    for ext in (".BMP", ".bmp", ".jpg", ".png"):
        cand = REAL_IMG_DIR / f"{group}_te_0{ext}"
        if cand.exists():
            src = ext
            break
    if src is None:
        raise SystemExit(f"找不到真实图 {group}_te_0.*（在 {REAL_IMG_DIR}）")

    labels = []
    paths = []
    for i, te in enumerate(("te_0", "te_1", "te_2")):
        img_src = REAL_IMG_DIR / f"{group}_{te}{src}"
        if not img_src.exists():
            raise SystemExit(f"缺图 {img_src}")
        dst = root / f"{group}_{te}{src}"
        shutil.copy2(img_src, dst)          # copy2 只读源，不动它
        paths.append(dst)
        js = REAL_JSON_DIR / f"{group}_{te}.json"
        if not js.exists():
            raise SystemExit(f"缺 json {js}")
        if i == 0:
            shutil.copy2(js, root / f"{group}_{te}.json")   # 只给主视图配 json
            labels = [s.get("label", "") for s in json.loads(js.read_text(encoding="utf-8")).get("shapes", [])]
    # 参考视图的 json 也放一份（真实工程里每视图都有一份、内容相同）——
    # 顺便验证"标签取自哪个视图的 json 都等价"这个前提还成立
    for te in ("te_1", "te_2"):
        js = REAL_JSON_DIR / f"{group}_{te}.json"
        if js.exists():
            shutil.copy2(js, root / f"{group}_{te}.json")
    classes = write_project_json(root, labels)
    print(f"真实类别表 = {classes}")
    return (paths[0], paths[1], paths[2], root / f"{group}_te_0.json")


# ─────────────── 取像素 ───────────────
def _img_to_arr(qimg) -> np.ndarray:
    qimg = qimg.convertToFormat(QImage.Format_RGB888)
    h, wd, stride = qimg.height(), qimg.width(), qimg.bytesPerLine()
    buf = np.frombuffer(qimg.constBits().asstring(stride * h), np.uint8)
    return buf.reshape(h, stride)[:, :wd * 3].reshape(h, wd, 3).copy()


def widget_rgb(w) -> np.ndarray:
    return _img_to_arr(w.grab().toImage())


def panel_rgb(panel) -> np.ndarray:
    """取参考窗上一次 render_view 落进 _buf 的画面（直接看渲染结果）。"""
    if panel._buf is None:
        return np.zeros((1, 1, 3), np.uint8)
    return _img_to_arr(panel._buf.toImage())


def diff_px(a: np.ndarray, b: np.ndarray, thr: int = 12) -> int:
    if a.shape != b.shape:
        return -1
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).sum(axis=2)
    return int((d > thr).sum())


# ─────────────── 主流程 ───────────────
def main(argv: list[str]) -> int:
    real_group = None
    if "--real" in argv:
        i = argv.index("--real")
        real_group = argv[i + 1] if i + 1 < len(argv) else "17_31_凹点"
    tag_name = "real" if real_group else "syn"

    OUT.mkdir(exist_ok=True)
    if real_group:
        te0, te1, te2, json_path = make_real(FIX_REAL, real_group)
        print(f"fixture(真实): {FIX_REAL}  组={real_group}")
    else:
        te0, te1, te2, json_path = make_synthetic(FIX_SYN)
        print(f"fixture(合成): {FIX_SYN}  ({W}x{H} ×3 视图 + chk_te_0.json)")

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])  # noqa: F841
    print(f"系统字体挂载 {load_fonts()} 个")

    from annotation.mask_editor import AnnotationCanvas
    from ui.multiview_preview import MultiViewPreview

    canvas = AnnotationCanvas()
    canvas.set_project(str(json_path.parent))
    canvas.load_image(str(te0))
    uniq = sorted(set(np.unique(canvas.mask).tolist())) if canvas.mask is not None else []
    print(f"主画布 image={canvas.image.size} mask={getattr(canvas.mask, 'shape', None)} 类别值={uniq}")

    host = QWidget()
    lay = QVBoxLayout(host)
    lay.setContentsMargins(0, 0, 0, 0)
    mv = MultiViewPreview(canvas)
    lay.addWidget(mv)
    host.resize(*IMG_WIN)
    host.setAttribute(Qt.WA_DontShowOnScreen, True)
    host.show()
    pump(700)

    mv.set_views([(str(te0), "漫反射"), (str(te1), "形状图1"), (str(te2), "形状图2")])
    pump(1200)                        # 等三个面板落尺寸（render_view 靠尺寸算 fit）
    mv._sync_from_canvas(force=True)  # ← 真实产品路径：它自己去 canvas 取 shapes 喂给参考窗
    pump(120)
    print(f"窗口 {host.width()}x{host.height()} / 参考窗 {len(mv._panels)} 个"
          f" / canvas {canvas.width()}x{canvas.height()} zoom={canvas.zoom_level:.4f}")

    fails = []

    # ---- A：产品路径确实把 shapes 取到了（数据源接上了）----
    n_shapes = len(getattr(mv, "_lab_shapes_cache") or [])
    n_draw = sum(1 for s in (getattr(mv, "_lab_shapes_cache") or [])
                 if len(s.get("points") or []) >= 2)
    print(f"\n[A] _current_shapes(): {n_shapes} 条标注（其中可画标签 {n_draw} 条）,"
          f" tag={getattr(mv, '_lab_tag_cache', None)}")
    if n_shapes < 1:
        fails.append("A: 产品路径没取到任何 shapes")
    for i, p in enumerate(mv._panels, start=1):
        if not p._shapes:
            fails.append(f"A: 参考窗{i} 没拿到 shapes")

    # ---- B：参考窗真的把标签画到像素上（有/无标签逐像素 diff）----
    print("\n[B] 参考窗标签像素 diff（有标签 vs 把 json 改名后的无标签）")
    got = mv._canvas_center_in_image()
    if got is None:
        print("!! 取不到画布中心，配置有问题")
        return 4
    z, cx, cy = got

    with_lab = []
    for p in mv._panels:
        p._state = None
        p.render_view(z, cx, cy)
        with_lab.append(panel_rgb(p))

    bak = json_path.with_suffix(json_path.suffix + ".bak")
    os.replace(str(json_path), str(bak))      # 本机 safe-delete 钩子拦 unlink，改名用 os.replace
    try:
        mv._sync_from_canvas(force=True)
        pump(60)
        for i, p in enumerate(mv._panels, start=1):
            p._state = None
            p.render_view(z, cx, cy)
            without = panel_rgb(p)
            n = diff_px(with_lab[i - 1], without)
            hh, ww = without.shape[:2]
            print(f"  参考窗{i}({p.title.text()}): 尺寸 {ww}x{hh}  标签像素差 = {n}")
            if n <= 300:
                fails.append(f"B: 参考窗{i} 的标签像素差只有 {n}（<=300）—— 标签没画上去")
    finally:
        os.replace(str(bak), str(json_path))

    # ---- C：主画布回归（改后仍画标签，别把主画布的弄丢了）----
    print("\n[C] 主画布回归：关掉两个显示开关后应有像素差")
    canvas.show()
    pump(200)
    on = widget_rgb(canvas)
    canvas._show_annotation = False
    canvas._show_ignore = False
    canvas.update()
    pump(200)
    off = widget_rgb(canvas)
    n_main = diff_px(on, off)
    print(f"  主画布 {canvas.width()}x{canvas.height()}  开关前后像素差 = {n_main}")
    if n_main <= 300:
        fails.append(f"C: 主画布标签像素差只有 {n_main}（<=300）")
    canvas._show_annotation = True
    canvas._show_ignore = True
    canvas.update()

    # ---- 出图 ----
    mv._sync_from_canvas(force=True)
    pump(200)
    shot = OUT / f"mv_labels{'_' + tag_name if real_group else ''}.png"
    host.grab().save(str(shot))
    print(f"\n并排截图 -> {shot}")

    print("\n" + "=" * 56)
    if fails:
        print(f"FAIL  {len(fails)} 项")
        for f in fails:
            print("  - " + f)
        return 1
    print(f"PASS[{tag_name}]  三窗标签一致性：产品路径取数 ✓  参考窗像素 ✓  主画布回归 ✓")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

"""Heatmap region analyzer — extract hotspots from Grad-CAM / class activation heatmaps.

零依赖（仅 numpy + PIL）。支持：
- RGB 热度提取（暖色优先：R - 0.5*(G+B)，再做最大值归一化）
- 自动 OTSU 阈值
- 4-邻 BFS 连通域分析
- 基于归一化坐标的网格聚类（=NG 种类估计）
- 文件名诊断标签提取（NG/误判/OK/区域/日期/缺陷描述）
- 原图上画 bbox + 中心点 + 标签
- HTML 报告导出（含标注图）

用法：
    from tools.heatmap_analyzer import analyze_folder, render_html_report
    result = analyze_folder(folder_path, recursive=False, min_area=30)
    render_html_report(result, output_dir)
"""
import os
import re
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXT = {'.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp', '.gif'}


# ─────────────────────────────── I/O ───────────────────────────────

def list_images(folder: str, recursive: bool = False) -> List[str]:
    """列出文件夹内所有支持格式的图片，按文件名排序。"""
    paths: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                paths.append(full)
    paths.sort()
    return paths


# ─────────────────────────────── 热度计算 ───────────────────────────────

def compute_heatmap(image: Image.Image) -> np.ndarray:
    """RGB → 0~1 的热度图。暖色（红/黄）分数高，冷色（蓝/紫）分数低。

    heat = max(R - 0.5*(G+B), 0)，再做最大值归一化。
    """
    arr = np.asarray(image.convert('RGB'), dtype=np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    heat = R - 0.5 * (G + B)
    heat = np.maximum(heat, 0.0)
    m = float(heat.max())
    if m > 1e-6:
        heat /= m
    return heat  # HxW, [0, 1]


def otsu_threshold(heat: np.ndarray) -> float:
    """OTSU 自动阈值（256 直方图）。"""
    hist, _ = np.histogram(heat.flatten(), bins=256, range=(0, 1))
    total = int(hist.sum())
    if total == 0:
        return 0.5
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    best_t = 0.5
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var = var
            best_t = t / 255.0
    return best_t


# ─────────────────────────────── 连通域 ───────────────────────────────

def _bfs_label(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-邻 BFS 连通域标记。
    mask: HxW 二值（0/1）。返回 labeled (HxW) 和连通域数。"""
    h, w = mask.shape
    labeled = np.zeros((h, w), dtype=np.int32)
    next_label = 0
    stack: List[Tuple[int, int]] = []
    for i in range(h):
        row = mask[i]
        for j in range(w):
            if row[j] and not labeled[i, j]:
                next_label += 1
                labeled[i, j] = next_label
                stack.append((i, j))
                while stack:
                    y, x = stack.pop()
                    ly = labeled[y]
                    if y > 0 and mask[y - 1, x] and not labeled[y - 1, x]:
                        labeled[y - 1, x] = next_label
                        stack.append((y - 1, x))
                    if y < h - 1 and mask[y + 1, x] and not labeled[y + 1, x]:
                        labeled[y + 1, x] = next_label
                        stack.append((y + 1, x))
                    if x > 0 and mask[y, x - 1] and not labeled[y, x - 1]:
                        labeled[y, x - 1] = next_label
                        stack.append((y, x - 1))
                    if x < w - 1 and mask[y, x + 1] and not labeled[y, x + 1]:
                        labeled[y, x + 1] = next_label
                        stack.append((y, x + 1))
    return labeled, next_label


def detect_hotspots(heat: np.ndarray, threshold: float = 0.4, min_area: int = 30) -> List[Dict]:
    """检测热度图中的高亮区域（连通域）。

    返回 list[dict]，每个区域包含：
        cx, cy: 像素中心
        norm_cx, norm_cy: [0,1] 归一化中心（用于跨图聚类）
        bbox: (x, y, w, h)
        area: 像素数
        area_ratio: 区域占图像比例
        mean_intensity, max_intensity: [0,1]
    按 max_intensity 降序。
    """
    mask = (heat >= threshold).astype(np.uint8)
    labeled, n = _bfs_label(mask)
    h, w = heat.shape
    spots: List[Dict] = []
    for k in range(1, n + 1):
        ys, xs = np.where(labeled == k)
        area = int(len(xs))
        if area < min_area:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        intensities = heat[ys, xs]
        spots.append({
            'cx': cx,
            'cy': cy,
            'norm_cx': cx / max(w - 1, 1),
            'norm_cy': cy / max(h - 1, 1),
            'bbox': (x0, y0, x1 - x0 + 1, y1 - y0 + 1),
            'area': area,
            'area_ratio': area / float(h * w),
            'mean_intensity': float(intensities.mean()),
            'max_intensity': float(intensities.max()),
        })
    spots.sort(key=lambda r: -r['max_intensity'])
    return spots


# ─────────────────────────────── 聚类 ───────────────────────────────

def cluster_hotspots(all_hotspots: List[Dict],
                     grid_w: int = 6, grid_h: int = 3) -> List[Dict]:
    """基于归一化坐标的网格聚类 —— 把图像分成 grid_w × grid_h 的格子。

    返回 list[dict]，每个聚类包含：
        cluster_id, cell (gx, gy), norm_center, count, hotspot_indices
    按 count 降序。
    """
    cells: Dict[Tuple[int, int], List[int]] = {}
    for idx, h in enumerate(all_hotspots):
        gx = int(h['norm_cx'] * grid_w)
        gy = int(h['norm_cy'] * grid_h)
        gx = max(0, min(gx, grid_w - 1))
        gy = max(0, min(gy, grid_h - 1))
        cells.setdefault((gx, gy), []).append(idx)
    out: List[Dict] = []
    for (gx, gy), idxs in sorted(cells.items(), key=lambda kv: -len(kv[1])):
        out.append({
            'cluster_id': len(out) + 1,
            'cell': (gx, gy),
            'norm_center': ((gx + 0.5) / grid_w, (gy + 0.5) / grid_h),
            'count': len(idxs),
            'hotspot_indices': idxs,
        })
    return out


# ─────────────────────────────── 文件名标签 ───────────────────────────────

_TAG_PATTERNS = {
    'date': re.compile(r'^(\d{8})'),
    'index': re.compile(r'\((\d+)\)$'),
    'region': re.compile(r'区域\s*(\d+)'),
}


def extract_filename_tags(filename: str) -> Dict[str, str]:
    """从文件名提取诊断标签。

    示例: "20260718正面铜翻折-0717区域1误判 (34).bmp"
        → {date:20260718, defect:'正面铜翻折', region:'区域1', judgement:'误判', index:'34'}
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    info: Dict[str, str] = {}
    for key, pat in _TAG_PATTERNS.items():
        m = pat.search(base)
        if m:
            info[key] = m.group(1) if key == 'index' else m.group(0)
    if 'judgement' not in info:
        if 'NG' in base:
            info['judgement'] = 'NG'
        elif '误判' in base:
            info['judgement'] = '误判'
        elif 'OK' in base:
            info['judgement'] = 'OK'
    # 缺陷描述：日期之后到 "区域|NG|误判|(" 之前的中文+数字
    m = re.match(r'^\d{0,8}\s*(.*?)(?:区域|NG|误判|OK|\(|$)', base)
    if m and m.group(1):
        defect = m.group(1).strip(' -_')
        if defect and not defect.isdigit():
            info['defect'] = defect
    return info


# ─────────────────────────────── 可视化 ───────────────────────────────

def annotate_image(image: Image.Image, hotspots: List[Dict],
                   color: Tuple[int, int, int] = (255, 64, 64)) -> Image.Image:
    """在原图上画检测到的高亮区域：bbox + 中心圆 + 编号标签。"""
    img = image.convert('RGB').copy()
    draw = ImageDraw.Draw(img)
    try:
        font_size = max(14, min(img.size) // 30)
        font = ImageFont.truetype('arial.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    stroke = max(2, min(img.size) // 200)
    for i, h in enumerate(hotspots, 1):
        x, y, w, hh = h['bbox']
        draw.rectangle([x, y, x + w, y + hh], outline=color, width=stroke)
        cx, cy = int(h['cx']), int(h['cy'])
        r = max(3, min(img.size) // 150)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
        label = f"#{i} {h['area']}px {h['mean_intensity']:.2f}"
        draw.text((x + 3, max(0, y - font_size - 2)), label, fill=color, font=font)
    return img


# ─────────────────────────────── 主流程 ───────────────────────────────

ProgressCb = Optional[Callable[[int, int, str], None]]


def analyze_folder(folder: str,
                   recursive: bool = False,
                   threshold: Optional[float] = None,
                   min_area: int = 30,
                   grid_w: int = 6,
                   grid_h: int = 3,
                   max_dim: Optional[int] = None,
                   progress_cb: ProgressCb = None,
                   cancel_cb: Optional[Callable[[], bool]] = None) -> Dict:
    """分析文件夹，返回完整报告 dict。

    参数:
        folder:  图片文件夹
        recursive: 是否递归子文件夹
        threshold: 固定阈值（None → 自动 OTSU，每张图独立）
        min_area: 最小连通域像素数
        grid_w, grid_h: 聚类网格大小
        max_dim: 最大边长缩放（None 保持原尺寸）
        progress_cb: 回调(done, total, current_filename)
        cancel_cb: 回调返回 True 则提前终止（用于用户取消）

    返回 dict: {
        'files': [...],
        'all_hotspots': [...],
        'clusters': [...],
        'params': {...},
        'summary': {total, with_hotspots, n_hotspots, n_clusters, errors, cancelled}
    }
    """
    paths = list_images(folder, recursive)
    per_file: List[Dict] = []
    all_hotspots: List[Dict] = []
    cancelled = False
    for i, p in enumerate(paths):
        if cancel_cb and cancel_cb():
            cancelled = True
            break
        entry: Dict = {
            'path': p,
            'filename': os.path.basename(p),
            'tags': extract_filename_tags(p),
            'hotspots': [],
            'nhotspots': 0,
        }
        try:
            with Image.open(p) as img:
                if max_dim and max(img.size) > max_dim:
                    img.thumbnail((max_dim, max_dim), Image.BILINEAR)
                w, h = img.size
                heat = compute_heatmap(img)
            t = float(otsu_threshold(heat)) if threshold is None else float(threshold)
            spots = detect_hotspots(heat, threshold=t, min_area=min_area)
            for s in spots:
                s['file_index'] = i
                s['filename'] = entry['filename']
                s['file_tags'] = entry['tags']
                all_hotspots.append(s)
            entry.update({
                'size': (w, h),
                'threshold': t,
                'hotspots': spots,
                'nhotspots': len(spots),
            })
        except Exception as e:  # noqa: BLE001
            entry['error'] = f"{type(e).__name__}: {e}"
        per_file.append(entry)
        if progress_cb:
            try:
                progress_cb(i + 1, len(paths), entry['filename'])
            except Exception:
                pass

    clusters = cluster_hotspots(all_hotspots, grid_w=grid_w, grid_h=grid_h)
    return {
        'files': per_file,
        'all_hotspots': all_hotspots,
        'clusters': clusters,
        'params': {
            'folder': folder,
            'recursive': recursive,
            'threshold': threshold,
            'auto_threshold': threshold is None,
            'min_area': min_area,
            'grid_w': grid_w,
            'grid_h': grid_h,
            'max_dim': max_dim,
        },
        'summary': {
            'total': len(per_file),
            'with_hotspots': sum(1 for f in per_file if f.get('nhotspots', 0) > 0),
            'n_hotspots': len(all_hotspots),
            'n_clusters': len(clusters),
            'errors': sum(1 for f in per_file if 'error' in f),
            'cancelled': cancelled,
        },
    }


# ─────────────────────────────── 报告 ───────────────────────────────

def render_html_report(result: Dict, output_dir: str,
                       progress_cb: ProgressCb = None,
                       cancel_cb: Optional[Callable[[], bool]] = None) -> str:
    """渲染 HTML 报告并落盘标注图，返回 report.html 路径。

    progress_cb(done, total, stage_name) / cancel_cb() → bool（提前停止）。
    单张标注图失败只跳过该图，不中断整体渲染。
    """
    os.makedirs(output_dir, exist_ok=True)
    annotated_dir = os.path.join(output_dir, 'annotated')
    os.makedirs(annotated_dir, exist_ok=True)
    summary = result['summary']
    clusters = result['clusters']
    files = result['files']
    params = result.get('params', {})

    overview_rows = ''.join(
        f'<tr><td>{k}</td><td>{v}</td></tr>'
        for k, v in (
            ('总图片数', summary['total']),
            ('含高亮图片数', summary['with_hotspots']),
            ('总高亮区域数', summary['n_hotspots']),
            ('聚类种类数', summary['n_clusters']),
            ('失败图片数', summary.get('errors', 0)),
            ('阈值', '自动 OTSU' if params.get('auto_threshold') else params.get('threshold')),
            ('最小面积', params.get('min_area')),
            ('网格', f"{params.get('grid_w')}x{params.get('grid_h')}"),
        )
    )

    cluster_rows = ''.join(
        f'<tr><td>#{c["cluster_id"]}</td><td>({c["cell"][0]}, {c["cell"][1]})</td>'
        f'<td>({c["norm_center"][0]:.2f}, {c["norm_center"][1]:.2f})</td>'
        f'<td>{c["count"]}</td></tr>'
        for c in clusters
    )

    detail_rows = []
    for idx, f in enumerate(files):
        if cancel_cb and cancel_cb():
            break
        rel_path = ''
        try:
            if 'error' not in f:
                with Image.open(f['path']) as img:
                    annotated = annotate_image(img, f.get('hotspots', []))
                ann_name = os.path.splitext(f['filename'])[0] + '.png'
                ann_path = os.path.join(annotated_dir, ann_name)
                annotated.save(ann_path)
                rel_path = os.path.relpath(ann_path, output_dir).replace('\\', '/')
        except Exception:
            rel_path = ''
        spots_str = '<br>'.join(
            f"#{i+1} 中心({h['cx']:.0f},{h['cy']:.0f}) 面积 {h['area']}px "
            f"强度 {h['mean_intensity']:.2f}"
            for i, h in enumerate(f.get('hotspots', []))
        ) or '-'
        tags = f.get('tags', {})
        tag_html = ' '.join(
            f'<span class="tag">{k}:{v}</span>' for k, v in tags.items()
        ) or '-'
        size_str = '×'.join(str(x) for x in f.get('size', ())) if 'size' in f else '-'
        detail_rows.append(
            f'<tr><td>{f["filename"]}</td><td>{size_str}</td>'
            f'<td>{f.get("nhotspots", 0)}</td><td>{spots_str}</td>'
            f'<td>{tag_html}</td>'
            f'<td><a href="{rel_path}" target="_blank">'
            f'<img src="{rel_path}" style="max-width:300px;max-height:80px;border:1px solid #ddd"></a></td></tr>'
        )
        if progress_cb:
            try:
                progress_cb(idx + 1, len(files), 'render')
            except Exception:
                pass

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Heatmap Region Report</title>
<style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;margin:20px;color:#333;background:#fdfdfd}}
h1{{color:#c0392b;border-bottom:2px solid #c0392b;padding-bottom:8px}}
h2{{color:#2c3e50;margin-top:30px}}
table{{border-collapse:collapse;margin:10px 0;width:100%;background:white}}
th,td{{border:1px solid #e0e0e0;padding:8px 12px;text-align:left;font-size:13px;vertical-align:top}}
th{{background:#f4f6f8;font-weight:600}}
tr:nth-child(even){{background:#fafafa}}
.section{{margin-bottom:30px;background:white;padding:15px;border-radius:8px;
       box-shadow:0 1px 3px rgba(0,0,0,0.05)}}
.summary table{{width:auto}}
.summary td:first-child{{font-weight:600;min-width:140px;background:#fef9e7}}
.tag{{display:inline-block;padding:2px 8px;background:#3498db;color:white;
     border-radius:10px;font-size:12px;margin:2px}}
.empty{{color:#999;font-style:italic}}
</style></head><body>
<h1>🔍 Heatmap Region Analysis Report</h1>
<div class="section summary">
<h2>总览</h2>
<table>{overview_rows}</table>
</div>
<div class="section">
<h2>聚类摘要（按位置归类 = NG 种类估计）</h2>
{'<table><tr><th>类#</th><th>网格坐标</th><th>归一化中心</th><th>样本数</th></tr>' + cluster_rows + '</table>' if cluster_rows else '<p class="empty">无聚类结果</p>'}
</div>
<div class="section">
<h2>每图详情</h2>
<table><tr><th>文件名</th><th>尺寸</th><th>高亮数</th><th>高亮位置</th><th>标签</th><th>标注图</th></tr>
{detail_rows}</table>
</div>
</body></html>
"""
    out_path = os.path.join(output_dir, 'report.html')
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write(html)
    return out_path


__all__ = [
    'list_images', 'compute_heatmap', 'otsu_threshold', 'detect_hotspots',
    'cluster_hotspots', 'extract_filename_tags', 'annotate_image',
    'analyze_folder', 'render_html_report',
    'SUPPORTED_EXT',
]
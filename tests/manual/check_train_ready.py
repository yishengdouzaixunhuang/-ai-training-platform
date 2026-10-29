"""训练条件体检：一个工程"现在能不能训"，跑一条命令就知道。

用途：每次往 images/ 加完图、标完注，先跑这个，别等训练跑废了才发现。

    C:\\aitp_venv\\Scripts\\python.exe check_train_ready.py <工程目录>

检查项（全部走**产品自己的**函数，不另写一套判断）：
  1. 工程元信息：任务类型 / 类别表 / 视图 / 输入通道
  2. 图片与标注完整性：组数、缺视图、缺标注、空标注、忽略区
  3. 类别分布：每类块数 + 覆盖组数（含"定义了但零样本"的告警）
  4. 数据集划分：train/val/test（全 train 会让验证集靠训练器临时切）
  5. 已有模型的状态：上一次训练到底训成了没有
  6. 算力：torch 能否用上 GPU
  7. 分块块数：开/不开分块、两种取样方式各多少块

提速提示：本脚本**不跑训练**，只算块数。真实吞吐用 `_smoke_mv_speed.py` 测。

⚠️ 不要在这个脚本里 import PyQt5 —— 本机 PyQt5 先于 torch 导入会撞 c10.dll。
   只依赖 training.multiview + torch，与 DL Studio 的训练通路完全一致。
"""
import collections
import json
import os
import re
import sys

try:
    import torch  # noqa: F401  先导 torch
except Exception as _e:  # pragma: no cover
    print("torch 导入失败: %s" % _e)
    sys.exit(2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def hr(title):
    print("\n" + "-" * 74)
    print(title)
    print("-" * 74)


def scan_groups(project_dir):
    """按 `<key>_te_N` 归组；返回 {key: {view: filename}}。"""
    img_dir = os.path.join(project_dir, "images")
    groups = collections.defaultdict(dict)
    if not os.path.isdir(img_dir):
        return groups
    for f in os.listdir(img_dir):
        if not f.lower().endswith(IMG_EXT):
            continue
        base = os.path.splitext(f)[0]
        m = re.match(r"^(.*)_(te_\d+)$", base)
        if m:
            groups[m.group(1)][m.group(2)] = f
        else:
            groups[base]["<无 _te_N 后缀>"] = f
    return groups


def main():
    if len(sys.argv) < 2:
        print("用法: check_train_ready.py <工程目录>")
        return 2
    proj = os.path.abspath(sys.argv[1])
    if not os.path.isdir(proj):
        print("工程目录不存在: %s" % proj)
        return 2

    print("=" * 74)
    print("训练条件体检")
    print("工程: %s" % proj)
    print("=" * 74)

    blocking = []      # 必须处理，否则训了也白训
    warn = []          # 建议处理

    # ── 1. 工程元信息 ──
    hr("[1] 工程元信息")
    pj = {}
    pj_path = os.path.join(proj, "project.json")
    if os.path.exists(pj_path):
        pj = json.load(open(pj_path, encoding="utf-8"))
    classes = pj.get("classes") or []
    views = pj.get("views") or []
    print("  任务类型 : %s" % pj.get("task_type", "<未写>"))
    print("  类别表   : %s" % (classes or "<空>"))
    print("  视图     : %s" % (views or "<单图>"))
    print("  in_channels: %s" % pj.get("in_channels", "<未写>"))
    is_mv = len(views) > 1
    if not classes:
        blocking.append("project.json 没有 classes，掩膜没有类别号可用")
    if is_mv and len(views) * 3 != pj.get("in_channels", len(views) * 3):
        warn.append("in_channels=%s 与 %d 视图 ×3 通道对不上" % (pj.get("in_channels"), len(views)))

    # ── 2. 完整性 ──
    hr("[2] 图片与标注完整性")
    groups = scan_groups(proj)
    n_group = len(groups)
    print("  图片组数 : %d" % n_group)
    if is_mv:
        missing = {k: sorted(set(views) - set(v)) for k, v in groups.items()
                   if set(views) - set(v)}
        print("  缺视图的组: %d %s" % (len(missing), list(missing)[:5]))
        if missing:
            warn.append("%d 组缺视图，训练时会被整组跳过" % len(missing))
    no_ann, empty_ann = [], []
    per_class = collections.Counter()
    cover = collections.defaultdict(set)
    ignore_blocks = 0
    unknown_labels = collections.Counter()
    prim = views[0] if is_mv else None
    for k in sorted(groups):
        if prim:
            jp = os.path.join(proj, "images", "%s_%s.json" % (k, prim))
            mp = os.path.join(proj, "annotations", "%s_%s_mask.png" % (k, prim))
        else:
            jp = os.path.join(proj, "images", k + ".json")
            mp = os.path.join(proj, "annotations", k + "_mask.png")
        if not (os.path.exists(jp) or os.path.exists(mp)):
            no_ann.append(k)
            continue
        if not os.path.exists(jp):
            continue
        try:
            sh = (json.load(open(jp, encoding="utf-8")).get("shapes") or [])
        except Exception:
            sh = []
        if not sh:
            empty_ann.append(k)
        for s in sh:
            lab = str(s.get("label", ""))
            if "ignore" in lab.lower() or "gnore" in lab.lower():
                ignore_blocks += 1
                continue
            if lab in classes:
                per_class[lab] += 1
                cover[lab].add(k)
            else:
                unknown_labels[lab] += 1
    print("  无任何标注的组: %d %s" % (len(no_ann), no_ann[:5]))
    print("  标注为空的组  : %d %s" % (len(empty_ann), empty_ann[:5]))
    print("  忽略区(__ignore__)块: %d（CE 里 ignore_index=255，不参与惩罚）" % ignore_blocks)
    if no_ann:
        warn.append("%d 组没有任何标注（会以全背景参与训练）" % len(no_ann))
    if unknown_labels:
        warn.append("JSON 里有类别表外的标签，会被当背景处理: %s"
                    % dict(unknown_labels))

    # ── 3. 类别分布 ──
    hr("[3] 类别分布（长尾是这类工程的头号问题）")
    print("  %-14s %8s %10s" % ("类别", "块数", "覆盖组数"))
    zero = []
    for c in classes:
        if c == "background":
            continue
        n, g = per_class.get(c, 0), len(cover.get(c, ()))
        flag = ""
        if g == 0:
            flag = "  <== 零样本，永远学不出来"
            zero.append(c)
        elif g <= 3:
            flag = "  <== 极少"
        print("  %-14s %8d %10d%s" % (c, n, g, flag))
    print("  缺陷块合计: %d" % sum(per_class.values()))
    if zero:
        blocking.append("类别 %s 零样本：它会白占一个输出通道，并把 mIoU 一路拉低"
                        % "/".join(zero))

    # ── 4. 划分 ──
    hr("[4] 数据集划分")
    sp_path = os.path.join(proj, "train_test_split.json")
    cnt = collections.Counter()
    sm = {}
    if os.path.exists(sp_path):
        try:
            sm = json.load(open(sp_path, encoding="utf-8"))
        except Exception:
            sm = {}

    def _split_of(key):
        # 组级：只看主视图那一条，避免三视图各算一次、样本数虚增。
        # ★ 2026-10-22：未划分**不再**默认算 train（未划分 = 空集，任何集都不取），
        #   所以这里必须返回 "" 而不是 "train" —— 否则体检报告会替训练器撒谎。
        if not isinstance(sm, dict):
            return ""
        if is_mv:
            return sm.get("%s_%s" % (key, prim), sm.get(key)) or ""
        return sm.get(key) or ""

    for k in sorted(groups):
        cnt[_split_of(k)] += 1
    print("  train=%d  val=%d  test=%d  未划分=%d（按组计，共 %d 组）"
          % (cnt.get("train", 0), cnt.get("val", 0), cnt.get("test", 0),
             cnt.get("", 0), n_group))
    if cnt.get("train", 0) == 0:
        blocking.append("训练集为空：一组都没划分到 train。未划分的组不属于任何集，"
                        "训练会直接报错 —— 去「工具 → 数据集划分管理…」划分")
    if cnt.get("", 0):
        warn.append("%d 组未划分（不属于任何集，训练/验证/测试都不会取它们）"
                    % cnt.get("", 0))
    if cnt.get("val", 0) == 0:
        warn.append("验证集为空 —— 训练器会自动从训练集切 15% 作验证，但不落盘，"
                    "界面上看不到哪几组在验证、也不便手工复核。"
                    "要规范就把主视图那几条的值改成 \"val\"。")
    if cnt.get("test", 0) == 0:
        warn.append("没有留出 test 集：最终指标只能在验证集上看，"
                    "而验证集是被训练过程反复选模用过的，会偏乐观。")

    # ── 5. 已有模型 ──
    hr("[5] 已有模型的状态（上次到底训成没有）")
    mdir = os.path.join(proj, "models")
    ckpts = []
    if os.path.isdir(mdir):
        ckpts = sorted(f for f in os.listdir(mdir) if f.endswith((".pth", ".pt")))
    if not ckpts:
        print("  models/ 下没有模型文件 —— 还没训过")
    for f in ckpts:
        p = os.path.join(mdir, f)
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print("  %-24s 读取失败 %s" % (f, type(e).__name__))
            continue
        if not isinstance(d, dict):
            print("  %-24s 非标准 checkpoint" % f)
            continue
        lm = d.get("last_metrics") or {}
        ious = lm.get("ious") or []
        fg = lm.get("mIoU_fg", d.get("best_mIoU_fg"))
        print("  %-24s epoch=%s  %s  mIoU_fg=%s  size=%s"
              % (f, d.get("epoch"), d.get("created", ""), fg, d.get("image_size")))
        if ious:
            print("      各类 IoU: %s" % [round(float(x), 4) for x in ious])
        if fg in (0, 0.0) or (ious and max(ious[1:] or [0]) == 0):
            blocking.append("模型 %s 的前景 IoU 全为 0 ⇒ 它只学会了\"整张图都判背景\"，"
                            "等于没训出来。别指望它，也别拿它当基线。" % f)
        if d.get("image_size") and not pj.get("tile_size"):
            print("      注: checkpoint 记的 image_size=%s；本工程 project.json 没有 "
                  "tile_size ⇒ 界面上不勾「分块」就是整图缩放到这个尺寸。"
                  % (d.get("image_size"),))

    # ── 6. 算力 ──
    hr("[6] 算力")
    # ⚠️ 别在这里再写 `import torch`：那会让 torch 变成**函数局部变量**，
    #    上面 [5] 里的 `torch.load` 就会抛 UnboundLocalError（实测踩过）。
    #    模块顶部已经导入过了，直接用。
    print("  torch    : %s（CUDA 编译版本 %s）" % (torch.__version__, torch.version.cuda))
    cuda_ok = torch.cuda.is_available()
    print("  cuda 可用: %s" % cuda_ok)
    if cuda_ok:
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            print("    设备%d: %s  %.1f GB" % (i, pr.name, pr.total_memory / 1073741824))
    else:
        blocking.append("torch 用不上 GPU ⇒ 只能 CPU 训练。33 组开分块有几千块，"
                        "CPU 上一轮要几十分钟、200 轮就是几天，实际不可用。"
                        "先确认显卡/驱动（管理员身份跑 nvidia-smi）。")

    # ── 7. 分块块数 ──
    hr("[7] 分块块数（开分块 = 原分辨率裁剪，缺陷像素不丢）")
    if not is_mv:
        print("  单图工程，走普通分割通路；本节跳过")
    else:
        try:
            from training.multiview import MultiViewSegDataset
            for tag, kw in (("不开分块（整图缩放到 256）", dict()),
                            ("滑窗遍历 256/75%", dict(tile_size=256, tile_mode="slide",
                                                   tile_stride_pct=75)),
                            ("缺陷居中+负样本 256", dict(tile_size=256, tile_mode="defect",
                                                   neg_ratio=1.5))):
                ds = MultiViewSegDataset(proj, "train", image_size=256, **kw)
                print("  %-26s 组数=%-4d 块数=%d" % (tag, len(ds.samples), len(ds)))
        except Exception as e:
            warn.append("分块块数计算失败: %s: %s" % (type(e).__name__, e))

    # ── 结论 ──
    hr("结论")
    if blocking:
        print("  ✗ 现在直接开训会白跑，先处理：")
        for i, m in enumerate(blocking, 1):
            print("    %d. %s" % (i, m))
    else:
        print("  ✓ 数据与算力都到位，可以直接开训")
    if warn:
        print("\n  ! 建议（不阻塞，但会明显影响结果可信度）：")
        for i, m in enumerate(warn, 1):
            print("    %d. %s" % (i, m))
    print()
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())

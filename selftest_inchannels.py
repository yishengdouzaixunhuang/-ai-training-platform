# -*- coding: utf-8 -*-
"""9 通道（三视图）训练链路自测 —— 2026-10-15。

**为什么要有这个**：改完 `create_model(in_channels=...)` 只证明"能建模"，
不证明"能训练"。真正会静默出错的三个点，各钉一条断言：

  ① 模型首层通道数对不对（3/6/9 都要能建、都要能前向）
  ② **膨胀初始化**：9 通道时三个视图块必须完全相同且 = RGB 核 × (3/9)。
     写错（比如只复制不缩放）不会报错，只会让激活量级 ×3、训练难收敛
  ③ **数据集真的吐 9 通道**，而且是**按组取样**（一个场景 3 视图 = 1 个样本，
     不是 3 个样本 —— 否则同场景重复训练 + 验证集泄漏，指标虚高）
  ④ 数据集通道数必须与模型一致（不一致就是"能跑但全错"的组合）
  ⑤ checkpoint 里记了 in_channels/views（推理端靠它复现同样的输入）

跑法：`C:/aitp_venv/Scripts/python.exe selftest_inchannels.py`
夹具建在系统临时目录，**不碰任何真实工程**；探针纪律：不写用户的注册表/settings。
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RESULTS = []


def ck(name, ok, detail=""):
    _RESULTS.append((bool(ok), name, str(detail)[:300]))
    print("  %-4s %s%s" % ("PASS" if ok else "FAIL", name,
                           ("  <- " + str(detail)[:200]) if not ok else ""), flush=True)


def _make_project(root, groups=3, views=("te_0", "te_1", "te_2"), size=(96, 64)):
    """造一个三图工程夹具：N 组 × N 视图 + labelme JSON（矩形标注）。"""
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)
    from PIL import Image, ImageDraw
    for g in range(groups):
        stem = "g%02d" % g
        for k, v in enumerate(views):
            arr = 40 + 30 * k + 10 * g
            im = Image.new("L", size, arr)
            ImageDraw.Draw(im).rectangle([8 + 4 * g, 8, 40 + 4 * g, 36], fill=min(255, arr + 60))
            im.save(os.path.join(img_dir, "%s_%s.jpg" % (stem, v)))
        # 标注：只放一份（组名.json），三视图共用
        with open(os.path.join(img_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "imageWidth": size[0], "imageHeight": size[1],
                "shapes": [{"label": "Ys", "shape_type": "rectangle",
                            "points": [[10, 10], [44, 38]]}],
            }, f, ensure_ascii=False)
    with open(os.path.join(root, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": os.path.basename(root), "task_type": "语义分割",
                   "classes": ["background", "Ys", "Zz"],
                   "views": list(views), "view_count": len(views),
                   "in_channels": 3 * len(views)}, f, ensure_ascii=False)
    with open(os.path.join(root, "train_test_split.json"), "w", encoding="utf-8") as f:
        json.dump({"g%02d" % g: "train" for g in range(groups)}, f)
    return root


def _make_single_project(root, groups=3, size=(96, 64)):
    """单视图夹具：每张图配同名 json（`sNN.json`），无 views 字段。"""
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)
    from PIL import Image, ImageDraw
    for g in range(groups):
        base = "s%02d" % g
        im = Image.new("L", size, 50 + 10 * g)
        ImageDraw.Draw(im).rectangle([8, 8, 40, 36], fill=200)
        im.save(os.path.join(img_dir, base + ".jpg"))
        with open(os.path.join(img_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump({"imageWidth": size[0], "imageHeight": size[1],
                       "shapes": [{"label": "Ys", "shape_type": "rectangle",
                                   "points": [[10, 10], [44, 38]]}]}, f, ensure_ascii=False)
    with open(os.path.join(root, "project.json"), "w", encoding="utf-8") as f:
        json.dump({"name": os.path.basename(root), "task_type": "语义分割",
                   "classes": ["background", "Ys", "Zz"]}, f, ensure_ascii=False)
    with open(os.path.join(root, "train_test_split.json"), "w", encoding="utf-8") as f:
        json.dump({"s%02d" % g: "train" for g in range(groups)}, f)
    return root


def main():
    import torch
    from training.models import create_model
    from training.dataset import SegmentationDataset

    # ── ① 模型：多种通道数都能建 + 前向 ──
    for nc in (3, 6, 9):
        m = create_model(3, "deeplabv3", pretrained=False, in_channels=nc)
        out = m(torch.randn(2, nc, 128, 128))["out"]
        ck("模型 in_channels=%d：首层 %d 通道、前向出 %s"
           % (nc, m.backbone.conv1.in_channels, tuple(out.shape)),
           m.backbone.conv1.in_channels == nc and out.shape[1] == 3)

    # ── ② 膨胀初始化规则 ──
    from training.multiview import patch_first_conv
    torch.manual_seed(0)
    m3 = create_model(3, "deeplabv3", pretrained=False, in_channels=3)
    w3 = m3.backbone.conv1.weight.data.clone()
    patch_first_conv(m3, 9)
    w9 = m3.backbone.conv1.weight.data
    dmax = float((w9[:, 0:3] - w9[:, 6:9]).abs().max())
    ratio = float(w9[:, 0:3].abs().mean() / w3.abs().mean())
    ck("膨胀初始化：三个视图块完全相同（差异 %.2e）" % dmax, dmax < 1e-6, dmax)
    ck("膨胀初始化：按 3/9=0.3333 缩放（实测 %.4f）" % ratio, abs(ratio - 1 / 3.0) < 1e-3, ratio)

    # ── ③ 数据集：9 通道 + 按组取样 ──
    tmp = tempfile.mkdtemp(prefix="aitp_mv_")
    try:
        pd = _make_project(tmp, groups=4)
        ds = SegmentationDataset(pd, split="train", image_size=(64, 64),
                                 views=("te_0", "te_1", "te_2"))
        ck("数据集：4 组工程按组取样 = 4 个样本（不是 12）", len(ds._image_samples) == 4,
           "得到 %d" % len(ds._image_samples))
        img, mask = ds[0]
        ck("数据集：吐出 9 通道图像 %s" % (tuple(img.shape),), img.shape[0] == 9, tuple(img.shape))
        ck("数据集：in_channels 与实际张量一致（%d）" % ds.in_channels,
           ds.in_channels == img.shape[0], (ds.in_channels, img.shape[0]))
        ck("数据集：掩膜形状匹配 %s" % (tuple(mask.shape),),
           tuple(mask.shape) == tuple(img.shape[1:]), (tuple(mask.shape), tuple(img.shape[1:])))
        # 不传 views 时必须**完全保持单视图行为**（不能被多视图改动带偏）
        #⚠️ 必须另造一个单视图夹具：三图工程的标注是**组级** `g00.json`，
        #    没有 `g00_te_0.json`，拿它测单视图路径本来就收不到样本（是我第一次写错的地方）。
        tmp_sv = tempfile.mkdtemp(prefix="aitp_sv_")
        try:
            _make_single_project(tmp_sv, groups=3)
            ds1 = SegmentationDataset(tmp_sv, split="train", image_size=(64, 64))
            ck("单视图工程：不传 views 仍取到 3 个样本", len(ds1._image_samples) == 3,
               len(ds1._image_samples))
            i1, _ = ds1[0]
            ck("单视图（不传 views）行为不变：3 通道", i1.shape[0] == 3, tuple(i1.shape))
        finally:
            shutil.rmtree(tmp_sv, ignore_errors=True)

        # 缺视图的组必须被跳过（不能拿别的视图顶替）
        os.remove(os.path.join(pd, "images", "g00_te_2.jpg"))
        ds2 = SegmentationDataset(pd, split="train", image_size=(64, 64),
                                  views=("te_0", "te_1", "te_2"))
        ck("缺视图的组被跳过（4 组 -> 3 组）", len(ds2._image_samples) == 3,
           "得到 %d" % len(ds2._image_samples))

        # ── ④ 数据集与模型能对上（这是"能训练"的真判据）──
        # ⚠️ 必须 batch>=2：训练态BatchNorm 在 batch=1 且空间尺寸塌到 1×1 时会直接报错
        #   （torch 的 _verify_batch_size，与本改动无关，但第一次写断言时踩到了）
        m9 = create_model(3, "deeplabv3", pretrained=False, in_channels=ds.in_channels)
        # 用 ds2（g00 因缺视图已被剔除的那份）：ds 里 g00 的 te_2 文件已被上面删掉
        img, mask = ds2[0]
        img2, mask2 = ds2[1]
        loss = torch.nn.functional.cross_entropy(
            m9(torch.stack([img, img2]))["out"],
            torch.stack([mask, mask2]))
        loss.backward()
        g1 = m9.backbone.conv1.weight.grad
        ck("端到端：9 通道数据 → 9 通道模型 → loss 反传，首层拿到梯度",
           (g1 is not None) and float(g1.abs().sum()) > 0,
           "grad=None" if g1 is None else float(g1.abs().sum()))

        # ── ⑤ trainer 识别多视图 + checkpoint 记通道 ──
        from training.trainer import Trainer
        tr = Trainer(pd, model_name="deeplabv3")
        ck("trainer：识别到 3 视图、in_channels=9（不识别就会静默只训第一个视图）",
           len(tr.views) == 3 and tr.in_channels == 9,
           (tr.views, tr.in_channels))
        tr.init_model()
        ck("trainer：建出的模型首层是 9 通道",
           tr.model.backbone.conv1.in_channels == 9,
           tr.model.backbone.conv1.in_channels)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    n_pass = sum(1 for ok, _, _ in _RESULTS if ok)
    n_fail = len(_RESULTS) - n_pass
    print("=" * 70)
    print("结果：PASS %d ｜ FAIL %d ｜ 合计 %d" % (n_pass, n_fail, len(_RESULTS)))
    print("=" * 70)
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
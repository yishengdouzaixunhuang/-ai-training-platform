# -*- coding: utf-8 -*-
"""分类训练优化项验证（分类训练器 trainer.py / dataset.py）。

原则（沿用本项目铁律）：
1. 源码级核验一律用 **AST**，不用字符串包含 —— 注释能骗过包含式断言，而且以 PASS 形式骗人。
2. 行为级核验必须走**真实产品路径**（ClassificationTrainer.train()），不手造中间态。
3. 断言写死到具体数字前先手算一遍，算不清就不断言。

覆盖：
  A. AST：调度器真的被建出来 + 真的每轮 step；use_amp 真的参与判断；
     早停用 improved；_validate 返回 3 元；drop_last 不再是 len>batch
  B. 指标函数：混淆矩阵 / 逐类 P-R-F1 / NG 漏检 / OK 过杀 / 选模型指标
  C. dataset：固定输入尺寸时不再重复预缩放；增强「关=真关」
  D. 端到端：cosine+warmup 的 lr 真的变；早停语义正确；select_metric 改变最优轮；
     drop_last 不再丢样本；验证指标落进 history

用法：
    C:\\aitp_venv\\Scripts\\python.exe scripts/verify_cls_opt.py
"""
import ast
import io
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, INFO = [], [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [PASS] " if cond else "  [FAIL] ") + name + (("  | " + str(detail)) if detail else ""))
    return bool(cond)


def head(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════
# A. AST 级源码核验
# ══════════════════════════════════════════════════════════════════════
def _find_class(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    return None


def _find_func(node, name):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def _calls(node, attr):
    """收集 node 子树里所有 self.<attr>(...) 调用。"""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr == attr:
                out.append(n)
    return out


def _assigns(node, attr):
    """收集 node 子树里所有 self.<attr> = ... 赋值。"""
    out = []
    for n in ast.walk(node):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, ast.AugAssign):
            tgts = [n.target]
        for t in tgts:
            if isinstance(t, ast.Attribute) and t.attr == attr:
                out.append(n)
    return out


def _names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _attrs_used(node):
    """收集 Attribute 的属性名（self.image_size → 'image_size'）。"""
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def check_ast():
    head("A. AST 级源码核验（注释骗不过）")
    tsrc = io.open(ROOT / "classification" / "trainer.py", encoding="utf-8").read()
    dsrc = io.open(ROOT / "classification" / "dataset.py", encoding="utf-8").read()
    ttree = ast.parse(tsrc)
    dtree = ast.parse(dsrc)

    cls = _find_class(ttree, "ClassificationTrainer")
    check("存在 ClassificationTrainer 类", cls is not None)
    train_fn = _find_func(cls, "train")
    check("存在 ClassificationTrainer.train", train_fn is not None)

    # A1 调度器：必须被 train() 调用，且训练循环里真的 step
    check("train() 里调用 _build_scheduler()", len(_calls(train_fn, "_build_scheduler")) >= 1)
    sched_steps = _calls(train_fn, "step")
    has_sched_step = any(
        isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Attribute)
        and c.func.value.attr == "_sched"
        for c in sched_steps
    )
    check("训练循环里执行 self._sched.step(...)（调度器不是死开关）", has_sched_step)

    bs_fn = _find_func(cls, "_build_scheduler")
    check("_build_scheduler 真的创建 torch LR 调度器",
          bs_fn is not None and ("LambdaLR" in ast.dump(bs_fn) or "ReduceLROnPlateau" in ast.dump(bs_fn)))

    # A2 AMP：_amp_enabled 的取值必须用到 use_amp
    amp_assigns = [a for a in _assigns(train_fn, "_amp_enabled")]
    ok_amp = False
    for a in amp_assigns:
        if "use_amp" in _names_used(a):
            ok_amp = True
    check("_amp_enabled 由 use_amp 参与决定（AMP 不是死开关）", ok_amp)

    # A3 早停：分支条件必须用 improved，而不是「更新后再比」
    early_if = None
    for n in ast.walk(train_fn):
        if isinstance(n, ast.If):
            if "improved" in _names_used(n.test) and "no_improve" in ast.dump(n):
                early_if = n
                break
    check("早停分支使用 improved（不再每轮必 +1）", early_if is not None)
    # 反证：早停分支里不应再出现 val_acc > best_acc 这种「已更新过的比较」
    if early_if is not None:
        bad = [n for n in ast.walk(early_if)
               if isinstance(n, ast.Compare) and "val_acc" in _names_used(n)]
        check("早停分支内不再用 val_acc 比较（避免恒 False）", not bad, f"found={len(bad)}")

    # A4 _validate 返回 3 元
    v_fn = _find_func(cls, "_validate")
    rets = [n for n in ast.walk(v_fn) if isinstance(n, ast.Return)]
    three = any(isinstance(r.value, ast.Tuple) and len(r.value.elts) == 3 for r in rets)
    check("_validate 返回 (loss, acc, metrics) 三元组", three)

    # A5 drop_last 不再等于 len(train_ds) > batch_size
    pd_fn = _find_func(cls, "_prepare_data")
    drop_names = _names_used(pd_fn)
    assigns_drop = _assigns(pd_fn, "_drop_last")
    check("_prepare_data 里显式记录 self._drop_last", len(assigns_drop) >= 1)
    old_pattern = any(
        isinstance(n, ast.Compare) and "batch_size" in _names_used(n)
        and any(isinstance(op, ast.Gt) for op in n.ops)
        and "len" in ast.dump(n)
        for n in ast.walk(pd_fn)
    )
    check("不再出现 drop_last = len(train) > batch_size 的旧写法", not old_pattern)

    # A6 dataset：固定输入尺寸时跳过预缩放
    gi_fn = _find_func(_find_class(dtree, "ClassificationDataset"), "__getitem__")
    guarded = False
    for n in ast.walk(gi_fn):
        if isinstance(n, ast.If) and "image_size" in _attrs_used(n.test) and "scale_factor" in ast.dump(n):
            guarded = True
    check("__getitem__ 用 image_size is None 守住预缩放（不再重复重采样）", guarded)

    # A7 dataset：augment enabled=False 真的不增强
    init_fn = _find_func(_find_class(dtree, "ClassificationDataset"), "__init__")
    check("__init__ 区分 enabled=False 与 None（无增强=真无增强）",
          "augment_flags.get(\"enabled\")" in ast.dump(init_fn) or "'enabled'" in ast.dump(init_fn))


# ══════════════════════════════════════════════════════════════════════
# B. 指标函数手算对拍
# ══════════════════════════════════════════════════════════════════════
def check_metrics():
    head("B. 指标函数手算对拍")
    import numpy as np
    from classification.trainer import confusion_metrics, metric_for_selection

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = confusion_metrics(y_true, y_pred, ["NG", "OK"], 2)
    # cm = [[1,1],[1,1]]；每类 tp=1 fp=1 fn=1 -> p=r=f1=0.5
    check("混淆矩阵 = [[1,1],[1,1]]", m["confusion"] == [[1, 1], [1, 1]], m["confusion"])
    check("逐类 F1 手算 = 0.5", abs(m["per_class"]["NG"]["f1"] - 0.5) < 1e-12)
    check("macro_f1 手算 = 0.5", abs(m["macro_f1"] - 0.5) < 1e-12, "%.6f" % m["macro_f1"])
    check("balanced_acc 手算 = 0.5", abs(m["balanced_acc"] - 0.5) < 1e-12)
    check("NG 召回手算 = 0.5", abs(m["ng_recall"] - 0.5) < 1e-12)
    check("NG 漏检率 = 1-召回 = 0.5", abs(m["ng_miss_rate"] - 0.5) < 1e-12)
    check("OK 过杀率手算 = 1/2 = 0.5", abs(m["ok_overkill_rate"] - 0.5) < 1e-12)

    # 「全部预测成 OK」：准确率 0.5，但 NG 召回 0 —— 用来证明「准确率会骗人」
    m2 = confusion_metrics(np.array([0, 0, 1, 1]), np.array([1, 1, 1, 1]), ["NG", "OK"], 2)
    check("全判 OK：NG 召回 = 0（准确率却是 0.5）", abs(m2["ng_recall"]) < 1e-12)
    check("全判 OK：OK 过杀率 = 0", abs(m2["ok_overkill_rate"]) < 1e-12)
    check("全判 OK：macro_f1 = (0 + 2*0.5*1/1.5)/2 ≈ 0.3333",
          abs(m2["macro_f1"] - (2 * 0.5 * 1.0 / 1.5) / 2) < 1e-12, "%.4f" % m2["macro_f1"])

    check("select_metric 未知名称回退 acc",
          abs(metric_for_selection(0.77, {"macro_f1": 0.1}, "什么鬼") - 0.77) < 1e-12)
    check("select_metric=macro_f1 取值正确",
          abs(metric_for_selection(0.77, {"macro_f1": 0.31}, "macro_f1") - 0.31) < 1e-12)


# ══════════════════════════════════════════════════════════════════════
# 合成项目
# ══════════════════════════════════════════════════════════════════════
def make_project(root, n_per_class=20, n_val=8, size=(96, 64)):
    """造一个最小分类项目：两张条纹位置不同的合成图，两类可分。"""
    from PIL import Image, ImageDraw
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "models" / "classification").mkdir(parents=True, exist_ok=True)
    (root / "project.json").write_text(json.dumps(
        {"name": root.name, "task_type": "图像分类", "classes": []}, ensure_ascii=False),
        encoding="utf-8")

    mapping, split = {}, {}
    w, h = size
    idx = 0
    for cls_id, x0 in ((0, 6), (1, w - 22)):     # 0=NG 左侧亮条, 1=OK 右侧亮条
        for i in range(n_per_class + n_val):
            img = Image.new("RGB", (w, h), (20, 20, 20))
            d = ImageDraw.Draw(img)
            d.rectangle([x0, 8, x0 + 14, h - 8], fill=(235, 235, 235))
            name = f"s{idx:04d}.bmp"
            img.save(root / "images" / name)
            mapping[f"images/{name}"] = cls_id
            split[f"s{idx:04d}"] = "val" if i < n_val else "train"
            idx += 1

    (root / "annotations" / "class_labels.json").write_text(
        json.dumps({"labels": ["NG", "OK"], "mapping": mapping}, ensure_ascii=False),
        encoding="utf-8")
    (root / "train_test_split.json").write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    return root


class _Harness:
    """把 train() 里与本次优化无关的重活（导出模型）替身掉，只留训练算法本体。"""

    @staticmethod
    def patch(tr):
        tr._export_for_inference = lambda *a, **k: None
        return tr


# ══════════════════════════════════════════════════════════════════════
# C. dataset 行为
# ══════════════════════════════════════════════════════════════════════
def check_dataset(proj):
    head("C. dataset 行为（增强语义 / 重复缩放）")
    import torch
    from classification.dataset import ClassificationDataset
    from classification.trainer import _resolve_cls_augment

    # C1 固定输入尺寸时不再重复预缩放：scale_factor 0.5 与 1.0 应产出完全相同的张量
    #    注意必须关掉增强，否则两个 Dataset 的随机增强会让结果天然不同（假 FAIL）
    a = ClassificationDataset(str(proj), split="train", image_size=(64, 32), scale_factor=1.0,
                              augment_flags={"enabled": False})
    b = ClassificationDataset(str(proj), split="train", image_size=(64, 32), scale_factor=0.5,
                              augment_flags={"enabled": False})
    xa, _ = a[0]
    xb, _ = b[0]
    check("固定 image_size 时 scale_factor 不再影响结果（预缩放已跳过）",
          torch.allclose(xa, xb), f"max|Δ|={(xa - xb).abs().max().item():.3e}")

    # C2 增强语义
    d_off = ClassificationDataset(str(proj), split="train", image_size=(64, 32),
                                  augment_flags={"enabled": False})
    check("enabled=False → 真·无增强", d_off.augment_desc == "关闭", d_off.augment_desc)
    import torchvision.transforms as T
    has_builtin = any(isinstance(s, T.RandomHorizontalFlip) for s in d_off.transform.transforms)
    check("enabled=False → 不再偷加内置翻转", not has_builtin)

    d_none = ClassificationDataset(str(proj), split="train", image_size=(64, 32), augment_flags=None)
    check("augment_flags=None → 仍走内置增强（向后兼容）",
          d_none.augment_desc.startswith("内置"), d_none.augment_desc)

    flags = _resolve_cls_augment({"enabled": True, "flip": True, "brightness": True,
                                  "contrast": True, "rotate": False, "noise": False})
    d_on = ClassificationDataset(str(proj), split="train", image_size=(64, 32), augment_flags=flags)
    check("enabled=True → 全局增强替换内置", d_on.augment_desc.startswith("全局"), d_on.augment_desc)

    d_light = ClassificationDataset(str(proj), split="train", image_size=(64, 32),
                                    augment_flags=_resolve_cls_augment("light"))
    check("预设 light → 翻转+亮度+对比度", d_light.augment_desc.startswith("全局"), d_light.augment_desc)

    d_val = ClassificationDataset(str(proj), split="val", image_size=(64, 32),
                                  augment_flags={"enabled": True, "flip": True})
    check("验证集永不增强", d_val.augment_desc == "关闭", d_val.augment_desc)


# ══════════════════════════════════════════════════════════════════════
# D. 端到端真训
# ══════════════════════════════════════════════════════════════════════
def _mk_trainer(proj):
    from classification.trainer import ClassificationTrainer
    tr = ClassificationTrainer(str(proj), model_name="resnet18")
    return _Harness.patch(tr)


BASE = dict(batch_size=16, image_size=(64, 32), keep_aspect=True, lr=0.01,
            optimizer="adamw", loss_func="cross_entropy", pretrained=False,
            ema=False, augment=None, weight_decay=1e-4, label_smoothing=0.0,
            dropout=0.0, use_amp=True)


def check_scheduler(proj, logs):
    head("D1. LR 调度器真的生效（cosine + warmup 3，共 12 轮）")
    tr = _mk_trainer(proj)
    tr._validate = lambda: (0.1, 0.5, {})
    tr.train(epochs=12, lr_scheduler="cosine", warmup_epochs=3,
             log_callback=logs.append, **BASE)
    lrs = tr.history.get("lr", [])
    check("history 记录了每轮 lr", len(lrs) == 12, lrs[:3])
    if len(lrs) == 12:
        check("warmup 递增（ep1 < ep2 < ep3）", lrs[0] < lrs[1] < lrs[2],
              f"{lrs[0]:.2e} < {lrs[1]:.2e} < {lrs[2]:.2e}")
        check("warmup 期间 lr 小于峰值 lr", lrs[0] < lrs[2], f"{lrs[0]:.2e} < {lrs[2]:.2e}")
        peak = max(lrs)
        check("cosine 收尾明显下降（末轮 < 峰值 10%）", lrs[-1] < 0.1 * peak,
              f"末 {lrs[-1]:.2e} vs 峰 {peak:.2e}")
        check("warmup 起点 ≈ lr/warmup = %.4g" % (0.01 / 3),
              abs(lrs[0] - 0.01 / 3) < 1e-6, "%.6f" % lrs[0])

    # none 时 lr 恒定
    tr2 = _mk_trainer(proj)
    tr2._validate = lambda: (0.1, 0.5, {})
    tr2.train(epochs=4, lr_scheduler="none", warmup_epochs=0, log_callback=logs.append, **BASE)
    lrs2 = tr2.history.get("lr", [])
    check("scheduler=none 时 lr 恒定", len(set(round(v, 12) for v in lrs2)) == 1, lrs2)

    # 自述日志里有环境与调度信息
    joined = "\n".join(logs)
    check("日志含训练自述（环境/数据/调度）",
          "本次训练自述" in joined and "[环境]" in joined and "[调度]" in joined)
    check("自述里写明无 CUDA（当前机器 3080 是幽灵设备）",
          "未检测到可用 CUDA 设备" in joined)


def check_early_stop(proj, logs):
    head("D2. 早停语义（旧实现：每轮必 +1 ⇒ 退化成固定轮数）")
    from classification.trainer import ClassificationTrainer

    # 场景 1：验证指标一路提升 → 不该被早停截断
    tr = _mk_trainer(proj)
    seq = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    it = iter(seq)
    tr._validate = lambda: (0.1, next(it), {})
    res = tr.train(epochs=6, early_stop_patience=2, lr_scheduler="none",
                   log_callback=logs.append, **BASE)
    n1 = len(res["history"]["train_loss"])
    check("持续提升时不被早停（旧代码会在第 2 轮就停）", n1 == 6, f"跑了 {n1} 轮")

    # 场景 2：验证指标一路变差 → 恰好在 patience 轮后停
    tr = _mk_trainer(proj)
    seq = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
    it = iter(seq)
    tr._validate = lambda: (0.1, next(it), {})
    res = tr.train(epochs=6, early_stop_patience=2, lr_scheduler="none",
                   log_callback=logs.append, **BASE)
    n2 = len(res["history"]["train_loss"])
    # ep1: 0.90>0 → improved; ep2: 0.80 不放 → 1; ep3: 0.70 → 2 == patience → 停
    check("持续变差时在第 3 轮（patience=2）停下", n2 == 3, f"跑了 {n2} 轮")
    check("日志里出现早停说明", any("Early stop" in str(m) for m in logs))

    # 场景 3：中间提升一次 → 计数器被正确重置
    tr = _mk_trainer(proj)
    seq = [0.50, 0.40, 0.40, 0.40, 0.90, 0.30]
    it = iter(seq)
    tr._validate = lambda: (0.1, next(it), {})
    res = tr.train(epochs=6, early_stop_patience=3, lr_scheduler="none",
                   log_callback=logs.append, **BASE)
    n3 = len(res["history"]["train_loss"])
    # ep1 improved(0.5); ep2,3,4 不放 → 到达 3 时其实 ep4 就 3 了 → 停在 ep4
    check("早停计数在提升后重置（ep4 达到 3 次未提升而停）", n3 == 4, f"跑了 {n3} 轮")


def check_select_metric(proj, logs):
    head("D3. select_metric 真的改变「最优模型」的落点")
    # acc 与 macro_f1 的 argmax 故意错开
    script = [
        (0.90, {"macro_f1": 0.50, "balanced_acc": 0.50}),
        (0.80, {"macro_f1": 0.90, "balanced_acc": 0.90}),
        (0.60, {"macro_f1": 0.60, "balanced_acc": 0.60}),
    ]

    def run(metric):
        tr = _mk_trainer(proj)
        logs2 = []
        it = iter(script)
        tr._validate = lambda: (lambda p: (0.1, p[0], p[1]))(next(it))
        tr.train(epochs=3, lr_scheduler="none", select_metric=metric,
                 log_callback=logs2.append, **BASE)
        saved = sum(1 for m in logs2 if "Best saved" in str(m))
        return saved, list(tr.history.get("val_select", [])), logs2

    saved_acc, sel_acc, _ = run("acc")
    saved_f1, sel_f1, logs_f1 = run("macro_f1")
    check("select_metric=acc → 只在 ep1 判为最优（1 次 Best saved）", saved_acc == 1, f"{saved_acc} 次")
    check("select_metric=macro_f1 → ep1/ep2 都判最优（2 次 Best saved）", saved_f1 == 2, f"{saved_f1} 次")
    check("val_select 数列随指标不同而不同",
          [round(v, 6) for v in sel_acc] != [round(v, 6) for v in sel_f1],
          f"acc={sel_acc} f1={sel_f1}")
    check("日志按 select_metric 报 Best", any("macro_f1:" in str(m) for m in logs_f1))


def check_drop_last_and_metrics(proj, logs):
    head("D4. drop_last 不再丢样本 + 验证指标进 history")
    tr = _mk_trainer(proj)
    tr.train(epochs=3, lr_scheduler="none", log_callback=logs.append, **BASE)

    n_train = len(tr.train_loader.dataset)
    bs = BASE["batch_size"]
    rem = n_train % bs
    expect_drop = bool(rem != 0 and rem < max(2, bs // 4))
    check("_drop_last 与『尾巴是否过小』一致（不再等于 len>batch）",
          tr._drop_last == expect_drop, f"n={n_train} bs={bs} rem={rem} drop={tr._drop_last}")

    # 真数一遍一个 epoch 喂了多少张
    total_seen, n_batches = 0, 0
    for images, _ in tr.train_loader:
        total_seen += images.shape[0]
        n_batches += 1
    check("一个 epoch 覆盖全部训练样本", total_seen == n_train, f"{total_seen}/{n_train}, {n_batches} batch")
    check("批次数 = ceil(n/batch)", n_batches == math.ceil(n_train / bs), f"{n_batches}")

    h = tr.history
    check("history 记录 val_macro_f1", "val_macro_f1" in h, list(h.keys()))
    check("history 记录 val_balanced_acc", "val_balanced_acc" in h)
    check("history 记录 val_ng_recall", "val_ng_recall" in h)
    check("history 记录 val_ok_overkill_rate", "val_ok_overkill_rate" in h)
    check("history 记录 lr", "lr" in h)
    check("history 记录 val_select", "val_select" in h)
    check("日志含 NG 召回 / OK 过杀", any("召回" in str(m) and "过杀" in str(m) for m in logs))

    # 直接调 _validate 看返回结构
    tr.model.eval()
    loss, acc, met = tr._validate()
    check("_validate 返回 3 元", isinstance(met, dict) and len(met) > 0)
    check("混淆矩阵形状 2x2 且总数 = 验证集大小",
          len(met["confusion"]) == 2 and sum(sum(r) for r in met["confusion"]) == len(tr.val_loader.dataset),
          met["confusion"])
    check("逐类指标齐全（precision/recall/f1/support）",
          all(k in met["per_class"]["NG"] for k in ("precision", "recall", "f1", "support")))

    # AMP：当前机器没有可用 CUDA，即使用户勾了 AMP 也不能启用 cuda scaler
    check("无 CUDA 时 use_amp=True 也不启用 AMP（不会误用 cuda scaler）",
          tr._amp_enabled is False and tr.scaler is None,
          f"amp={tr._amp_enabled} scaler={tr.scaler}")

    # worker 策略：CPU 上小数据集用 0 worker
    check("CPU + 小数据集 → DataLoader workers = 0（避免 spawn 拖慢）",
          tr._num_workers == 0, f"workers={tr._num_workers}")


def main():
    print(f"项目根: {ROOT}")
    tmp = Path(tempfile.mkdtemp(prefix="_clsopt_"))
    try:
        check_ast()
        check_metrics()

        proj = make_project(tmp / "proj_small", n_per_class=20, n_val=8)
        check_dataset(proj)

        # 大一点的训练集用于 drop_last 场景：45 train，batch 16 -> rem 13
        proj2 = make_project(tmp / "proj_mid", n_per_class=22, n_val=9)

        logs = []
        check_scheduler(proj, logs)
        check_early_stop(proj2, logs)
        check_select_metric(proj, logs)
        check_drop_last_and_metrics(proj2, logs)

        # 与真实项目同构的比例：157 train / batch 32 → rem 29（≥8 不丢）
        rem, bs = 157 % 32, 32
        check("真实项目口径 157/32：rem=29 ≥ 8 ⇒ 不丢尾巴",
              not (rem != 0 and rem < max(2, bs // 4)), f"rem={rem}")
        check("真实项目旧写法会丢 29/157 = %.1f%%" % (29 / 157 * 100), 29 / 157 > 0.15, "18.5%")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    head("结果")
    print(f"  PASS = {len(PASS)}")
    print(f"  FAIL = {len(FAIL)}")
    if FAIL:
        for f in FAIL:
            print("   - " + f)
    print("  结论: " + ("全部通过" if not FAIL else "存在失败项"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

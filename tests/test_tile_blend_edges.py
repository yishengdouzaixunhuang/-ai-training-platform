# -*- coding: utf-8 -*-
"""分块推理拼接：图像最外一圈不许被 Hann 的 0 抹掉（审核 2026-11-17 §3.3）。

纯 numpy 复刻拼接过程，不加载模型、不需要 GPU、不需要真实工程。

为什么这条必须存在：
    Hann 窗两端恒为 0，而图像第一列只被第一块覆盖 ⇒ `count_map == 0`
    ⇒ 被 `count_map[count_map == 0] = 1` 兜住 ⇒ `pred_sum / 1` 仍是 0
    ⇒ 输出类别 0（background）。贴边缺陷会被**确定性**漏掉，
    而 `_pred.png` 看上去一切正常 —— 只能靠数值用例钉住。

⚠️ 窗口与步长**必须用产品代码**（`Predictor._blend_window` / `_tile_strides` /
   `_edge_safe_weights`），别在测试里另抄一份公式，否则测的是抄写水平。

★ 两类用例分工（2026-11-17 审核打回后补的）：
   · 上半部分（`simulate`）钉 **helper 的数学**——但它不驱动产品函数，
     把 `self._edge_safe_weights(...)` 从拼接循环里删掉，这些用例照样全绿。
   · 下半部分「调用点」直接驱动 `_predict_tiled_torch` / `_onnx` / `_trt`，
     断言 helper **真的被调用**（次数 = 块数）且边界不再是 0。
     删掉任何一处调用，这里立刻红——这才是"删掉就会红"的那半句。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from PIL import Image

from inference.predictor import Predictor

FILL = 2          # 假装模型认为「整幅图都是类别 2（缺陷）」
TILE = 256
OVERLAP = 128


def simulate(w, h, tile_size=TILE, overlap=OVERLAP, use_edge_fix=True):
    """按 `_predict_tiled_torch` 的累加逻辑复刻：所有块都预测 FILL。"""
    stride_x, stride_y, use_hx, use_hy = Predictor._tile_strides(w, h, tile_size, overlap)
    hann = Predictor._blend_window(tile_size, use_hx, use_hy)

    pred_sum = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, stride_y):
        for x in range(0, w, stride_x):
            x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
            x1 = max(0, x2 - tile_size); y1 = max(0, y2 - tile_size)
            th, tw = y2 - y1, x2 - x1
            wgt = hann[:th, :tw] if (th != tile_size or tw != tile_size) else hann
            if use_edge_fix:
                wgt = Predictor._edge_safe_weights(wgt, x1, y1, x2, y2, w, h)
            pred_sum[y1:y2, x1:x2] += float(FILL) * wgt
            count_map[y1:y2, x1:x2] += wgt

    count_map[count_map == 0] = 1
    return np.round(pred_sum / count_map).astype(np.int64)


def border_pixels(a):
    return np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]])


# ---------------------------------------------------------------- 用例

def test_old_behavior_loses_border():
    """旧行为（不打补丁）：600×600 整圈 0 —— 这就是审核报的那条。"""
    pred = simulate(600, 600, use_edge_fix=False)
    assert border_pixels(pred).max() == 0, "旧行为的边界应全为 0（否则本用例失去意义）"
    assert pred[300, 300] == FILL, "内部应仍是模型预测值"


def test_new_behavior_keeps_border_600():
    """打了补丁：600×600 双轴重叠，最外一圈必须是 FILL，内部也不能变。"""
    pred = simulate(600, 600)
    assert border_pixels(pred).min() == FILL, f"边界最小值 {border_pixels(pred).min()} ≠ {FILL}"
    assert pred.min() == FILL and pred.max() == FILL, "整幅图都应是 FILL，不允许出现 0"


def test_new_behavior_keeps_border_2001x1200():
    """审核实测的第二组尺寸（77 块、双轴重叠）同样不许有 0。"""
    pred = simulate(2001, 1200)
    assert border_pixels(pred).min() == FILL
    assert pred.min() == FILL and pred.max() == FILL


def test_single_tile_axis_unchanged():
    """456×456（不重叠轴用均匀窗）本来就没问题，打补丁后也不许变。"""
    for fix in (True, False):
        pred = simulate(456, 456, use_edge_fix=fix)
        assert pred.min() == FILL and pred.max() == FILL, f"use_edge_fix={fix} 时 456×456 不该有 0"


def test_helper_does_not_mutate_shared_window():
    """`wgt` 常常是共享 `hann` 的切片视图 —— 补丁必须 copy，不能改到后面所有块。"""
    hann = Predictor._blend_window(256)
    before = hann.copy()
    wgt = Predictor._edge_safe_weights(hann, 0, 0, 256, 256, 600, 600)
    assert np.array_equal(hann, before), "共享窗被就地修改了：后续所有块都会跟着变"
    assert hann[0, 0] == 0.0, "原始 Hann 两端就是 0，这里不该被改"
    assert wgt[0, 0] == 1.0 and wgt[0, -1] == 1.0
    assert wgt[100, 100] == hann[100, 100]


def test_helper_leaves_interior_tile_alone():
    """不贴边的块：权重一个数都不许动（否则就是把平滑性改坏了）。"""
    hann = Predictor._blend_window(256)
    wgt = Predictor._edge_safe_weights(hann, 128, 128, 384, 384, 600, 600)
    assert np.array_equal(wgt, hann)


# ---------------------------------------------------------------- 调用点（产品函数）
# 上面只用 simulate 复刻；这里**直接调产品方法**，钉住"谁在调 helper"。

class _ConstantModel(torch.nn.Module):
    """假模型：整幅图都判为类别 1（缺陷）。"""

    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros((b, self.num_classes, h, w), dtype=x.dtype, device=x.device)
        logits[:, 1] = 1.0
        return logits


class _FakeOrtSession:
    """假 ONNX session：`run()` 回一份"全判类别 1"的 logits。"""

    class _Input:
        name = "input"

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feed):
        batch = list(feed.values())[0]
        b, _c, h, w = batch.shape
        logits = np.zeros((b, 2, h, w), dtype=np.float32)
        logits[:, 1] = 1.0
        return [logits]


class _HelperSpy:
    """统计 `_edge_safe_weights` 被调用了几次（= 拼了多少块）。"""

    def __init__(self):
        self.n = 0
        self._orig = Predictor.__dict__["_edge_safe_weights"].__func__

    def __enter__(self):
        orig = self._orig

        def spy(wgt, x1, y1, x2, y2, w, h):
            self.n += 1
            return orig(wgt, x1, y1, x2, y2, w, h)

        Predictor._edge_safe_weights = staticmethod(spy)
        return self

    def __exit__(self, *exc):
        Predictor._edge_safe_weights = staticmethod(self._orig)


def _bare_predictor():
    """不跑 __init__（不加载权重），只装运行 `_predict_tiled_*` 必需的字段。"""
    p = Predictor.__new__(Predictor)
    p.device = "cpu"
    p.model = _ConstantModel()
    p.trt_server = None
    p.ort_session = None
    p.last_prob = None
    p._batch_px_budget = 4096
    return p


def _run_without_helper(p, fn):
    """把实例上的 helper 换成恒等函数，模拟"调用被删掉"。"""
    p._edge_safe_weights = lambda *a: a[0]
    try:
        return fn()
    finally:
        del p._edge_safe_weights


def _assert_border(mask, value, where):
    bp = border_pixels(mask)
    got = bp.min() if value == 1 else bp.max()
    assert got == value, f"{where}: 边界应全为 {value}，实际 min={bp.min()} max={bp.max()}"


def _expected_tiles(w, h, tile_size, overlap):
    """按产品自己的步长规则算块数（不写死 16/25，步长逻辑变了也能跟上）。"""
    stride_x, stride_y, _hx, _hy = Predictor._tile_strides(w, h, tile_size, overlap)
    return len(range(0, w, stride_x)) * len(range(0, h, stride_y))


def test_torch_call_site():
    """torch 路径：helper 的调用次数必须等于块数，边界全为 1。"""
    p = _bare_predictor()
    img = Image.new("RGB", (600, 600), (0, 0, 0))
    expected = _expected_tiles(600, 600, 256, 128)
    with _HelperSpy() as spy:
        pred = p._predict_tiled_torch(img, tile_size=256, overlap=128)
    assert spy.n == expected, f"helper 被调用 {spy.n} 次，块数 {expected} —— 调用点漏了？"
    _assert_border(pred, 1, "torch")
    assert pred.min() == 1

    pred_off = _run_without_helper(p, lambda: p._predict_tiled_torch(img, 256, 128))
    _assert_border(pred_off, 0, "torch(关补丁)")


def test_onnx_call_site():
    """ONNX 路径：块数与 torch 相同，边界全为 1；关补丁后边界全为 0。"""
    p = _bare_predictor()
    p.ort_session = _FakeOrtSession()
    img = Image.new("RGB", (600, 600), (0, 0, 0))
    expected = _expected_tiles(600, 600, 256, 128)
    with _HelperSpy() as spy:
        pred = p._predict_tiled_onnx(img, tile_size=256, overlap=128)
    assert spy.n == expected, f"helper 被调用 {spy.n} 次，块数 {expected}"
    _assert_border(pred, 1, "onnx")
    assert pred.min() == 1

    pred_off = _run_without_helper(p, lambda: p._predict_tiled_onnx(img, 256, 128))
    _assert_border(pred_off, 0, "onnx(关补丁)")


def test_trt_call_site():
    """TRT 路径：步长固定 tile-overlap，边界全为 1；关补丁后全为 0。"""
    p = _bare_predictor()
    p._trt_infer_batch = lambda batch: np.ones((batch.shape[0], 256, 256), dtype=np.int64)
    img = Image.new("RGB", (600, 600), (0, 0, 0))
    stride = 256 - 64
    expected = len(range(0, 600, stride)) ** 2
    with _HelperSpy() as spy:
        pred = p._predict_tiled_trt(img, tile_size=256, overlap=64)
    assert spy.n == expected, f"helper 被调用 {spy.n} 次，块数 {expected}"
    _assert_border(pred, 1, "trt")
    assert pred.min() == 1

    pred_off = _run_without_helper(p, lambda: p._predict_tiled_trt(img, 256, 64))
    _assert_border(pred_off, 0, "trt(关补丁)")


def main():
    tests = [
        ("旧行为：600×600 整圈为 0", test_old_behavior_loses_border),
        ("新行为：600×600 边界恢复", test_new_behavior_keeps_border_600),
        ("新行为：2001×1200 边界恢复", test_new_behavior_keeps_border_2001x1200),
        ("不重叠轴：456×456 不回归", test_single_tile_axis_unchanged),
        ("补丁不污染共享窗", test_helper_does_not_mutate_shared_window),
        ("补丁不动内部块", test_helper_leaves_interior_tile_alone),
        ("调用点·torch：真调 helper + 边界恢复", test_torch_call_site),
        ("调用点·onnx：真调 helper + 边界恢复", test_onnx_call_site),
        ("调用点·trt：真调 helper + 边界恢复", test_trt_call_site),
    ]
    passed = 0
    print("=== 分块拼接边界 ===")
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
    print()
    if passed == len(tests):
        print(f"✅ 全部通过  {passed}/{len(tests)} 通过")
        return 0
    print(f"❌ {len(tests) - passed}/{len(tests)} 项未通过")
    return 1


if __name__ == "__main__":
    sys.exit(main())

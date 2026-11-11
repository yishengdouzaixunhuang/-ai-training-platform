# 改动说明：OCV 批量 NameError（我引入的 P0）+ 未定义名扫描

**日期** 2026-11-11 · **文件** `ui/ocr_ocv_mixin.py`、`ui/main_window.py`、
`tests/test_task_isolation.py`、`tests/test_undefined_names.py`（新增）、
`training/multiview.py`、`classification/trainer.py`、`ui/system_monitor.py`

---

## 一、我引入的 P0：OCV 批量检测直接不可用

上一轮把 `stop_check=self._active_stop` 批量替换成 `stop_check=stop`，
`ocr_ocv_mixin.py` 里**两处**都换了，但只给 `_run_ocr_batch` 补了
`stop = self._infer_stop(stop)` —— `_ocv_batch_inspect` 里的 `stop` 从未定义：

```python
291:  def _ocv_batch_inspect(self):        # ← 没有 stop 参数
...
321:      br = BatchRunner(flow, project_dir=pd, stop_check=stop)   # ← NameError
```

**后果**：点「Batch Inspect」→ `run()` 子线程里抛 `NameError` → 被线程 excepthook 吞掉
→ 状态栏停在 `Batch inspecting...`，没有任何报错。用户以为"卡住了"，重试也一样。

**修法**：补上参数与令牌（与 `_run_ocr_batch` 同形），并且 `stop` 是 keyword-only ——
它被 `clicked` 直接连到按钮上，位置参数会被 Qt 的 `checked` 顶掉。

## 二、为什么 91 项测试上一轮没抓到

上一轮新增的护栏能扫出「**方法名**不存在」（`_start_det_batch_inference` 那个），
但扫不出「**变量名**不存在」—— 同一个错误换了个形态。而 15 条时序用例全在队列层/标志层，
**一条都不走 OCV 批量路径**。

所以补两条，方向不同：

### 1. `tests/test_undefined_names.py` —— 按作用域链扫全项目

用 stdlib `symtable`（不 import 被测代码）：某个名字"被引用、但不是局部/参数/import/闭包
变量"，它就是**全局查找** —— 必须出现在模块级名字或 builtins 里，两边都没有 ⇒ 运行必 NameError。

- 扫 `ui/ core/ training/ inference/ flow/ annotation/ classification/ detection/ ocr/ ocv/ tools/ scripts/ tests/`；
- 跳过 `_attic/ vendor/ wheels/ installers/ dist2/`；`import *` 的模块跳过（名字来源静态看不到）；
- **自带自检**：先喂一段"有未定义名 / 有参数定义"的源码，确认它抓得到且不误报 ——
  否则扫描器写错时会返回空列表，测试永远是绿的。

### 2. `test_ocv_batch_path_wires_stop` —— 真跑 OCV 那条路径

用假 `BatchRunner` 走通 `_ocv_batch_inspect` → 断言拿到的 `stop_check` 可调用、初始为 False、
点「停止推理」后变 True。NameError 会让它超时失败。

## 三、顺带：扫描器抓出的三条**既有** NameError（都不是本轮引入，但都是真炸）

| 位置 | 名字 | 什么时候炸 | 处理 |
|---|---|---|---|
| `training/multiview.py:729` `train_multiview` | `_log` | **用户点「停止」那一刻**（那条日志就在停止路径上） | 改成 `log`（本作用域的日志函数） |
| `classification/trainer.py:1300` `_train_kfold` | `scale_factor` | `k_folds > 1` 训练到"存折 checkpoint" | 改用 `self._scale_factor`（本文件别处也这么取） |
| `ui/system_monitor.py:205` `_pdh_read` | `ctypes` | 核显读共享显存 | 函数内 `import ctypes` |

第三条最隐蔽：它被自己的 `try/except: return {}` 吞掉，表现是"核显机器上显存永远读不到"，
不报错、不崩溃。

第一条和本轮的 P0-3 直接相关 —— **三视图训练点"停止"会 NameError**，
而那天我在改的正好就是停止路径。

## 四、P2：令牌容器的瘦身是摆设（审核指出）

```python
if len(self._infer_stops) > 32:
    self._infer_stops = {ev for ev in self._infer_stops if not ev.is_set()}   # 旧
```

正常跑完的推理**永远不会 set 令牌**，这个过滤清不掉任何东西；长度一直涨，
而且超过 32 之后每次调用都白重建一遍 set。改成按**添加顺序**丢最老的：

```python
self._infer_stops = deque(maxlen=32)    # __init__
self._infer_stops.append(ev)            # _infer_stop，满了自动淘汰
```

同时最多 5 条推理在跑，32 的窗口足够；`test_infer_token_store_is_bounded` 钉住
（连造 40 个令牌 → 长度 ≤ 32、最新那个仍是干净的）。

## 五、验证

| 项 | 结果 |
|---|---|
| `test_undefined_names` | 2 条（扫描器自检 + 全项目 0 处） |
| `test_task_isolation` | 13 → **15**（新增 OCV 路径、令牌上限） |
| 全套 | **12 个脚本 91 项**全绿 |

**变异检验**（把缺陷形状塞回去，改完还原）：

| 塞回的缺陷 | 目标用例 | 结果 |
|---|---|---|
| `_ocv_batch_inspect` 去掉参数与令牌（复现我引入的 P0） | 全项目无未定义名字 | **变红**，报 `ui\ocr_ocv_mixin.py → run → stop`（与审核报的那行一致） |
| 同上 | OCV 批量路径真的能跑 | **变红**（NameError 让线程死掉 → 超时） |
| 只删令牌行、保留参数（`stop_check=None`） | OCV 批量路径真的能跑 | **变红**（"没拿到可调用的 stop_check"） |
| `deque(maxlen=32)` 改回 `deque()` | 令牌容器有上限 | **变红** |
| 三视图 `log(` 改回 `_log(` | 全项目无未定义名字 | **变红** |

## 六、请重点核对

1. **`_ocv_batch_inspect` 的 `stop` 是 keyword-only** —— 它直接连在
   `QPushButton("Batch Inspect", clicked=...)` 上，普通参数会被 `checked` 顶掉。
2. **扫描器的边界**：`import *` 的模块整体跳过；模块级动态注入（`globals()["x"]=...`）
   会被误报，真遇到请加进 `ALLOW` 并写清理由 —— 别拿它当消音器。
3. **三条既有 NameError 的处理方式**都是"就地补齐正确名字"，没有改行为；
   但 `_train_kfold` 那条改的是写进 checkpoint 的 `scale_factor` 取值来源
   （`self._scale_factor`，与同文件 1333 行一致），如果 K 折需要按 fold 不同的缩放，
   这里要再想 —— 目前 `train()` 里对每个 fold 用的是同一个值，与 1333 行同源。

## 七、没做的部分

- 扫描器只查"名字不存在"。**方法名拼错**靠上一轮 `test_dispatch_targets_exist` 兜；
  属性名拼错（`self.xxx`）目前没人管 —— 那是另一类静态检查（需要类型信息），
  本轮不做。
- 重构清单 P0-2（`core/settings.py`）尚未开始。

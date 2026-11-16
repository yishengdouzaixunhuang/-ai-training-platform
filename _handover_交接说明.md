# 交接说明（AI 训练平台 · 2026-10-21）

> 这份文档是给**接手继续开发的那台机器上的 AI** 看的。
> 读完这一页，你就能立刻上手、并且不会重复踩这半个月踩过的坑。

---

## 1. 这是什么项目

工业视觉**缺陷检测 AI 训练平台**（PyQt5 桌面软件），做电池极片缺陷的
标注 → 训练 → 推理 → 评估闭环。现场用：孚能/新松等客户的**三视图工程**
（漫反射 te_0 + 形状图1 te_1 + 形状图2 te_2，早融合 9 通道）。

- 源码根目录：本包解压后的 `ai_training_platform/`
- **真实工程数据不在仓库里**（在 `E:\05-TrainingProject\`），本包只有代码
- 真实训练解释器：`C:\aitp_venv\Scripts\python.exe`
  （仓库里的 `.venv310` / `.venv312` 是**坏的**，已从包里排除，**不要用**）
  ⚠️ 这个路径**必须是纯英文**，装到项目目录里 Qt 会卡死 —— 见地雷 31、
  以及第 8 节「换电脑 SOP」

## 2. 怎么跑

```bash
# 启动平台
"C:/aitp_venv/Scripts/python.exe" main.py

# 自测（四个都是只读/临时目录，不会碰用户工程）
"C:/aitp_venv/Scripts/python.exe" selftest_marks.py       # 标记体系 175 条
"C:/aitp_venv/Scripts/python.exe" selftest_taskqueue.py  # 任务队列 21 条
"C:/aitp_venv/Scripts/python.exe" selftest_inchannels.py  # 9 通道训练链路 15 条
"C:/aitp_venv/Scripts/python.exe" probe_mark_ui.py        # 主 UI 探针 271 条（真窗口）
"C:/aitp_venv/Scripts/python.exe" scripts/verify_sweep_metrics.py --render   # 主题清洗 16 条
"C:/aitp_venv/Scripts/python.exe" _probe_dialogs.py       # 数据集划分 / 类别管理 对话框 60 条
"C:/aitp_venv/Scripts/python.exe" _probe_split_semantics.py # 未划分 = 空集 的取数语义 24 条
"C:/aitp_venv/Scripts/python.exe" -u _probe_ann_table.py  # 标注统计表按「块」出行 13 条
"C:/aitp_venv/Scripts/python.exe" -u _probe_switch_perf.py # 真实大图切图耗时拆段（只读）
"C:/aitp_venv/Scripts/python.exe" _probe_startup_flash.py # 启动闪烁：顶层杂散窗口必须 0 个
"C:/aitp_venv/Scripts/python.exe" -u _probe_menu_entry.py # 两个菜单入口能打开（原来必崩）
"C:/aitp_venv/Scripts/python.exe" _probe_dialogs_shot.py  # 把两个对话框渲染成 PNG 看版面
"C:/aitp_venv/Scripts/python.exe" tests/test_counts_cache.py  # 类别统计缓存 16 条（该失效时必须失效）
"C:/aitp_venv/Scripts/python.exe" tests/test_task_isolation.py  # 任务级停止标志 15 条（两个任务同时在队列里 + OCV 路径）
"C:/aitp_venv/Scripts/python.exe" tests/test_undefined_names.py  # 未定义名字扫描（改完批量替换必跑）
"C:/aitp_venv/Scripts/python.exe" tests/test_project_switch_view.py  # 切换工程后视图同步 10 条（含哨兵值）
"C:/aitp_venv/Scripts/python.exe" tests/test_duplicate_defs.py      # 同名定义扫描 8 条（根目录+模块级+类内，搬方法/改名前必跑）
"C:/aitp_venv/Scripts/python.exe" tests/test_settings.py          # 配置统一入口 28 条（面板+对话框+增强设置三个写入者）
"C:/aitp_venv/Scripts/python.exe" tests/test_config_files.py      # 配置写入者 8 条（含"坏配置下平台仍能启动"）
"C:/aitp_venv/Scripts/python.exe" tests/test_result.py           # 统一返回与错误码 11 条（含真实模块已用上）
"C:/aitp_venv/Scripts/python.exe" -u _probe_counts_ui.py     # 真实窗口 A→B→切回，表格文字必须一致
```

`probe_mark_ui.py` 还支持 `--real-project "<工程名>"`（跑真实工程）与 `--theme light`。
它的结果会落盘 `_ui_probe_shots/probe_last_{ok,FAIL}.log` —— **间歇性失败时看这个文件**。

> 换机器 / 新机器首次部署，看第 8 节的 SOP，别从头 pip install。

## 3. 目录速查

| 目录 | 干什么 |
|---|---|
| `main.py` | 入口。**内含全局基础 QSS**（几百行深色主题表），用 AST 可提取 |
| `ui/` | 全部界面：`main_window.py`（主窗，310KB 单文件，最大）、`annotation_toolbar.py`、`task_queue.py`/`task_queue_panel.py`、`mark_panel.py`、`themes.py`（主题表）、`split_dialog.py`（数据集划分）、`class_dialog.py`（类别管理）… |
| `core/` | 领域逻辑：`marks.py`（四态标记）、`filter_expr.py`（筛选表达式）、`project_manager.py`、`class_discovery.py` |
| `training/` | `trainer.py`（单视图训练）、`multiview.py`（**三图 9 通道训练**）、`dataset.py`、`models.py` |
| `inference/` | `predictor.py`（推理，含多视图推理） |
| `annotation/` | 标注画布、labelme 导入导出、掩膜编辑 |
| `flow/` | 推理流程节点（批量推理走这里） |
| `classification/` `detection/` `ocr/` `ocv/` | 分类/检测/OCR/OCV 任务类型 |
| `tools/` `scripts/` | 命令行工具、渲染/核验脚本 |

## 4. 2026-10-15 ~ 10-16 这一批改了什么

**UI 侧**（全部对标友商 iCogtiveFlow 截图）
- 左栏由"上下叠放"改成**竖排页签三页**：工程管理 / 图像管理 / **任务队列**
- 左右栏折叠：折叠 = 面板整个隐藏（画布占满整宽），恢复按钮是**贴在画布边缘的浮层**（不占布局宽度）
- 标注/检测工具栏拆成「常驻条 + 可折叠工具行」（默认折叠，Ctrl+B），常驻条含上一张/下一张/保存
- 图层开关由方块按钮改成「显示 + 真复选框」（勾上=实心青块）
- 统计表列宽按 max(表头, 内容) 定，「平均灰度」列三视图并列显示（`86.2，188.2，68.3`）
- 底栏资源占用：CPU 分整机/进程、磁盘分「空间/IO」两个口径、GPU 读不到就明写"不可读"
- **新模块「任务队列」**：训练+推理统一登记，串行执行，显示进度/剩余时间，每行可取消

**训练侧**
- `create_model` 支持 `in_channels`（3/6/9），首层按 I3D 式膨胀（复制 RGB 预训练核 × 3/N）
- `SegmentationDataset` 支持多视图（**按组取样**，一个场景 3 视图 = 1 个样本）
- `trainer` 识别 `project.json` 的 `views` → 推 `in_channels`，并写进 checkpoint
- ⇒ **三图工程在 GUI 里也能训了**（此前静默只训第一个视图）

## 5. 已知地雷（★ 都是这半个月实际踩过的，别再踩）

1. **配色一律写进 `ui/themes.py` 的主题表**。控件上的内联白名单色会被
   `sweep_inline_styles` 剥掉（只留度量）—— 底栏滑块曾因此"占着 84px 一个像素不画"。
   新增任何自绘/文字型控件，都要问"主题表里有它吗"。
2. **`torch` 必须先于 `PyQt5` 导入**，否则 `OSError [WinError 1114] c10.dll`。
3. **`QT_QPA_PLATFORM=offscreen` 在 Windows 上没有字体引擎**，文字会全空白 ——
   要看文字必须 `windows`；`offscreen` 只适合量几何。
4. **判"控件画出来了没有"要数像素**，别只看 value/geometry/信号（都会骗你）。
   数像素前先问"分母是背景还是透明" —— `QCheckBox.grab()` 里占比最大的是**透明像素**。
5. **`isVisible()` vs `isHidden()`**：前者要求**所有祖先可见**。整页不可见时用
   `isVisible()` 判"按钮可点"会得到假失败 ⇒ 用 `isHidden()`。
6. **跨线程投递只能用 Qt 信号**（`pyqtSignal(object)`）。在工作线程里
   `QTimer.start()` / `invokeMethod(字符串)` 都不可靠（后者只对 slot 有效）。
   队列回调是 `on_change(tasks)` 带参 ⇒ 信号必须声明带参，否则 TypeError 会被
   队列里的 `except Exception: pass` **静默吞掉**。
7. **几何断言要量"谁变形了"**，量"两个控件之间的距离"会被任一方撑大掩盖（假绿实录）。
8. **批量改文本前，先把"不可动的行"用 `git show HEAD:<file>` 按行号兜底** ——
   占位符保护法实测失效（把本机路径改成了 `01_维护者`）。
9. **改完脚本必比对「脚本 mtime vs 输出 mtime」**（脚本尾部崩了、输出还是旧文件）。
10. **dataclass 的字段必须带注解**：`run=None` 没有注解就不是字段 ⇒
    `Task(run=...)` 直接报 unexpected keyword argument。
11. **长跑的 Qt 脚本：输出重定向到文件**（`TerminateProcess` 不 flush stdout，屏幕上看不到），
    收尾用 `TerminateProcess`（`sys.exit`/`os._exit` 会卡在 DLL_PROCESS_DETACH）。
12. **行/控件删除要 `removeWidget` + `setParent(None)`**，只 `deleteLater` 会留下
    "已删但仍在布局里"的幽灵行。
13. **「未划分」= 不属于任何集（空集）**，别再改回"默认 train"（2026-10-22 按用户要求
    统一）。以前 `get_train_test_split()` 在内存里补默认 train、`multiview.py` 用
    `split_map.get(key,"train")`、分类侧三处同样 —— 后果是"没划分"= 偷偷进训练集，
    而界面显示成空集，**用户核对标注时的判断依据是假的**。
    现在的口径：`training/dataset.py`、`training/multiview.py`、`classification/dataset.py`、
    `classification/dual_dataset.py`、`tools/infer_cell_farasis.py` 一律"查不到就取不到"，
    训练集为空**当场报错**并指路「工具 → 数据集划分管理…」。
    ⚠️ 连带坑：`annotation/convert.add_to_train()` 以前靠这个默认值才写得进盘，
    现在它必须**显式** `split_map[b] = "train"`；改回去就是"数着改了 N 条、盘上一字未动"。
    ⚠️ 分类工程 ImageFolder 布局的划分键是 `类别/文件名`（`ok/a`），不是裸文件名 ——
    `ui/split_dialog.py` 已按 `images/` 下的相对路径写键，别再改成 basename。
14. **老工程升级会停在"训练集为空"**：用划分对话框里的「未划分 → 训练集」一键补齐
    （它只动未划分的，已划好的 val/test 不碰）。
15. **右栏 Annotation(n) 表是「一个标注块一行」**，标题里的 n 直接取表格行数。    原来的实现是 `for cid in np.unique(mask)` = **一个类别一行**（同类的 40 块面积被求和成一行），
    而「已标注 40」数的是 JSON shape 数 ⇒ 两个数字对不上，用户以为丢标注。
    行数据优先取 labelme JSON 的 shape 列表（与列表「标注」列同源，永远相等），
    无 JSON 时退回"一个连通域一行"。
    ⚠️ 别用 `scipy.ndimage.find_objects`：`C:\aitp_venv` 里**没装 scipy**。
    ⚠️ 也别逐块 `np.nonzero(idmap==cid)`：那是每块扫一遍全图，6MP×40 块实测 0.45s；
    现在用"一次 flatnonzero + 按编号分组"拿到所有 bbox，稳态 0.013s。
16. **大图切图的两处白解码（2026-10-22 实测 1.62s → 0.62s）**
    ① `_gray_text_for_region()`：统计表的 region 是**显示尺寸**的，而视图是原图尺寸，
       旧代码"先 `_load_gray_cached()` 解码 38MB 图 → 再发现形状不匹配 → 丢掉"，
       每切一张大图白扔 0.6s。现在先 `Image.open(p).size` 读文件头比尺寸，对上了才解码。
    ② `_load_prediction_overlay()`：`_overlay.jpg` 是**全尺寸**位图（4096×9320，解码
       成 RGB 114MB，0.29s/张），但 `paintEvent` 优先走 `_build_prediction_overlay()`
       （用 `_pred.png` 按显示尺寸现合成，且带缓存）⇒ 有 `_pred.png` 时这张 JPG
       **永远不会被画**。现在只在"没有 _pred.png 的老工程"才解码它。
       （顺带修好一个旧 bug：以前 `_prediction_mask` 的读取写在 `if overlay.jpg 存在`
       里面 ⇒ 只有 pred.png 没有 overlay.jpg 的工程，Result 表永远是空的。）
    剩下的 0.6s ≈ Result 表 97 个连通域 0.35s + 显示图降采样 0.10s + 解码 0.04s，
    是真实计算量；要再压得把它改成"面板可见时才算"或异步算，属于设计改动。
17. **`"—"` 是 truthy —— 别用 `if not text.strip()` 判"没算出来"**
    真实反馈（2026-10-22）：20MP 图上 Result 表的「平均灰度」**整列显示 —**。
    链条是：大图显示时 `_cache_qimage` 会降采样到 ≤2048 ⇒ 统计表里的 region 是
    **显示尺寸**的，而 `_gray_text_for_region()` 拿原图尺寸的灰度去套 ⇒ 尺寸不匹配 ⇒
    按设计返回 `"—"`。调用处写的是 `if not gray_text.strip(): 用 avg_gray 兜底`，
    而 `"—"` 非空 ⇒ 兜底永不触发 ⇒ 明明算好的 avg_gray 被丢掉。
    现在统一走 `_gray_cell_text(region, avg_gray)`（两张表共用，判空把 `"—"` 也算进去）。
    ⚠️ 另外：Annotation 表原来的兜底会去**解码原图全尺寸灰度**（38MB 一张 0.6s），
    已改成 `_gray_from_display()` 用 `canvas._cached_gray`（显示图灰度）按比例裁切算 ——
    同一份数据、同一套口径，零额外解码。
    ⚠️ 已知取舍：**大图 + 多视图**时平均灰度只显示 1 个值（其余视图没有显示尺寸灰度缓存），
    小图仍是三视图并列。要补就给每个视图加一份"显示尺寸灰度"缓存（每视图约 +50ms）。
18. **勾选框的 ✓ 是图片，不是 QSS 画出来的**（2026-10-22 用户要求"把蓝色小方块换成√"）：
    QSS 的 `::indicator` 只有 background / border / image 三种手段，**画不了文字或图形**。
    所以做了 `resources/check_dark.svg`（青底+近黑勾）和 `resources/check_light.svg`（蓝底+白勾），
    在 `ui/themes.py` 里用 `image: url(...)` 挂上去（`_res_uri()` 负责把反斜杠换成 `/`，
    Windows 路径直接写进 url() 会解析失败）。
    ⚠️ `DARK_CHECK` / `LIGHT_CHECK` 必须拼在 `DARK_INDICATOR` / `LIGHT_OVERLAY` **之后** ——
       前面那句 `image: none` 靠"同优先级后者胜"才被覆盖。
    ⚠️ 这两段是**应用级**样式表，`sweep_inline_styles` 只扫控件级内联样式，不会剥它们。
    ⚠️ 若将来打成 onefile 包，要把 `resources/*.svg` 加进 datas，否则路径失效、勾变回方块。
19. **画布"糊"= 把降采样缓存图放大**（2026-10-22 用户拿友商截图对比）
    链条：`_cache_qimage` 为切图速度把显示图限制在**长边 2048**；456×7041 的图缓存成
    132×2048（0.291 倍）。而注释里承诺的"放大到更高倍时改用原图裁剪（见 `_crop_qimage_at`）"
    —— **那个方法全文件根本不存在**（`grep -c "def _crop_qimage_at"` = 0）。
    于是用户按 100% 看图，实际是把 0.29 倍的图**放大 3.45 倍**，必然糊。
    ① 补上 `_use_crop_path()` / `_crop_for_view()` / `_draw_base_image()`：
       缩放比例 > 缓存比例×1.02 时，从 `self.image` 按**可见区域**裁切（+96px 余量、单条缓存）。
    ② **叠加层也必须一起改**：`_build_overlay` 把原图**烘进**了 overlay（显示尺寸），
       底图再清晰也会被它盖糊 —— 只改底图时实测像素误差一点没变（22.48 → 23.76）。
       现在同时缓存一份**纯掩膜 RGBA 层**（`_cached_overlay_rgba_qimage`），高倍时只叠它。
    ⚠️ 底层图与叠加层**必须用同一个判据** `_use_crop_path()`，否则就是"清晰底图 + 糊盖层"。
    ⚠️ 排查时注意：`resizeEvent` 会把 zoom 拉回 fit，写探针要在绘制前把 `_fit_to_window`
       屏蔽掉，否则你以为在 100% 测、其实一直在 fit 下测（这次白绕了两轮）。
20. **`self.trainer` 是"单槽位"共享属性，训练线程必须用局部快照**
    2026-10-22 用户实拍：一次训练跑到 epoch 206 时又点了一次「开始训练」，
    再点「停止」→ 报 `'NoneType' object has no attribute 'train'`（main_window 的 run 里）。
    链条：第二次 `_start_training` 把 `self.trainer` 换成新实例 → 第一个任务的 finally
    执行 `_cleanup_trainer()` → `self.trainer = None`（清掉的是**第二个**）→
    第二个任务线程走到 `self.trainer.train()` 时是 None。
    三处修：① 线程里用 `_trainer_ref` 局部快照；② `_cleanup_trainer(who)` **只清自己那个**
    （`if self.trainer is target` 才置空，别人的不动）；③ `_start_training` 开头若队列里
    已有 TRAIN 任务在排队/执行，直接提示并 return，不再允许开第二个。
    ⚠️ 训练成果不受影响：`best_model.pth` 每次 val 提升就落盘，`last_model.pth` 停机时落盘，
    报错发生在"第二个任务启动"那一刻，此时第一个任务的东西早就存好了。
21. **推理的滑窗块尺寸必须等于训练裁剪尺寸，且大图不许"整图缩放"**（2026-10-22）
    训练这次用的是 `image_size=256` + 原分辨率切块；而 `predict()` 里块尺寸**写死 200**，
    还把长边 >1500 的图**先缩到 1024**（456×7041 → 66×1024，横向压 7 倍）。
    实测同一条产线图：旧口径**检出 0 块**，新口径 42 块（GT 40，IoU 0.956）。
    三处改：
    ① `trainer.save_model` 写入 `image_size`（老模型没这字段）；
    ② `Predictor._resolve_train_tile()`：ckpt → 工程/平台 `.train_settings.json` → 200；
    ③ `predict()`：`_要分块 = tiled or 大图`；精细档 + 分块 + 估算块数 ≤600 时走
       **原分辨率滑窗**（不再缩整图），块数太多才退回 1024。
    ⚠️ 「快速」档仍是整图缩放（对标友商 100ms 档），对细长条大图会 0 检出 ——
       这是设计取舍，不是 bug；精细档才是默认正确路线。
22. **滑窗推理返回 `int64` 掩膜，喂给 PIL 会当场炸**
    `_predict_tiled*` 全部 `return np.round(...).astype(np.int64)`；而
    `mask_editor._build_prediction_overlay()` 会 `Image.fromarray(mask)` ⇒
    `TypeError: Cannot handle this data type: (1, 1), <i8>`，而且是在 **paintEvent** 里抛，
    直接被全局 excepthook 兜成"程序遇到未处理异常"弹窗（不是 "Inference failed"）。
    以前大图走整图前向（uint8）所以从没暴露；第 21 条把大图改成强制分块后就必踩。
    修：① `predict()` 返回前统一 `astype(uint8)`；② `_build_prediction_overlay` 里
    `Image.fromarray` 之前兜底强转 uint8（双保险，别再让 dtype 泄漏到绘制层）。
    ⚠️ 掩膜像素值就是类别 id ⇒ 类别数必须 ≤255，uint8 是硬上限。
23. **批量推理不保存置信度图 ⇒ Result 表「分数」整列 0.0%**（2026-10-22 用户实拍）
    「分数」优先取 `prob_map` 里该块的均值；而 `canvas._pred_prob` **只有单张推理才赋值**，
    批量走 flow 只落 mask/overlay ⇒ 退化成"与标注 IoU"，没标注的图上就是 0.0%。
    改：① `SegResultWriter` 多存一张 `<base>_prob.png`（uint8 = 概率×255，
    20MP 也才几百 KB，新参数 `save_prob` 默认开）；
    ② `_update_stats` 取置信度图的顺序：`canvas._pred_prob` → `_load_prob_for_current()`
    读 `outputs/<base>_prob.png`（按显示尺寸缩放）。
    ⚠️ 老结果没有 `_prob.png`，必须**重跑一次批量**分数才会出来。
    ③ 列表「分数」列同步改成**平均置信度**（前景像素上的概率均值，百分比格式），
    与 Result 表同口径；没有 `_prob.png` 就**留空**，不再拿 IoU 冒充（那是"和标注比"，
    没标注的图恒为 0，摆在这一列会被读成"模型没把握"）。

## 6. 这个包里**没有**什么

| 没带 | 原因 / 怎么办 |
|---|---|
| `.venv310/` `.venv312/` | 坏的虚拟环境（1.66GB）。用 `C:\aitp_venv` |
| `models/sam_vit_b_01ec64.pth`（358MB） | SAM 预训练权重。需要的话从原机器拷，或让程序自动下载 |
| `yolo26n.pt`（5MB） | 同上（YOLO 预训练权重） |
| `tmp/` `_ui_render/` `_ui_probe_shots/` | 探针与出图的临时产物，可重新生成 |
| `E:\05-TrainingProject\`（真实工程数据） | **在原机器上**，不在仓库里 |

## 7. 接手前先看这两件事

1. `git status` —— **工作区里有未提交的改动**（10-15/10-16 的 UI 重构 + 训练侧优化，
   其中 `classification/*`、`ui/infer_params.py` 等是别人在途的训练侧工作）。
   本包是**按工作区当前状态**打的包（含未提交内容），`.git` 历史也在。
2. 上面第 5 节的 12 条地雷 —— 那是这个项目"看起来能跑、其实悄悄错"的主要来源。

24. **推理提速两刀（2026-10-22，对标友商 ~120ms/张）**
    ① **窄条图横向不重叠**：`Predictor._tile_strides()`。456 宽的图用 256 块、stride 128 时
       横向切 4 列，同一段像素前向算了 2 遍。判定 `w <= tile*2` 时该轴 stride = tile。
       ⚠️ **不重叠的轴窗口必须换成 1.0**（`np.ones`）—— Hann 两端是 0，
       零重叠直接拼会在缝上得到 count_map≈0，除出噪声/黑线。
       实测 predict 0.71s → **0.194s**（warm），检出 42 不变，IoU 0.9560 → 0.9558。
    ② **连通域属性按 bbox 裁切算**：`inference/params.component_props()`。
       原来每个块都 `lab == i` + findContours + `gray[region]` 扫**整幅** 20MP，
       41 个块要 0.227s（比模型推理还贵）。改成只在 bbox 子图内算，→ 约 0.02s。
       结果不变（面积/宽高仍取 stats bbox 口径，角度/灰度/分数只看块内像素）。
       整条批量流程 0.68s → **0.46s/张**。
    ⚠️ 剩余构成：模型前向 ~0.22s、overlay 合成 0.08s、落盘 0.045s。
       **TRT 只能优化那 0.22s**，想压到 120ms 还得砍 overlay 落盘/合成。
    ⚠️ 本机 `C://aitp_venv` **没有** tensorrt / onnx / onnxruntime，也没有 py3.12 TRT venv
       —— 上 TRT 前必须先装包并验证 CUDA 版本匹配。

25. **缩放锚点公式漏了"居中项"，图像会往左上漂**（2026-10-22 用户实拍）
    屏幕坐标 = `(W - dw)/2 + pan + zoom*u`，而在**光标不动**的前提下解出的正确更新是
        `pan1 = (px - W/2) - f * ((px - W/2) - pan0)`
    原实现（`wheelEvent` 与 `_zoom_step` 两处）写的是 `pan1 = px - f*(px - pan0)`，
    **少了 `W/2` 那一项** ⇒ 每次缩放都往 pan 里多塞一个 `(W/2)(1-f)`：
    放大时 f>1 ⇒ 负 ⇒ 图像一步步往**左上**漂；缩小则往右下漂。
    光标恰在正中央时该漂移最纯粹（中央本该完全不动），所以最容易被发现。
    `_zoom_step`（按钮/快捷键/滑块）以视图中心为锚点时，正确结果就是 `pan *= f`。
    实测修复后：滚轮连放大 10 次中心像素偏移 **1.00px**（旧实现约 900px 量级），
    按钮路径 **0.00px**，且缩回原倍率偏移 0.00px（可逆）。
    ⚠️ `ui/multiview_preview.py` 的参考窗用的是另一套公式（`(iw/2 - cx) * nz`），
       本次**未动**；若参考窗也出现类似漂移，照这条推导重算。

26. **批量不再合成/落 overlay JPG**（2026-10-22，用户选的"方案 C"）
    批量每张要在 20MP 上合成一次 RGB 叠加图（实测 0.084s）再编码成 JPG，
    而画布根本不用它 —— `_build_prediction_overlay()` 是按 `_pred.png` 在**显示尺寸**现合成的。
    改：`SegInfer` 新增参数 `return_overlay`（默认 False，不调 `_create_overlay`）；
        `SegResultWriter` 的 `save_overlay` 默认改 False。
    实测整条批量流程 **0.46s → 0.36s/张**，产物只剩 `_pred.png` + `_prob.png`，块数 41 不变。
    ⚠️ 只有"没有 `_pred.png` 的老工程"才真的需要 overlay.jpg（`_load_prediction_overlay`
       拿它当兜底位图）。那种工程要手动把 `save_overlay` 打开。
    ⚠️ 已有 outputs 里历史遗留的 `_overlay.jpg` 不用管，不影响任何计算。

27. **ONNX Runtime 提速试过，结论是"不用"**（2026-10-22，方案 A 的实测）
    背景：对标友商 ~120ms/张，想用 ORT CUDA 替掉 PyTorch 前向。
    过程与坑：
    ① `pip install onnxruntime-gpu` 默认装到 **1.31**，它要求 **CUDA 13.x**，而本机 torch 是
       cu128（CUDA 12.8）⇒ CUDAExecutionProvider 加载失败、静默退回 CPU，
       单块 16.7ms（PyTorch 1.8ms），**慢 9 倍**。
    ② 降到 `onnxruntime-gpu==1.22.0`（CUDA 12 + cuDNN 9）并 `os.add_dll_directory(torch/lib)`
       （复用 torch 自带的 cublasLt64_12.dll / cudnn64_9.dll）后 CUDA EP 才真正加载。
    ③ 但**端到端反而更慢**：ONNX 0.309s vs PyTorch 0.215s（同一张图，3 次取最快），
       掩膜只差 6 个像素（数值等价）。
    原因：这个模型太小（MultiViewUNet 1.96M，单块前向 0.5~1.2ms），
    ORT 的 per-run 开销 + 固定 batch=16 + 每块现算 numpy 窗，把收益吃光了。
    ⇒ 已 `pip uninstall onnxruntime-gpu onnx` 并删除 `models/model.onnx`，环境复原。
    ⚠️ 以后想再试 ORT，直接测**整图端到端**，别只看单块前向——单块数字会骗人。
    ⚠️ 真的要再快，方向是**更小的网络**（友商 "SGTiny" 那种），不是换运行时。

28. **训练时 val mIoU 的两个指标 bug（2026-10-22，用户问 IGNORE 语义时顺手挖出）**
    ① **ignore 像素没从 IoU 分母剔除**：`unions[c] += (pred_c | mask_c)` 里 pred_c 含
       "落在 ignore 区的预测"，而 mask_c 在那恒 0 ⇒ 分子不进、分母照加，IoU 被系统性压低。
       （训练损失侧一直是对的：CE 用 `ignore_index=255`，Lovasz 用 `_flatten_probas` 剔除。）
       修：先算 `valid = masks != 255`，`pred_c/mask_c` 都 `& valid`。
    ② **图上没出现的类别被按 0 计入平均**：`ious.mean()` 把 Zw/Hh 的 0 也算进去。
       本工程 val 图只有 bg+Hh ⇒ mIoU 天花板被人为压到 0.5。
       修：只在**真值里出现过**的类上平均（`gt_pixels > 0`）。
       ⚠️ 判据别写成 `unions > 0` —— 模型在 GT 没有的类上乱预测时 union 也会 >0，
          那类会被算进来（我先写错一次，实测才发现）。
    修完后同一模型的真实指标：bg 0.645 / Hh **0.006** / mIoU 0.3255
    —— 数字终于可信，但结论没变：Hh 基本没学会。
    ⚠️ 还有个坑：`prepare_data()` 是 `rng.shuffle` 后取 20% 当 val ⇒ **随机 1 张图**。
       抽到哪张，指标就取决于那张图里有哪些类，跨轮次不可比。要可比必须显式划分。

29. **平均灰度精度自校验（合成数据集 + 探针）**
    `_gen_gray_testset.py` 生成 28 张 512×512：背景固定 255，前景 200×200 的平均灰度
    从 0 覆盖到 255（含 127.5 / 63.5 这类半像素值）；前景用「固定种子随机置换取前 n 个」
    抖动（base/base+1），40000 像素 ⇒ 分辨率 1/40000，任意两位小数真值都精确成立。
    真值表在 `manifest.csv`，工程可直接被平台打开（images + labelme json + annotations/_mask.png）。
    `_probe_gray_accuracy.py` 走**平台自己的** `_update_ann_stats()` / `_update_stats()`
    读「平均灰度」单元格，与真值比对：28 张全部误差 0.0000。
    ⚠️ 造这个集时踩到两个坑，以后改它要注意：
       ① 多边形右下角写 `X0+BOX` 会 fillPoly 出 201×201（多一圈 255 背景），
          真值必须按**光栅化后的区域**算，掩膜也要用 `shapes_to_mask` 生成才同源；
       ② 抖动不能用棋盘（只有一半像素可抬，frac 最大 0.5，183.6 会变成 183.5）。

30. **JSON 已存但 `annotations/<base>_mask.png` 不跟着更新**（2026-10-23 用户实拍）
    现象：用户给 `上大面_NG_划痕裂纹磕伤_008` 新画了 Hh×2 + Ps×1，JSON 里 3 个缺陷都在
    （mtime 09:39:40），但 `_mask.png` 里**只有 background + ignore**，缺陷全缺。
    原因：`_auto_save_json()` 每次编辑后只写 JSON；`_mask.png` 只在
    「导入标注」或手动 `save_mask()` 时生成 ⇒ 掩膜长期滞后于 JSON。
    影响面：训练读 `SegmentationDataset._load_mask`（**优先读 JSON**）所以没被污染；
    但任何读 `_mask.png` 的地方（评估工具、外部脚本）会拿到缺了缺陷的"真值"。
    修：① `ui/main_window.py` 新增 500ms 防抖定时器 → `_sync_mask_png_for_current()`，
    标注改动后自动把当前图掩膜写回（630×400 约 3ms，20MP 约 40ms）；
    ② 已按 JSON 重建该工程所有不一致掩膜（只 008 一个不一致，`[0,255]` → `[0,2,3,255]`）。
    ⚠️ 历史工程若怀疑掩膜陈旧，用同样的比对方式扫一遍：
       逐图 `shapes_to_mask(JSON)` 与现有 `_mask.png` 比对，不等才重写。

31. **venv 绝不能放项目目录：中文路径下 `QApplication` 永久卡死**（2026-11-06 实测）
    项目路径含中文和 `&`（`05-VibeCoding软件&工程`）。把 4.7GB 的 venv 建在项目内装完依赖，
    启动时 `QApplication([])` **永远不返回** —— 进程挂住、无异常、无日志、`try/except` 也拦不住，
    看起来就像卡死。
    **同一个 venv 原样 `mv` 到 `D:\_venv_moved`（纯 ASCII）立刻正常**，所以问题在路径不在包。
    排查手法：`faulthandler.dump_traceback_later(20, repeat=True)`，栈顶会停在
    `app = QApplication([])`；换 cwd 无效，只有换路径才有效。
    ⇒ 环境一律装纯英文路径（默认 `C:\aitp_venv`，备选 `%LOCALAPPDATA%\aitp_venv`），
      **代码 / wheels / vendor 可以放项目里**（它们只是文件，不参与 DLL 加载）。
    ⇒ `启动_便携.bat` 刻意**不找**项目内的 `.venv`，就是这个原因；
      `启动平台.bat` 会在项目里建 `.venv310`，别用。

32. **类别统计有了结果缓存：改代码时别漏掉指纹里的任何一项**（2026-11-10）
    `ClassTableView._compute_counts()` 现在先算一份"工程内容指纹"，没变就直接返回上次
    结果。实测极片 1002 张：**131ms → 5ms**，切走再切回 128ms → 4ms（冷算 +40ms，只付一次）。
    改动说明见 `docs/改动说明_类别统计缓存.md`。维护时要记住四条：
    ① 指纹（`_counts_signature()`）里**任何一个来源都不能少**：类别表、`images/` 快照、
       `images/<类别>/` 子目录、`annotations/` 快照、`train_test_split.json`、
       `project.json`、根目录 `class_labels.json`。少一个就是"改了它数字不动"，
       而且**界面上完全看不出异常** —— 这比算错更难发现。
    ② `_split_bases` 是计算的**副作用**（点数字格子过滤列表要用），缓存命中时必须一起
       还原（现在是存 set 副本）；漏了就是"点了数字格子没反应"。
    ③ 「刷新」按钮走 `MainWindow._refresh_class_stats()` → `invalidate_counts()`，
       **必须绕过缓存**，否则按钮看着没反应，也没有"mtime 没变但内容变了"的自救通道。
    ④ `_dir_index()` 返回 `None`（目录读不了）时**不进缓存** —— 不能把"读失败"当成
       "文件都没了"写进去。`_subdirs_sig()` 返回 None 也是同一条：任何一处读不了都不缓存。
    ⑤ **读写必须对称**：写入存的是副本（`dict(result)`），命中返回的也是副本
       （`dict(hit[1])`），`bases_for()` 返回 set 副本。第一版只有命中那边拷贝，
       审核实测"改返回值 ⇒ 缓存本体被污染 ⇒ 下次还错"。谁改成共用同一对象，就是埋雷。
    ⑥ 淘汰是 **LRU**（命中时把该条挪到 `dict` 末尾）。改成 FIFO 会把"来回切的两个工程"
       里最常用的那个先淘汰。
    ⚠️ `_flat_sig` 用内置 `hash()` ⇒ 受 `PYTHONHASHSEED` 影响 ⇒ **这份缓存只能活在进程内**。
       谁要落盘/跨进程复用，必须换成 `hashlib` 的稳定摘要，否则指纹永远对不上。
    ⚠️ 指纹口径与 `core.label_scan` 一样是 `mtime_ns + size`。SMB 网络盘上若时间戳
       只到 1 秒且大小不变，两套缓存都会漏，兜底只有「刷新」按钮。
    ⚠️ **跑绿不算数**：`test_counts_cache.py` 里两条用例第一版都是假绿（一条自证、
       一条从没命中过缓存）。要证明测试有杀伤力，就把防线逐个打掉看它红不红
       （变异检验，脚本放系统 TEMP、改完立即还原）—— 本轮 5 条防线 5/5 都验证过。
       另外注意 `new_table()` 是 `with_counts=False`，**它不算统计**，别拿它当"已经算过一次"。
    ⚠️ 顺手发现**未修**的旧问题：分类工程「空集」的比对区分大小写（`mapping` 键
       `images/ok/a.bmp` vs 类别名 `OK`）会把已标注的图算成空集。只影响 ImageFolder
       布局的工程，修它会改变那些工程现有数字，等拍板。

33. **停止标志现在挂在任务上，谁再写 `self._stop_flag` 谁就是开倒车**（2026-11-11，清单 P0-3）
    原来全平台一个 `self._stop_flag`，撑起训练/分类训练/批量推理/单张推理/OCR 六条路径。
    四个真实病灶（都已修，见 `docs/改动说明_任务级停止标志.md`）：
    ① **连坐**：取消队列里排队的一行推理，正在跑的训练一起停（`_on_task_cancel` 里那句
       `self._stop_flag = True` —— 已删）；
    ② **秒停**：取消过一次之后，后面每个任务在第一步就停（全局标志要靠各处 `finally` 复位）；
    ③ **分类训练压根不走队列**（`_start_cls_training` 是裸 `threading.Thread`），
       它和分割训练共用标志，点"开始分类训练"顺手把分割训练的取消状态抹掉 —— 现在入队了；
    ④ **提交即抹**：`_start_training` / `_batch_inference._run` 在提交那一刻就清全局标志，
       把用户刚点的取消悄悄干掉。
    现在：`task.cancel_requested` 是唯一状态，`task.stop_flag` 只是它的**读写别名**（别加第二个字段）；
    队列任务 `stop_check=lambda: ctx.task.stop_flag`；深层循环（`BatchRunner`/OCR）用
    `self._active_stop()`（有任务上下文就问任务，没有才看全局）；「停止」按钮走
    `_cancel_running_task(kind)`，**只打同类**正在跑的那个。
    ★ 2026-11-11 第二轮（审核实测）：「分家」原来**只对队列任务成立**。分类批量 /
       三视图 / 检测批量 / OCR 都是裸线程，走 `_active_stop()` 的全局兜底 ⇒
       点「停止训练」会把正在跑的推理静默停掉。现在：
       - **`_active_stop()` 已删除**（它靠单槽 `_active_infer_ctx` 猜"我是哪个任务"，
         有第二个任务就必错）。别再把它加回来。
       - 改成 `_infer_stop(stop=None)`：队列任务传自己的标志，裸线程推理现造一个
         `threading.Event` 令牌（存在 `self._infer_stops`，`_stop_infer_tokens()` 统一点亮）。
         五个推理入口都把 `stop` 显式往下传，深层 `BatchRunner(stop_check=stop)`。
       - 令牌**不写 `finally: release()`** —— "每个出口都要记得复位"就是旧 bug 的根因；
         已置位的旧令牌不影响下一轮（每次现造）。容器是 `deque(maxlen=32)`，
         按**添加顺序**丢最老的（原来写"清掉已置位的"，而跑完的令牌根本不会被 set，
         等于清不掉 —— 审核抓的 P2）。
    ⚠️ **批量改 `stop_check=` 时，每一个函数都得确认 `stop` 在自己作用域里有定义。**
       我就在这里漏了 `_ocv_batch_inspect`（只改了用法、没定义）⇒ OCV 批量一点就
       NameError，还是抛在子线程里被吞掉。现在由 `tests/test_undefined_names.py` 兜。
    ⚠️ `self._stop_flag` **还在**，但只剩一个用途：**没有队列控件时的训练兜底线程**。
       推理一行都不读它了。**新写的流程别再用它** —— 走 ctx 里的任务标志，或 `_infer_stop()`。
       等所有训练/推理都入队后才能删，现在删就是"删了半边"。
    ⚠️ 同轮还抓到一个真 bug：`_dispatch_batch_inference` 调的 `_start_det_batch_inference`
       **全项目不存在**（真名 `_run_det_batch_inference`）⇒ 检测工程批量推理一进队列就
       AttributeError，队列记成 failed 且不弹窗。已修，并加了"分派方法必须存在"的静态用例。
    ⚠️ 检测训练（YOLO）仍未入队，走它自己的 `_det_trainer.stop()`；要统一得先让它入队。
    ⚠️ **这类 bug 不能拿"跑一遍不崩"当验证**：复现窗口是"A 正在跑、B 还在排队"的那一瞬间。
       `tests/test_task_isolation.py` 就是按这个时序造的（起一个卡住的任务当"正在跑的那个"，
       再塞第二个进去，然后对第二个动手），并且把**旧的缺陷形状**逐条塞回去验证它会红（7/7）。
    ⚠️ 顺手记一条：`new_table()` 那类"看起来像已经算过一次"的辅助函数害人不浅 ——
       本轮又遇到一次（`TaskQueue.running` 之前是私有 `_running`，测试想验"当前任务"就得自己造）。

34. **失败要能判定：`core/result.py` 已落地，迁一个模块算一个**（2026-11-11，清单 P0-1）
    全局 610 处 `except Exception:`、220 处紧跟 `pass` —— 问题不是难看，是**失败没有语义**：
    "缓存坏了"和"缓存没建过"返回同一个值，日志里一个字都没有。
    现在 `core/result.py` 给新代码用（**零侵入**，老代码照旧）：
    - `Result.success(v)` / `Result.fail(code, msg, where=...)`；`bool(r) == r.ok`
      ⇒ `if save(...)` 这种老写法不改也能跑，可以一个文件一个文件迁。
    - `ErrCode` 是**字符串常量**（`"corrupt"`/`"io"`/`"not_found"`…），日志里能直接 grep。
      `ErrCode.BENIGN = (OK, NOT_FOUND)` 表示"本来就没有"，调用方不该报警。
    - `ok` 是**属性**不是字段 —— 构造不出 `ok=True` 又 `code=IO` 的矛盾对象；
      `fail(ErrCode.OK, ...)` 会被当场纠成 `UNKNOWN`。
    - `@guard(ErrCode.IO, where="...")` 给会抛异常的老函数套一层；
      函数本来就返回 Result 时**原样透传**，不会包第二层。
    已迁移的**唯一真实模块**：`core/mask_stats_cache`（load/save），四种静默失败拆成
    `not_found / corrupt / version / io`；`main_window` 的调用点会在日志里写
    「[掩膜缓存] corrupt / load: … —— 本次全量重算」（以前只表现为"打开工程慢 6 秒"）。
    ⚠️ 失败时 `value` 是 `None`，取默认值要 `.unwrap_or({})`，别直接 `.value`。
    ⚠️ 下一步迁移建议按"失败最该被看见"排序：`ui/task_queue._save_history`（静默吞异常）、
       `ui/class_table` 的两处 JSON 读取。**别一口气改 610 处** —— 那等于重写整个错误处理。

35. **`tests/test_undefined_names.py`：名字写错不再靠人眼**（2026-11-11）
    用 stdlib `symtable` 扫 `ui/ core/ training/ …`，找"被引用、但不是局部/参数/import/
    闭包变量、也不在模块级或 builtins 里"的名字 —— 那种名字运行到必 `NameError`。
    它是被一次真实事故逼出来的：把 `stop_check=self._active_stop` 批量替换成
    `stop_check=stop` 时，`ui/ocr_ocv_mixin.py` 的 `_ocv_batch_inspect` **只改了用法、
    没定义 `stop`** ⇒ OCV 批量检测 NameError，抛在子线程里被吞，界面只停在
    "Batch inspecting..."。**91 项测试全绿也没抓到**：时序用例全在队列层/标志层。
    这一扫还揪出三条**既有**的同类 NameError（都不是新引入的）：
    `training/multiview.py` 的 `_log`（正好在"用户点停止"那条日志上）、
    `classification/trainer.py` 的 `scale_factor`（`k_folds>1` 存折 ckpt 时）、
    `ui/system_monitor.py` 的 `ctypes`（被 `except` 吞成"核显显存读不到"）。
    ⚠️ 它只查**名字**。方法名拼错靠 `test_dispatch_targets_exist`；`self.xxx` 属性名拼错
       目前没人管（要类型信息）。三类是互补的，别指望一个静态检查包打天下。
    ⚠️ 扫描器自带自检（喂一段坏/好源码，确认能抓且不误报）—— 没有自检的静态检查
       写错了会返回空列表，测试永远是绿的。

36. **切换工程必须"先清后载"，而且要在改 `_project_dir` 之前清**（2026-11-11 用户报）
    现象：双击「工程管理」里另一个工程，右边画布还是**上一个**工程的图（连文件标签、
    已标注数都是旧的）。不是"没切换"，是切换期间**还挂着上一个工程的东西**：
    - `canvas.set_project()` 只设 `project_dir`/`label_manager`，**不清 image/mask/叠加层**；
    - 首图要等后台列表扫完（海目星二期 2001 张实测 **9.6s 才出列表、10.8s 才出图**）。
    现在 `_open_project_by_name()` 开头调 `_reset_view_for_project_switch()`：
    停掩膜防抖 + 清画布全部字段 + 清列表行 + 信息栏归零；紧接着
    `_preview_first_image()` 直接 scandir 第一张图显示出来（~0.3s），
    实测"看到新工程的图" 10.8s → **1.1s**。`_close_project()` 共用同一套清理。
    ⚠️ **顺序**：清理必须在 `self._project_dir = <新工程>` **之前** ——
       `_mask_png_timer`（标注改动→写 `annotations/<base>_mask.png`，500ms 防抖）
       在触发时才读 `_project_dir`。放后面就会把**旧图的掩膜写进新工程的 annotations**
       （文件名对不上任何图，统计时像"孤儿文件"，极难发现）。
    ⚠️ 列表构建那 9.6 秒**没优化**（扫描+掩膜预热是真实开销）。想让首屏更快，
       方向是"先出列表、统计后补"，不是继续改这段。
    ⚠️ 预览只认 `images/` **顶层**第一张：分类工程是 `images/<类别>/` 布局，直接跳过
       （猜错类别的图比空白更让人困惑）。
    ⚠️ 预览的**排序必须和列表一致**：默认（未勾「Recent」）列表是
       `sorted(os.listdir())` ⇒ 取字典序最小；勾了「Recent」列表按**标注 JSON 的 mtime
       降序** ⇒ 预览要按同一 key 挑（`_preview_first_image` 里已分两支）。
       不一致的表现是"预览一张、列表选中另一张"，用户觉得图没变却跳了行。
    ★ 清字段的**清单住在 `AnnotationCanvas.reset_for_project_switch()`**
       （`annotation/mask_editor.py`），不在主窗口。给 canvas 加任何"跟当前图绑定"的字段，
       都必须去那儿补一行 —— 加字段的人和清字段的人在同一个文件，才不会再漏。
       用户偏好（`_mode`/`current_class_id`/`brush_size`/显示开关/色表）刻意不清。
    ⚠️⚠️ 方法名**绝不能叫 `reset_view`**：本类末尾已有一个 `reset_view()`（工具栏
       「刷新视图」，只做 `_fit_to_window()`），Python 类体里后定义的覆盖先定义的
       —— 我第一版就叫 `reset_view`，那 58 行清理**一行都没执行**，主窗口的 15 个
       `setattr` 又已经删了，净效果是切换工程时**一个字段都没清**（比改前更糟）。
       没报错、没警告、测试全绿。现在由 `tests/test_duplicate_defs.py` 兜。
    回归：`tests/test_project_switch_view.py` 5 条（含"返回那一刻不许还是旧图"、
    "不等列表就有新工程首图"、"来回切不串工程"），变异 3/3 全红。

37. **`tests/test_duplicate_defs.py`：类内同名方法 = 后一个静默覆盖前一个**（2026-11-11）
    Python 不会警告。改完后症状是"新代码好像没生效"，极难往"方法重名"上想。
    扫**两个作用域**：模块级（`tree.body`，作用域名记 `<module>`）+ 类体内
    （`ClassDef.body`，记类名）。按名字分组，≥2 次即报。
    ⚠️ 第一版**只扫类体**，所以 `tests/` 里两个 `def test_xxx` 永远抓不到
       （审核 2026-11-11 指出的场景：前一条用例静默消失，条数变少却没人发现）。
       只看**直接子节点**，所以 `if 平台: def f()` / `try-except: def f()` 这类条件定义不误报。
    排除语言允许的情形：`@property` + `@x.setter/getter/deleter`、`typing.overload`、
    `singledispatch.register`。当前 194 个 `.py` 里 **0 处**真重名（4 处 property 对被正确放过）。
    ★ 扫描范围**含根目录**（`main.py` / `launcher.py` 是程序入口，不在任何子目录里），
      只排除一次性探针 `_probe_*.py`。判断逻辑抽成纯函数 `root_file_ok(fn)` ——
      原因见下一条。
    ⚠️ 另一条假绿也堵了：解析不了的文件原来被静默跳过（`except SyntaxError: return []`），
       大半个项目读不进来时结果永远是"0 处"而测试照绿。现在 `test_source_parse_failures_are_zero`
       断言解析失败数**必须为 0**，`test_tests_dir_is_scanned` 断言 `tests/` 真的在范围内。
    ⚠️ 写这类断言时**别依赖"环境里恰好存在某样东西"**：第一版用"扫出来的列表里没有
       `_probe_`"来证明前缀过滤生效，而 `git archive` 的副本里根本没有 `_probe_*.py`
       ⇒ 变异时这条断言没红（空转）。改成断言纯函数 `root_file_ok("_probe_hang.py") is False`。
    ⚠️ 它与 `test_undefined_names.py` 互补：那个查"名字不存在"，这个查"名字存在两份"。
    ⚠️ 读源码给 `ast.parse` 时**传 bytes**，别 `io.open(encoding="utf-8")` ——
       本项目一半文件带 UTF-8 BOM，读成 str 会变成 `U+FEFF` 然后满屏"语法错误"。
    ⚠️ `annotation/mask_editor.py` 整份是 **CRLF**，写脚本改它时锚点要按 CRLF 或先归一化；
       但**别顺手把整个文件的换行符也改了**（会让 diff 变成 2400 行）。

38. **变异检验必须打在"会被执行"的路径上**（2026-11-11，审核驳回那次的根教训）
    上一轮我做的变异是「把字段名写回主窗口 → 源码扫描红」，那条打在**源码文本层**。
    真实情况是清理函数根本没被调用，所以"删掉清理逻辑"这种变异**不会红** ——
    **变异护住了一个不存在的东西**，测试全绿也说明不了任何事。
    判据：一个变异能变红，前提是那段代码真的在跑。改完搬迁/改名/重构后，
    **先用 `inspect.getsource()` 把"实际生效的那段"打出来看一眼**，比读源码可靠
    （重名方法 / 装饰器包装 / 条件导入都会让你读到的那份不是执行的那份）。
    同理，断言要挑"没有别的路径会顺手修好"的字段：切工程后 `current_image_path`
    会被首图预览覆盖，所以拿它当唯一断言等于没验。

39. **配置读写统一走 `core/settings.py`（P0-2 第一批，1/7）**（2026-11-12）
    把配置声明成一张表，`load()` 逐字段校验：缺失补默认（**不算错误**）、
    类型错/超范围报 `invalid` 并指名键、未知键保留、文件坏/不存在各自错误码。
    `save()` 合并写 + 原子替换（临时文件 + fsync + `os.replace`）。
    已迁两个真实调用点：`_load_panel_train_settings` / `_save_panel_train_settings`。
    关键的行为改进：**一个坏字段不再拖垮整份配置** ——
    改前 `int(s.get("epochs"))` 抛异常被 `except: pass` 吞掉，后面所有字段都不再加载，
    面板一半旧值一半文件值，日志一个字都没有。
    ⚠️ `validate()` 是唯一判据，`merge()` 只认它返回的 `Problem`。别在 merge 里另判一套。
    ⚠️ 超范围是"夹到边界"、类型错才是"退回默认值" —— 前者保住用户意图，后者没意图可保。
    ⚠️ `augment_flags` 只校验最外层是 dict，内层交给 `training.augment_config.merge_flags()`。
    ⚠️ 还剩 6 组配置没迁：`cls_train` / `ui` / `gate` / `task_history` / 自定义工程表 / 其余。
       别在清单上打勾，P0-2 标的是"进行中 1/7"。
    ★ `save()` 返回 `SaveReport(written, ignored)` —— 调用方给了 schema 没声明的键
      （多半拼错）**不写进文件，但会报回来**，调用点写日志。别把它改回"返回 True"：
      `save({"epohcs": 200})` 静默不写，调用方以为存上了，下次读出来还是老值 ——
      那正是 P0-1 要治的"失败没语义"。
      往 `Result` 上加 `ignored_keys` 是另一种选择，但 `Result` 是通用容器，没这么做。
    ⚠️ 为了让"拼错键写日志"这条**测得到**，收集逻辑抽成了
      `_collect_panel_train_settings()`（测试里替换成返回带 typo 的字典）。
      面板正常收集不会拼错，不抽出来那条分支永远走不到 = 写了不验。

40. **跑测试会改写用户的 `.train_settings.json`**（2026-11-12 修）
    `closeEvent` 会调 `_save_panel_train_settings()`，而 `test_smoke_startup` /
    `test_load_progress` / `test_close_event` 都会建窗口再 `close()` ⇒
    **跑一遍测试，用户的 epochs / batch_size / tile_size 就被换成测试窗口里的值**。
    以前没人发现，是因为没人 diff 这个文件。
    现在三个测试在关窗之前把 `_train_settings_path` 指到临时文件
    （与 `.task_history.json` 那套做法一致）。**新写会关窗的测试请照做**：
    ```python
    w._train_settings_path = lambda: os.path.join(
        tempfile.mkdtemp(prefix="aitp_cfg_"), ".train_settings.json")
    ```
    验证办法：跑全套前后 `md5sum .train_settings.json .task_history.json`，必须一致。

41. **`.train_settings.json` 有 3 个写入者，谁都不能全量覆盖**（2026-11-13）
    | 写入者 | 状态 | 说明 |
    |---|---|---|
    | `main_window._save_panel_train_settings` | 活 | 走 `settings.save`，合并 + 原子 |
    | `training.augment_config.save_augment_flags` | 活 | 只改 `augment_flags`，合并 + 原子 |
    | `ui.widgets.TrainSettingsDialog._save_and_accept` | **不可达** | 已修，见下 |
    ⚠️ 那个对话框**当前没有任何地方构造它**：`main_window.py:8834` 那两行被注释掉了
       （原注释"不再弹 Training Settings 窗（用户要求）"）。它原来只写 9 个键、
       而文件有 15 个 —— 一旦有人照注释"回退成弹窗模式"，第一次点「确定」
       就会静默删掉 `augment_flags / early_stop_patience / infer_mode /
       tile / tile_size / tile_stride` 这 6 个键。**已改成 `settings.save`**。
    ⚠️ 对话框**读** `tile`/`tile_size`/`tile_stride` 来初始化控件但**不写**它们
       （面板那份才是权威）。别"顺手"补上——那会让对话框和面板互相覆盖，
       这是产品决定，不是 bug 修复。
    ⚠️ 判断"这个写入者会不会丢键"的最快办法：搜它是不是 `open(path, "w")`，
       再看它写的键数是不是等于 schema 的键数。不等就会丢。

42. **写变异脚本必须先过语法检查**（2026-11-13，审核建议，已落地）
    变异把源码改出语法错误时，测试也会"红"—— 但那是 import 失败，什么也证明不了。
    （2026-11-12 踩过：第 4 条变异把代码替换成了半句，报红是假的。）
    现在每条变异改完先：
    ```python
    def parses(path):
        try:
            ast.parse(open(path, "rb").read(), path)   # 传 bytes，别读成 str
            return True
        except SyntaxError as e:
            print("!! 变异把文件改坏（%s:%s %s）—— 判为无效" % (path, e.lineno, e.msg))
            return False
    ```
    不通过就恢复并打印"[无效]"，**不进入跑测试那一步**。
    ⚠️ 读源码时用 `utf-8-sig` 或直接喂 bytes：`training/augment_config.py` 带 BOM，
       `utf-8` 解码会留 `U+FEFF`，`ast.parse` 报"非法字符"（本项目第二次栽在 BOM 上）。
    ⚠️ 替换整个函数别用长字符串锚点（会被转义搞坏），用 `ast` 拿 `lineno/end_lineno`
       切行更稳。

43. **配置文件 × 写入者 有一张总表，改配置前先看一眼**（2026-11-13）
    见 `docs/配置写入者清点.md`。三个要点：
    ⚠️ **`core/config.load_config()` 是平台启动的第一步**（`main_window.__init__:176`
       直接 `self.config = load_config()`，外面没有 try）。它原来是裸 `json.load`，
       而 `save_config` 是非原子写 —— 保存时崩一次，下次**双击图标毫无反应**
       （实测抛 JSONDecodeError）。现在读容错（坏文件改名 `*.bad` 留档 + 与
       `DEFAULT_CONFIG` 合并）、写原子。别把 try 拿掉。
    ⚠️ **`.custom_projects.json` 登记所有外部工程的路径**，写坏后
       `_load_custom_dirs()` 静默返回 `{}` ⇒ 「工程管理」里外部工程全消失
       （文件还在磁盘上）。已改原子写。
    ⚠️ 判断一个新的写入者危不危险，三步：搜 `open(..., "w")` / `json.dump(` →
       看它写的键数是否等于这份文件的全部键（等于=全量覆盖，小于=合并）→
       看有没有 `os.replace`（没有就是非原子，半截文件 = 读侧整份作废）。
    ★ 已知但**本轮没修**的（都在清点文档 P2/P3）：
       `_save_linked_models` / `train_test_split.json` 两个写入者 /
       `class_labels.json` 两个写入者 / `project.json` 两个活写入者 —— 全是"合并对、
       但非原子"；`project_manager.save_project_meta` 是**没有调用方的全量覆盖**；
       `main_window._save_ui_setting()` **根本不落盘**（只进内存，重启就丢，
       主题除外——`ui/themes.py` 自己写文件）。

44. **对话框里"只读但照实显示"的控件**（2026-11-13）
    `TrainSettingsDialog` 的 `tile_check` / `tile_spin` / `stride_spin` /
    `tile_mode_combo` 现在**置灰**，并有一行 `tile_hint` 指向"主界面「分块」设置"。
    原因：这四个键的权威在主面板，对话框写了会和面板互相覆盖；而以前它们可以改、
    点「确定」却不保存 —— **UI 在撒谎**。
    ⚠️ 别"顺手"把 `setEnabled(False)` 去掉，除非同时让 `_save_and_accept()` 写这些键
       并想清楚谁赢。

45. **目标检测：「导入图片 + 导入标注」之后能直接训练**（2026-11-16）
    用户的验收口径就是这两步做完能设参数开训。以前不行，根因两条：
    ① 训练只认 `images/<base>_det.json`（`detection/dataset.py`）；
    ② 界面导入 JSON 后不分任务类型，只生成**分割掩膜** `_mask.png`，
       一个 `_det.json` 都不产出 ⇒ 训练报 `No detection annotations found`。
    现在 `_import_json_for_selected()` 按 `_is_detection_project()` 分流：
    检测 → `_rebuild_det_for_imported()`（labelme 矩形 → `_det.json`）；
    其余 → 老路生成掩膜。转换实现**只有一份**：`detection/labelme_rect.py`，
    命令行 `tools/labelme_rect_to_det.py` 也调它。
    ⚠️ 别在 UI 里再抄一份转换逻辑；也别用"有没有 `_mask.png`"判断分支 ——
       画布预览首图时会自己写一张掩膜，那个判断必然误判（测试第一版就栽了）。

46. **顺带修的两条（同批，都是真缺陷）**
    ① `_open_project()` 里 `internal` 只在 `try` 成功时赋值，工程缺 `project.json`
       或 JSON 损坏时下一行 `internal in (...)` 直接 NameError。映射表已收到
       `core.project_manager.read_task_type()`（中英两套 task_type 的唯一出口）。
       ⚠️ `test_undefined_names` 抓不到这种"可能未绑定"，别以为静态扫描过了就没事。
    ② `detection/__init__.py` 硬导入 `box_overlay`，于是 `import detection.dataset`
       会顺带把 PyQt5 拉进来 —— 无显示环境直接 ImportError，且 PyQt5 先于 torch
       导入会触发 WinError 1114（地雷 #2）。`DetectionOverlay` 改成 PEP 562 懒加载。

47. **一条自查纪律（这次变异测试实拍）**
    新写的 UI 测试**必须走用户真实入口**（`_import_json_for_selected`），
    不许直接调私有的 `_rebuild_det_for_imported` —— 后者会让
    "把 `if self._is_detection_project():` 改成 `if False:`" 的变异照样全绿。
    同理，验证"类别顺序"的用例，数据里的顺序**不能刚好等于字典序**，
    否则实现偷懒重排一遍也测不出来。

48. **重建 YOLO 数据集前必须清空旧产物**（2026-11-16，指标会骗人）
    `build_yolo_dataset()` 拷图是 `if not dst_img.exists()` 跳过，全文件没有
    清理逻辑。划分又是 `seed(42)+shuffle(按文件名排序)`：**列表长度一变，置换整体改变**。
    补图后某张图跑到 val，上一轮的 train 副本不会被删 ⇒ 同一张图**同时在 train 和 val**
    ⇒ 训练见过、验证又考它 ⇒ mAP 虚高，界面上看不出来。删图方向则是旧副本继续被训练。
    现在划分算完先 `_wipe_previous_dataset()` 整目录删（含 `data.yaml`），返回值
    `removed_files` 打到日志；清理失败**转 RuntimeError 抛**（别 `ignore_errors`，
    那就是"看着重建了其实还是旧的"）。
    **防护用不变式，别按特例列举**：`if proj not in out.parents: raise`。
    一条盖住三种误传 —— 工程根 / 工程上级 / **另一个工程**（上一版是两条特例，
    正好漏掉第三种：它同样有 `images/`，会被弱标记判成数据集整个删掉）。
    第二级再看长相：非空且没有 `data.yaml`、也没有 `images/`+`labels/` → 拒绝，
    挡 `output_dir=proj/images` 这种工程内手滑。
    判据**不许退回 `any(...)`**（三个标记有一个就算）：工程根、另一个工程都长那样。
    ⚠️ 写这类用例必须**先证明前提成立**：实测 seed=42、val_split=0.2 时，
       原张数 5 / 10 / 20 补一张图**划分根本不变**，那种规模下"断言不重叠"是空转。
       `tests/test_det_dataset_rebuild.py` 里 `_assert_split_changed()` 就是干这个的。
    ⚠️ 造"文件被占用"的失败**不能 patch `shutil.rmtree`**（连 `ignore_errors=True`
       一起屏蔽掉，变异照样绿）；要真 `open()` 住一个文件再删。
    ⚠️ 护栏用例要断言**错误信息里的关键词**。否则第二级会替第一级挡枪：
       实测把不变式换回旧特例，"另一个工程"仍然被二级防护拦住，测试看着是红的，
       但红在信息不对 —— 不加信息断言，将来放松判据就悄悄只剩一层。

49. **检测训练不许连点两次**（2026-11-16，与 #48 同批）
    `_start_det_training()` 是直接起线程（走任务队列的是**分割**训练），连点两次
    就两个训练同时跑：抢显存，而且第二份重建数据集时会把第一份正在读的
    `yolo_dataset/` 删掉。现在用 `_det_training_running` 挡住，弹一句提示；
    结束（`_on_det_training_done`）/ 线程没起来时都复位。
    ⚠️ 如果以后把检测训练也挪进任务队列，这个标志要一起删，别留两套并发控制。



## 8. 换电脑 / 换机器维护 SOP

### 8.1 东西都在哪

| 东西 | 位置 | 体积 | 跟项目走 |
|---|---|---|---|
| 代码 | 项目文件夹 | 79MB | ✅ |
| 离线 wheel（37 个） | `项目\wheels\` | 2.8GB | ✅ |
| Python 3.12 安装包 | `项目\installers\` | 26MB | ✅ |
| OCR / onnxruntime / shapely / pyyaml | `项目\vendor\` | 72MB | ✅ |
| Python 环境 | `C:\aitp_venv` | 4.7GB | ❌ 必须英文路径（地雷 31） |
| 真实工程数据 | `E:\05-TrainingProject\` | — | ❌ 单独拷 |
| SAM / YOLO 预训练权重 | `项目\models\` | 363MB | 按需 |

`wheels/` `installers/` `vendor/` 都在 `.gitignore` 里（不进 git），但**跟着文件夹拷贝走**。

### 8.2 搬到新机器：四步，全程可以不联网

1. 拷整个项目文件夹（约 2.9GB）
2. 双击 **`安装Python3.12.bat`**
   - 已经装了 → 检测到直接跳过
   - 没装 → 用项目自带的 `installers\python-3.12.7-amd64.exe` 静默安装（`InstallAllUsers=0`，
     装到 `%LOCALAPPDATA%\Programs\Python\Python312`，不需要管理员权限）
   - 自带包缺失 → 退到 `winget install Python.Python.3.12`
   - 都没有 → 提示手动下载地址
3. 双击 **`离线安装.bat`** —— 没环境就建 `C:\aitp_venv` 并离线装，约 4 分钟
4. 双击 **`启动_便携.bat`** —— 依次找 `C:\aitp_venv` → `%LOCALAPPDATA%\aitp_venv` → PATH

⚠️ wheel 全是 cp312，所以**必须 Python 3.12**，3.13 / 3.14 装不上。
⚠️ **不要**把环境装进项目目录，也不要手工在项目里 `python -m venv .venv`（地雷 31）。

### 8.3 只更新依赖、不换机器

```cmd
:: 小依赖（OCR 那几个）→ 装进项目 vendor\
pip install --target .\vendor --no-deps --upgrade 包名

:: 大包（torch 等）升级 → 重下 wheel 并刷新锁文件
pip download -d wheels --no-deps 包名==版本
pip freeze > requirements-lock.txt
```

也可以直接双击 `安装vendor依赖.bat`（重建整个 vendor\，需要联网）。

### 8.4 三条命令判断环境健不健康

```cmd
:: ① 关键包能导入
C:\aitp_venv\Scripts\python.exe -c "import torch, PyQt5, rapidocr_onnxruntime, shapely, yaml; print('ok')"

:: ② Qt 能起来（最容易暴露路径问题：卡住不返回 = 路径含中文）
C:\aitp_venv\Scripts\python.exe -c "from PyQt5.QtWidgets import QApplication; a=QApplication([]); print('qt ok')"

:: ③ 平台能构造主窗口（无显示器也能跑）
set QT_QPA_PLATFORM=offscreen
C:\aitp_venv\Scripts\python.exe tests\test_smoke_startup.py
```

### 8.5 本机日常

双击 `启动_本机.bat` 或 `启动_便携.bat` 都行，两者都指向 `C:\aitp_venv`。

### 8.6 脚本别混用

| 脚本 | 说明 |
|---|---|
| `安装Python3.12.bat` | 装解释器：自带包静默装 → winget → 手动提示，已装则跳过 |
| `启动_便携.bat` | **推荐**，自动找环境，只认英文路径 |
| `离线安装.bat` | 换机器用；已有可用环境会自动跳过，不会重装 |
| `安装vendor依赖.bat` | 只重建 `vendor\`（72MB，需联网） |
| `启动_本机.bat` | 老脚本，硬编码 `C:\aitp_venv`，可用 |
| `启动平台.bat` | 会建 `.venv310` 并联网装依赖，**别用**（地雷 31 + 版本不可控） |


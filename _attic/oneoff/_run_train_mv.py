"""无头启动三视图训练（与界面「开始训练」走同一个 MultiViewTrainer）。

为什么要有它：CPU 上跑一轮要几十分钟，放前台会一直占着。它把输出**同时**写到
控制台和日志文件，所以双击 `train_mv.bat` 就行，不用让工具进程陪着等。

    C:\\aitp_venv\\Scripts\\python.exe _run_train_mv.py [工程目录] [epochs] [batch] [--dry]
    C:\\aitp_venv\\Scripts\\python.exe _run_train_mv.py --dry      # 只做数据预检，不训练

⚠️ 不传工程目录时用下面的 DEFAULT_PROJ。**中文路径只写在这里，不许写进 .bat**：
cmd.exe 用本机 OEM 代码页（GBK）解析 .bat，字面量中文路径会被毁成问号
（这条在 ai_chatroom 项目上已经栽过一次，见其运维铁律：守护进程跑在错的房间里还看起来正常）。

⚠️ 为什么必须从工具外启动：工具侧后台作业**会在命令返回时回收整棵子进程树**。
2026-10-14 实测 —— 用 subprocess + DETACHED_PROCESS|CREATE_BREAKAWAY_FROM_JOB 起的训练，
日志停在启动后 4.2 秒（模型信息刚打完、正要开跑），进程随后消失且**没有 traceback**
⇒ 不是 Python 异常退出，是被外部终止。双击 bat 启动的进程不属于工具会话，不会被回收。
"""
import os
import sys
import time

import torch  # noqa: F401  必须最先导，PyQt5 在前会撞 c10.dll（本机铁律）

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_PROJ = "E:\\05-TrainingProject\\新松_233_赣州孚能_大面-缺陷检测_iSense104_Arthur_20260913"
LOGDIR = "E:\\05-TrainingProject\\_train_logs"


class _Tee(object):
    """把 stdout/stderr 同时喂给控制台和日志文件。

    只写文件的话双击时看不到进度；只写控制台的话窗口一关证据就没了。
    日志文件必须留得住，否则"跑到一半没了"这种事根本没法复盘。
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


def _install_tee():
    try:
        os.makedirs(LOGDIR, exist_ok=True)
    except Exception:
        return None
    logpath = os.path.join(LOGDIR, "mv_train_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
    try:
        fh = open(logpath, "w", encoding="utf-8", buffering=1)
    except Exception:
        return None
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    return logpath


def main():
    argv = list(sys.argv[1:])
    dry = "--dry" in argv
    argv = [a for a in argv if not a.startswith("--")]

    # 空字符串也要当"没传"（bat 里 `train_mv.bat "" 3 8` 会送来一个空参数，
    # 而 os.path.abspath("") 会悄悄变成当前目录 —— 那就会去训一个空工程）
    proj = os.path.abspath(argv[0]) if (argv and argv[0].strip()) else DEFAULT_PROJ
    epochs = int(argv[1]) if len(argv) > 1 else 1
    batch = int(argv[2]) if len(argv) > 2 else 4

    logpath = _install_tee()

    from training.multiview_gui import MultiViewTrainer

    t_start = time.time()

    def L(msg):
        print("[%s %+7.1fs] %s" % (time.strftime("%H:%M:%S"), time.time() - t_start, msg),
              flush=True)

    L("日志文件: %s" % (logpath or "(写不了，只能看控制台)"))
    L("工程: %s" % proj)
    L("epochs=%d batch=%d 设备=%s%s" % (epochs, batch,
                                       "cuda" if torch.cuda.is_available() else "cpu",
                                       "  [--dry：只预检，不训练]" if dry else ""))
    if not os.path.isdir(proj):
        L("!! 工程目录不存在，先检查 DEFAULT_PROJ")
        return 2

    L("读一个样本确认 9 通道...")
    from training.multiview import MultiViewSegDataset
    ds = MultiViewSegDataset(proj, "train", image_size=256, tile_size=256,
                             tile_mode="slide", tile_stride_pct=75)
    x, y = ds[0]
    import numpy as np
    L("x=%s y=%s 类别=%s 块数=%d" % (tuple(x.shape), tuple(y.shape),
                                     sorted(set(np.unique(y).tolist())), len(ds)))
    if dry:
        L("--dry 结束（未训练）")
        return 0

    tr = MultiViewTrainer(proj, model_name="unet", unet_base=16, unet_depth=4,
                          tile_size=256, tile_mode="slide", tile_stride_pct=75)
    t0 = time.time()
    m = tr.train(epochs=epochs, batch_size=batch, image_size=256, loss_name="ce_dice",
                 augment=None, k_folds=1, resume=False, log_callback=L)
    L("训练耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    L("metrics: %s" % m)
    L("模型已写入 models/ 目录")
    return 0


if __name__ == "__main__":
    # torch 的多进程 DataLoader 需要 __main__ 保护；这里 num_workers 在 CPU 下是 0，
    # 但 spawn 的 start_method 一旦被设过就要求入口保护，留着更安全。
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

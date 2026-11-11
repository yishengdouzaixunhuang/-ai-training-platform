r"""底栏实时资源占用（CPU / 内存 / 磁盘 / GPU）。

**为什么 2026-10-15 重写**：用户拿底栏和任务管理器对，发现"不同步"。逐项核实后
结论是——**有两项压根不是同一个指标，却被同一个"%" 呈现**：

| 项| 原实现 | 任务管理器 | 真相 |
|----|--------|-----------|------|
| 内存 | psutil used/total | 13.9/31.8 GB | ✅ 口径本来就一致 |
| CPU  | psutil.cpu_percent(None) | 77% | ⚠️不是错，是**刷新相位不同**（且只显示整机，看不到自己吃多少） |
| 磁盘 | `C: 86.2%` = psutil.disk_usage.percent（**空间已用率**） | 磁盘0(C:) 1% = **I/O 活动率** | ❌ **同名不同义**，两个数本来就不可能对上 |
| GPU  | 无 CUDA 就不显示（静默消失） | 0.2/15.9 GB（WDDM 共享显存） | ⚠️ nvml 读不到核显 |

所以本版做三件事：
1. 磁盘**两个口径都给**：`C: 空间86% · IO 1%`（IO = 两次采样间的读写字节/时间，
   与任务管理器"磁盘活动"同源）。
2. CPU **整机 + 本进程**并排（`CPU 66% · 进程 12%`）：训练时能直接看自己吃多少。
3. GPU 改读 **Windows 性能计数器**（`\GPU Adapter Memory(*)\Shared Usage`），
   核显也能拿到；确实读不到时**明写"不可读"**，不再默默消失 ——
   "测不到"≠"没占用"，静默不显示会让人误判。

★ CPU 采样为什么要换线程：`psutil.cpu_percent(interval=None)` 的语义是
"距**上次调用**的平均CPU%"。若定时器被阻塞（弹模态框、调试断点），那个窗口会被
拉长到十几秒，出来的数与任务管理器差得更远，看着更像"不同步"。
改成后台线程按**固定 1s 窗口**采样并缓存，主线程只读缓存 ⇒ 数值稳定可对照，
且**主线程不做任何采样工作**（psutil 的同步采样会卡 UI）。
"""
import threading
import time

import psutil
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel

#: I/O 活动率按这个字节/秒算 100%（任务管理器的"活动 %"同量级）
_IO_FULL_RATE_MBPS = 40.0


class _Sampler(threading.Thread):
    """后台采样线程：固定 1s 窗口，UI 线程只读缓存。"""

    daemon = True

    def __init__(self):
        super().__init__()
        self._stop = threading.Event()
        self.lock = threading.Lock()
        self.data = {"cpu": 0.0, "cpu_proc": 0.0,
                     "mem_used": 0.0, "mem_total": 0.0, "mem_pct": 0.0,
                     "disk": {}, "io_pct": {}, "gpu": None}
        self._proc = psutil.Process()
        self._io_prev = None
        self._io_ts = None
        self._gpu = _GpuProbe()

    def stop(self):
        self._stop.set()

    def run(self):
        # 预热：psutil 第一次调用返回 0.0，必须先"空跑"一次
        psutil.cpu_percent(interval=None)
        psutil.Process().cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)
        try:
            self._io_prev = psutil.disk_io_counters()
            self._io_ts = time.time()
        except Exception:
            self._io_prev = None
        while not self._stop.is_set():
            self._stop.wait(1.0)
            if self._stop.is_set():
                break
            try:
                self._sample()
            except Exception:
                pass          # 单次采样失败不该让监控线程死掉

    def _sample(self):
        cpu = psutil.cpu_percent(interval=None)          # 距上次调用≈1s 的均值
        try:
            cpu_proc = self._proc.cpu_percent(interval=None)
        except Exception:
            cpu_proc = 0.0
        mem = psutil.virtual_memory()
        disk = {}
        for d in ("C:\\", "D:\\"):
            try:
                u = psutil.disk_usage(d)
                disk[d[0]] = (u.percent, u.used / 1024 ** 3, u.total / 1024 ** 3)
            except Exception:
                pass
        io_pct = {}
        try:
            now_io = psutil.disk_io_counters()
            t1 = time.time()
            if self._io_prev is not None and self._io_ts:
                span = max(1e-6, t1 - self._io_ts)
                r = (now_io.read_bytes - self._io_prev.read_bytes)
                w = (now_io.write_bytes - self._io_prev.write_bytes)
                mbps = (r + w) / span / 1024 ** 2
                pct = min(100.0, mbps / _IO_FULL_RATE_MBPS * 100.0)
                io_pct["ALL"] = pct
                io_pct["_total_mbps"] = mbps
            self._io_prev = now_io
            self._io_ts = t1
        except Exception:
            pass
        gpu = self._gpu.read()
        with self.lock:
            self.data.update({"cpu": cpu, "cpu_proc": cpu_proc,
                              "mem_used": mem.used / 1024 ** 3,
                              "mem_total": mem.total / 1024 ** 3,
                              "mem_pct": mem.percent,
                              "disk": disk, "io_pct": io_pct, "gpu": gpu})

    def snapshot(self):
        with self.lock:
            d = dict(self.data)
            d["disk"] = dict(d["disk"])
            d["io_pct"] = dict(d["io_pct"])
            return d


class _GpuProbe:
    """GPU 显存：优先 nvml（独显），退到 Windows 性能计数器（核显共享显存）。"""

    def __init__(self):
        self.mode = "none"
        self.nvml = None
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.mode = "nvml"
            return
        except Exception:
            pass
        # 核显：Windows 性能计数器
        try:
            import ctypes
            from ctypes import wintypes

            class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
                _fields_ = [("CStatus", wintypes.DWORD),
                            ("padding", wintypes.DWORD),
                            ("value", ctypes.c_double)]

            advapi = ctypes.WinDLL("advapi32")
            pdh = ctypes.WinDLL("pdh")
            advapi.OpenProcessToken.restype = wintypes.HANDLE
            self._pdh = pdh
            self._fmt = _PDH_FMT_COUNTERVALUE
            self.query = None
            # 只需要 OpenQuery/AddCounter/CollectQueryData，可在首次 read 时惰性建立
            self._counters = []
            self._ok = self._init_counters()
            if self._ok:
                self.mode = "pdh"
        except Exception:
            self.mode = "none"

    def _init_counters(self):
        """用 Windows 的 `GPU Adapter Memory` 计数器取**共享显存**（核显）。"""
        try:
            import ctypes
            from ctypes import wintypes
            pdh = self._pdh

            class PDH_HQUERY(ctypes.Structure):
                _fields_ = [("ptr", ctypes.c_void_p)]
            PDH_HQUERY.__fields_ = [("ptr", ctypes.c_void_p)]

            class PDH_HCOUNTER(ctypes.Structure):
                _fields_ = [("ptr", ctypes.c_void_p)]

            q = PDH_HQUERY()
            rc = pdh.PdhOpenQueryW(None, 0, ctypes.byref(q))
            if rc != 0:
                return False
            self._query = q
            for path in (r"GPU Adapter Memory(*)\Shared Usage",
                         r"GPU Adapter Memory(*)\Dedicated Usage",
                         r"GPU Engine(*)\Utilization Percentage"):
                c = PDH_HCOUNTER()
                rc = pdh.PdhAddEnglishCounterW(q, path, 0, ctypes.byref(c))
                if rc == 0:
                    self._counters.append((path, c))
            if not self._counters:
                return False
            pdh.PdhCollectQueryData(q)
            time.sleep(0.2)
            pdh.PdhCollectQueryData(q)
            return True
        except Exception:
            return False

    def _pdh_read(self):
        try:
            import ctypes          # 本函数原来直接用 ctypes 但作用域里没有（被 except 吞成"读不到"）
            out = {}
            self._pdh.PdhCollectQueryData(self._query)
            for path, c in self._counters:
                val = self._fmt()
                rc = self._pdh.PdhGetFormattedCounterValue(
                    c, 0x400, None, ctypes.byref(val))
                if rc == 0 or rc == 1:
                    out[path] = val.value
            return out
        except Exception:
            return {}

    def read(self):
        if self.mode == "nvml":
            try:
                info = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                return {"used_gb": info.used / 1024 ** 3,
                        "total_gb": info.total / 1024 ** 3,
                        "src": "nvml"}
            except Exception:
                return None
        if self.mode == "pdh":
            # 实测（2026-10-15，本机 Intel HD 530）：`PdhExpandWildCardPath("\\Machine\\GPU*")`
            # 返回 0xC0000BC4（对象不存在）⇒ 这台机器**没有注册 GPU 性能计数器**，
            # 任务管理器那个 "0.2/15.9GB" 来自 WDDM/DXGI，不是 PDH。
            # ⇒ 本机只能显示"GPU:不可读"；换到注册了这些计数器的机器上本分支自动生效。
            vals = self._pdh_read()
            shared = 0.0
            shared_key = r"GPU Adapter Memory(*)\Shared Usage"
            dedicated = 0.0
            dedicated_key = r"GPU Adapter Memory(*)\Dedicated Usage"
            for path, v in vals.items():
                if path.endswith("Shared Usage"):
                    shared = max(shared, v)
                elif path.endswith("Dedicated Usage"):
                    dedicated = max(dedicated, v)
            if shared <= 0 and dedicated <= 0:
                return None
            used = max(shared, dedicated) / 1024 ** 3
            # 核显总显存 Windows 不直接给，用 1024MB 的整数倍做保守上限仅用于百分比
            return {"used_gb": used, "total_gb": 0.0, "src": "pdh",
                    "shared_gb": shared / 1024 ** 3,
                    "dedicated_gb": dedicated / 1024 ** 3}
        return None


class SystemMonitor(QLabel):
    """Status bar widget：CPU(整机+进程) / GPU / 内存 / 磁盘(空间 + I/O 活动率)。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QLabel { font-family: Consolas; font-size: 15px; padding: 0 6px; }")
        self._sampler = _Sampler()
        self._sampler.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self._refresh()

    def _refresh(self):
        d = self._sampler.snapshot()
        parts = []

        # CPU：整机 + 本进程（训练时真正关心的是后者）
        cpu = d["cpu"]
        cpu_txt = "CPU:{:.0f}%".format(cpu)
        if cpu >= 90:
            cpu_txt = "<span style='color:#ff4444;font-weight:bold'>" + cpu_txt + "</span>"
        parts.append(cpu_txt)
        cp = d["cpu_proc"]
        cp_txt = "进程:{:.0f}%".format(cp)
        if cp >= 90:
            cp_txt = "<span style='color:#ff4444;font-weight:bold'>" + cp_txt + "</span>"
        parts.append(cp_txt)

        # GPU：nvml(独显) 或 Windows 计数器(核显)；读不到就**明说**
        g = d.get("gpu")
        if g:
            used = g["used_gb"]
            if g.get("src") == "nvml":
                total = g.get("total_gb", 0.0)
                pct = (used / total * 100) if total > 0 else 0.0
                txt = "GPU:{:.0f}%（{:.2f}/{:.2f}GB）".format(pct, used, total)
            else:
                txt = "GPU:{:.2f}GB（共享显存）".format(used)
            if used >= 0.9 * (g.get("total_gb") or 1e9):
                txt = "<span style='color:#ff4444;font-weight:bold'>" + txt + "</span>"
            parts.append(txt)
        else:
            parts.append("<span style='color:#8A93A6'>GPU:不可读</span>")

        # 内存（口径与任务管理器一致）
        mp = d["mem_pct"]
        mtxt = "内存:{:.2f}/{:.2f}GB".format(d["mem_used"], d["mem_total"])
        if mp >= 90:
            mtxt = "<span style='color:#ff4444;font-weight:bold'>" + mtxt + "</span>"
        parts.append(mtxt)

        # 磁盘：**空间已用率**与**I/O 活动率**分开写，别再同名不同义
        io_all = d["io_pct"].get("ALL")
        for k, (pct, used, total) in sorted(d["disk"].items()):
            txt = "{}: 空间{:.0f}%（{:.0f}/{:.0f}GB）".format(k, pct, used, total)
            if pct >= 90:
                txt = "<span style='color:#ff4444;font-weight:bold'>" + txt + "</span>"
            parts.append(txt)
        if io_all is not None:
            itxt = "IO {:.0f}%".format(io_all)
            if io_all >= 80:
                itxt = "<span style='color:#ff4444;font-weight:bold'>" + itxt + "</span>"
            parts.append(itxt)

        self.setText(" | ".join(parts))

    def stop(self):
        self._timer.stop()
        self._sampler.stop()
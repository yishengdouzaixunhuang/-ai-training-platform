"""Real-time system monitor widget for status bar."""
import psutil
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer


class SystemMonitor(QLabel):
    """Status bar widget showing CPU/GPU/Memory/Disk usage."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("QLabel { font-family: Consolas; font-size: 15px; padding: 0 6px; }")
        self._gpu_available = False
        self._init_gpu()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(1000)
        self._refresh()

    def _init_gpu(self):
        """Try pynvml first, fall back to torch.cuda."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_available = True
            self._use_nvml = True
            return
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                self._gpu_available = True
                self._use_torch = True
                self._gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                return
        except Exception:
            pass
        self._gpu_available = False

    @staticmethod
    def _fmt(val, suffix, warn_threshold=90):
        """Format value with red coloring if >= warn_threshold."""
        if val >= warn_threshold:
            return f"<span style='color:#ff4444;font-weight:bold'>{val:.2f}{suffix}</span>"
        return f"{val:.2f}{suffix}"

    def _refresh(self):
        parts = []

        # CPU
        cpu = psutil.cpu_percent(interval=None)
        parts.append(self._fmt(cpu, "%", 90).replace("%", "%</span>") if cpu >= 90 else f"CPU:{cpu:.2f}%")
        # Fix the span wrapping properly
        if cpu >= 90:
            parts[-1] = f"CPU:<span style='color:#ff4444;font-weight:bold'>{cpu:.2f}%</span>"
        else:
            parts[-1] = f"CPU:{cpu:.2f}%"

        # GPU
        if self._gpu_available:
            try:
                if hasattr(self, "_use_nvml"):
                    info = self._nvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                    used = info.used / 1024**3
                    total = info.total / 1024**3
                    if util.gpu >= 90:
                        parts.append(f"GPU:<span style='color:#ff4444;font-weight:bold'>{util.gpu:.2f}%</span>\uff08{used:.2f}/{total:.2f}GB)")
                    else:
                        parts.append(f"GPU:{util.gpu:.2f}%\uff08{used:.2f}/{total:.2f}GB)")
                elif hasattr(self, "_use_torch"):
                    import torch
                    used = torch.cuda.memory_allocated(0) / 1024**3
                    mem_pct = (used / self._gpu_total) * 100 if self._gpu_total > 0 else 0
                    if mem_pct >= 90:
                        parts.append(f"GPU:<span style='color:#ff4444;font-weight:bold'>--</span>\uff08{used:.2f}/{self._gpu_total:.2f}GB)")
                    else:
                        parts.append(f"GPU:--\uff08{used:.2f}/{self._gpu_total:.2f}GB)")
            except Exception:
                parts.append("GPU:--")

        # Memory
        mem = psutil.virtual_memory()
        used = mem.used / 1024**3
        total = mem.total / 1024**3
        mem_pct = mem.percent
        if mem_pct >= 90:
            parts.append(f"\u5185\u5b58\uff1a<span style='color:#ff4444;font-weight:bold'>{used:.2f}/{total:.2f}GB</span>")
        else:
            parts.append(f"\u5185\u5b58\uff1a{used:.2f}/{total:.2f}GB")

        # Disks
        for drive in ["C:", "D:"]:
            try:
                pct = psutil.disk_usage(drive + "\\").percent
                if pct >= 90:
                    parts.append(f"{drive}<span style='color:#ff4444;font-weight:bold'>{pct}%</span>")
                else:
                    parts.append(f"{drive}{pct}%")
            except Exception:
                pass

        self.setText(" | ".join(parts))

    def stop(self):
        self._timer.stop()
        try:
            if hasattr(self, "_nvml"):
                self._nvml.nvmlShutdown()
        except Exception:
            pass

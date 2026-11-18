"""Heatmap Region Analyzer — batch analyze Grad-CAM / class-activation heatmaps.

Tools → Heatmap Region Analyzer... 入口。

设计要点：
- 分析、报告渲染全部在后台线程（threading.Thread，daemon），主线程只更新 UI
- 信号经 _Signals(QObject) 从 Python 线程 emit（与 main_window 的 log_signal 模式一致）
- 单图异常不中断；支持中途取消；关闭对话框时安全停止
"""
from core.i18n import tr
import os
from datetime import datetime

import threading
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
    QListWidget, QProgressBar, QTextEdit, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QWidget, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QFont


# 延迟导入：UI 启动时不一定需要 numpy/PIL，运行时才需要
def _analyze(*args, **kwargs):
    from tools.heatmap_analyzer import analyze_folder
    return analyze_folder(*args, **kwargs)


def _render_report(*args, **kwargs):
    from tools.heatmap_analyzer import render_html_report
    return render_html_report(*args, **kwargs)


class _Signals(QObject):
    """跨线程信号桥（worker 线程 emit → 主线程槽）。"""
    progress = pyqtSignal(int, int, str, str)   # done, total, phase, name
    finished = pyqtSignal(dict)                 # result dict（分析结果）
    render_done = pyqtSignal(str)               # 报告路径
    failed = pyqtSignal(str)                    # 错误消息
    status = pyqtSignal(str)                    # 任意状态日志


class _AnalysisWorker:
    """后台线程跑分析（threading.Thread，daemon，可取消）。"""

    def __init__(self, folder, recursive, threshold, min_area,
                 grid_w, grid_h, max_dim):
        self.signals = _Signals()
        self.folder = folder
        self.recursive = recursive
        self.threshold = threshold
        self.min_area = min_area
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.max_dim = max_dim
        self._cancel = False
        self._render_cancel = False
        self._result = None
        self._thread = None
        self._render_thread = None

    def cancel(self):
        self._cancel = True

    def cancel_render(self):
        self._render_cancel = True

    def is_busy(self):
        return (self._thread and self._thread.is_alive()) or \
               (self._render_thread and self._render_thread.is_alive())

    def start(self):
        self._cancel = False
        self._thread = threading.Thread(target=self._run_analyze, daemon=True)
        self._thread.start()

    # ── 分析阶段 ──
    def _run_analyze(self):
        try:
            result = _analyze(
                folder=self.folder,
                recursive=self.recursive,
                threshold=self.threshold,
                min_area=self.min_area,
                grid_w=self.grid_w,
                grid_h=self.grid_h,
                max_dim=self.max_dim,
                progress_cb=lambda d, t, n: self._safe_emit(d, t, 'scan', n),
                cancel_cb=lambda: self._cancel,
            )
            self._result = result
            self.signals.finished.emit(result)
        except Exception as e:  # noqa: BLE001
            import traceback
            self._safe_fail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def _safe_emit(self, d, t, phase, name):
        """后台线程安全 emit（信号已断开时忽略，避免线程内抛异常）。"""
        try:
            self.signals.progress.emit(d, t, phase, name)
        except Exception:
            pass

    def _safe_fail(self, msg):
        try:
            self.signals.failed.emit(msg)
        except Exception:
            pass

    # ── 渲染阶段（分析完成后单独起线程，避免阻塞主线程 10+ 秒） ──
    def start_render(self, output_dir):
        self._render_cancel = False
        self._render_thread = threading.Thread(
            target=self._run_render, args=(output_dir,), daemon=True)
        self._render_thread.start()

    def _run_render(self, output_dir):
        try:
            result = self._result
            if result is None:
                self._safe_fail("无分析结果，无法渲染报告。")
                return
            report = _render_report(
                result, output_dir,
                progress_cb=lambda d, t, n: self._safe_emit(d, t, 'render', ''),
                cancel_cb=lambda: self._render_cancel,
            )
            self.signals.render_done.emit(report)
        except Exception as e:  # noqa: BLE001
            import traceback
            self._safe_fail(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class HeatmapAnalyzerDialog(QDialog):
    """批量热力图区域分析对话框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Heatmap Region Analyzer · 热力图区域分析"))
        self.setMinimumSize(960, 640)
        self._result: dict = {}
        self._worker: _AnalysisWorker = None
        self._init_ui()

    # ────────────────── UI 构造 ──────────────────

    def _init_ui(self):
        root = QVBoxLayout(self)

        # ===== 顶部：输入 + 参数 =====
        top = QHBoxLayout()
        left_box = QGroupBox(tr("输入"))
        lv = QVBoxLayout(left_box)

        # 文件夹
        h1 = QHBoxLayout()
        h1.addWidget(QLabel(tr("热力图文件夹:")))
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText(tr("选择文件夹…"))
        h1.addWidget(self._folder_edit)
        b1 = QPushButton(tr("浏览…"))
        b1.clicked.connect(self._choose_folder)
        h1.addWidget(b1)
        lv.addLayout(h1)

        # 输出
        h2 = QHBoxLayout()
        h2.addWidget(QLabel(tr("报告输出目录:")))
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(tr("默认 = 输入文件夹/_heatmap_report_时间戳"))
        h2.addWidget(self._out_edit)
        b2 = QPushButton(tr("浏览…"))
        b2.clicked.connect(self._choose_output)
        h2.addWidget(b2)
        lv.addLayout(h2)

        # 选项
        opts = QHBoxLayout()
        self._chk_recurse = QCheckBox(tr("递归子文件夹"))
        opts.addWidget(self._chk_recurse)
        self._chk_auto = QCheckBox(tr("自动 OTSU 阈值"))
        self._chk_auto.setChecked(True)
        opts.addWidget(self._chk_auto)
        opts.addStretch()
        lv.addLayout(opts)
        top.addWidget(left_box, stretch=2)

        # 参数
        param_box = QGroupBox(tr("参数"))
        pg = QFormLayout(param_box)
        self._spin_thresh = QDoubleSpinBox()
        self._spin_thresh.setRange(0.05, 0.95)
        self._spin_thresh.setSingleStep(0.05)
        self._spin_thresh.setValue(0.4)
        self._spin_thresh.setEnabled(False)
        pg.addRow(tr("固定阈值:"), self._spin_thresh)
        self._chk_auto.toggled.connect(self._spin_thresh.setDisabled)

        self._spin_min_area = QSpinBox()
        self._spin_min_area.setRange(5, 100000)
        self._spin_min_area.setValue(40)
        pg.addRow(tr("最小区域像素:"), self._spin_min_area)

        gw = QHBoxLayout()
        self._spin_grid_w = QSpinBox(); self._spin_grid_w.setRange(2, 20); self._spin_grid_w.setValue(8)
        gw.addWidget(self._spin_grid_w)
        gw.addWidget(QLabel("×"))
        self._spin_grid_h = QSpinBox(); self._spin_grid_h.setRange(2, 20); self._spin_grid_h.setValue(3)
        gw.addWidget(self._spin_grid_h)
        gw_w = QWidget(); gw_w.setLayout(gw)
        pg.addRow(tr("聚类网格(W×H):"), gw_w)

        self._spin_max_dim = QSpinBox()
        self._spin_max_dim.setRange(64, 8192)
        self._spin_max_dim.setValue(1200)
        self._spin_max_dim.setSingleStep(100)
        self._spin_max_dim.setSpecialValueText("原始尺寸")
        pg.addRow(tr("缩放上限(像素):"), self._spin_max_dim)

        top.addWidget(param_box, stretch=1)
        root.addLayout(top)

        # ===== 中部：分隔（左侧文件列表 / 右侧结果表） =====
        mid = QSplitter(Qt.Horizontal)

        # 文件列表
        left_mid = QWidget(); ll = QVBoxLayout(left_mid); ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel(tr("📁 处理队列（实时刷新）")))
        self._file_list = QListWidget()
        ll.addWidget(self._file_list)
        ll.addWidget(QLabel(tr("运行日志:")))
        self._log = QTextEdit(); self._log.setReadOnly(True)
        self._log.setMaximumHeight(180)
        ll.addWidget(self._log)
        mid.addWidget(left_mid)

        # 结果表（Tab：详情 / 聚类汇总）
        self._tabs = QTabWidget()
        # 详情
        self._tbl_files = QTableWidget(0, 5)
        self._tbl_files.setHorizontalHeaderLabels(
            [tr("文件名"), tr("尺寸"), tr("高亮数"), tr("标签"), tr("高亮位置")]
        )
        self._tbl_files.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._tbl_files.horizontalHeader().setStretchLastSection(True)
        self._tbl_files.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl_files.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl_files.setColumnWidth(0, 240)
        self._tbl_files.setColumnWidth(1, 80)
        self._tbl_files.setColumnWidth(2, 60)
        self._tbl_files.setColumnWidth(3, 200)
        self._tabs.addTab(self._tbl_files, tr("每图详情"))

        # 聚类
        self._tbl_clusters = QTableWidget(0, 4)
        self._tbl_clusters.setHorizontalHeaderLabels(
            [tr("类#"), tr("网格坐标"), tr("归一化中心"), tr("样本数")]
        )
        self._tbl_clusters.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._tbl_clusters.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl_clusters.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tabs.addTab(self._tbl_clusters, tr("聚类汇总 (=NG 种类估计)"))
        mid.addWidget(self._tabs)
        mid.setStretchFactor(0, 1)
        mid.setStretchFactor(1, 2)
        root.addWidget(mid, stretch=1)

        # ===== 底部：进度 + 按钮 =====
        bot = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        bot.addWidget(self._progress, stretch=1)
        self._btn_run = QPushButton(tr("▶ 运行分析"))
        self._btn_run.clicked.connect(self._run)
        bot.addWidget(self._btn_run)
        self._btn_cancel = QPushButton(tr("■ 取消"))
        self._btn_cancel.setEnabled(False)
        self._btn_cancel.clicked.connect(self._cancel_run)
        bot.addWidget(self._btn_cancel)
        self._btn_export = QPushButton(tr("💾 导出 HTML 报告"))
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._export_report)
        bot.addWidget(self._btn_export)
        self._btn_close = QPushButton(tr("关闭"))
        self._btn_close.clicked.connect(self.close)
        bot.addWidget(self._btn_close)
        root.addLayout(bot)

    # ────────────────── 槽函数 ──────────────────

    def _choose_folder(self):
        d = QFileDialog.getExistingDirectory(self, tr("选择热力图文件夹"))
        if d:
            self._folder_edit.setText(d)
            if not self._out_edit.text():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._out_edit.setText(os.path.join(d, f"_heatmap_report_{ts}"))

    def _choose_output(self):
        d = QFileDialog.getExistingDirectory(self, tr("选择报告输出目录"))
        if d:
            self._out_edit.setText(d)

    def _log_append(self, msg: str):
        try:
            self._log.append(msg)
            sb = self._log.verticalScrollBar()
            sb.setValue(sb.maximum())
        except RuntimeError:
            pass  # 对话框已销毁

    def _run(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, tr("提示"), tr("请选择有效的热力图文件夹。"))
            return
        out = self._out_edit.text().strip()
        if not out:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = os.path.join(folder, f"_heatmap_report_{ts}")
            self._out_edit.setText(out)
        if self._worker and self._worker.is_busy():
            QMessageBox.information(self, tr("提示"), tr("分析或报告渲染正在进行中…"))
            return

        # 重置 UI
        self._btn_run.setEnabled(False)
        self._btn_export.setEnabled(False)
        self._btn_cancel.setEnabled(True)
        self._file_list.clear()
        self._log.clear()
        self._tbl_files.setRowCount(0)
        self._tbl_clusters.setRowCount(0)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        params = dict(
            folder=folder,
            recursive=self._chk_recurse.isChecked(),
            threshold=None if self._chk_auto.isChecked() else float(self._spin_thresh.value()),
            min_area=int(self._spin_min_area.value()),
            grid_w=int(self._spin_grid_w.value()),
            grid_h=int(self._spin_grid_h.value()),
            max_dim=int(self._spin_max_dim.value()) if self._spin_max_dim.value() > 64 else None,
        )
        self._log_append(f"[启动] {datetime.now().strftime('%H:%M:%S')}")
        self._log_append(f"[参数] {params}")

        self._worker = _AnalysisWorker(
            folder=params['folder'],
            recursive=params['recursive'],
            threshold=params['threshold'],
            min_area=params['min_area'],
            grid_w=params['grid_w'],
            grid_h=params['grid_h'],
            max_dim=params['max_dim'],
        )
        # 连接信号（主线程槽）
        sig = self._worker.signals
        sig.progress.connect(self._on_progress)
        sig.finished.connect(self._on_finished)
        sig.render_done.connect(self._on_render_done)
        sig.failed.connect(self._on_failed)
        self._worker.start()

    def _cancel_run(self):
        if self._worker is None:
            return
        self._worker.cancel()
        self._worker.cancel_render()
        self._btn_cancel.setEnabled(False)
        self._log_append("[取消] 正在停止…（当前图片处理完即停）")

    def _on_progress(self, done: int, total: int, phase: str, name: str):
        try:
            self._progress.setMaximum(max(total, 1))
            self._progress.setValue(done)
            if phase == 'scan':
                # 限制列表显示条数，避免上万条 item 卡 UI
                if self._file_list.count() >= 400:
                    self._file_list.clear()
                self._file_list.addItem(f"[{done}/{total}] {name}")
                if done == total:
                    self._log_append(f"[完成] 扫描 {total} 张图")
            elif phase == 'render':
                self._log_append(f"[渲染] 标注图 {done}/{total}")
        except RuntimeError:
            pass  # 对话框已销毁

    def _on_finished(self, result: dict):
        if not result:
            return
        self._result = result
        s = result['summary']
        self._log_append(
            f"[结果] 总图 {s['total']} / 含高亮 {s['with_hotspots']} / "
            f"总区域 {s['n_hotspots']} / 聚类 {s['n_clusters']} / "
            f"失败 {s.get('errors', 0)} / 取消 {s.get('cancelled', False)}"
        )
        self._populate_tables(result)
        self._btn_cancel.setEnabled(True)  # 允许中断渲染
        # 渲染报告放后台线程
        out = self._out_edit.text().strip() or \
            os.path.join(result['params']['folder'], '_heatmap_report')
        try:
            os.makedirs(out, exist_ok=True)
            self._log_append(f"[渲染] 正在生成标注图与报告 → {out}")
            self._worker.start_render(out)
        except Exception as e:  # noqa: BLE001
            self._log_append(f"[渲染启动失败] {e}")
            self._btn_run.setEnabled(True)
            self._btn_export.setEnabled(True)
            self._btn_cancel.setEnabled(False)

    def _on_render_done(self, report_path: str):
        self._btn_run.setEnabled(True)
        self._btn_export.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        self._progress.setValue(self._progress.maximum())
        self._log_append(f"[报告] 已生成: {report_path}")

    def _on_failed(self, err: str):
        self._btn_run.setEnabled(True)
        self._btn_cancel.setEnabled(False)
        # 有分析结果时仍允许导出报告
        self._btn_export.setEnabled(bool(self._result))
        self._log_append(f"[失败] {err}")

    def _populate_tables(self, result: dict):
        # 详情表
        files = result['files']
        self._tbl_files.setRowCount(len(files))
        for i, f in enumerate(files):
            self._tbl_files.setItem(i, 0, QTableWidgetItem(f['filename']))
            size_str = '×'.join(str(x) for x in f.get('size', ())) if 'size' in f else '-'
            self._tbl_files.setItem(i, 1, QTableWidgetItem(size_str))
            n_item = QTableWidgetItem(str(f.get('nhotspots', 0)))
            if f.get('nhotspots', 0) > 0:
                n_item.setForeground(QColor('#c0392b'))
            self._tbl_files.setItem(i, 2, n_item)
            tags = f.get('tags', {})
            tag_str = ' '.join(f"{k}:{v}" for k, v in tags.items()) or '-'
            self._tbl_files.setItem(i, 3, QTableWidgetItem(tag_str))
            if 'error' in f:
                spots_str = f"[ERROR] {f['error']}"
            else:
                spots = f.get('hotspots', [])
                spots_str = ' | '.join(
                    f"#{k+1} ({h['cx']:.0f},{h['cy']:.0f}) {h['area']}px {h['mean_intensity']:.2f}"
                    for k, h in enumerate(spots)
                ) or '-'
            self._tbl_files.setItem(i, 4, QTableWidgetItem(spots_str))

        # 聚类表
        clusters = result['clusters']
        self._tbl_clusters.setRowCount(len(clusters))
        for i, c in enumerate(clusters):
            self._tbl_clusters.setItem(i, 0, QTableWidgetItem(f"#{c['cluster_id']}"))
            self._tbl_clusters.setItem(i, 1, QTableWidgetItem(f"({c['cell'][0]}, {c['cell'][1]})"))
            nc = c['norm_center']
            self._tbl_clusters.setItem(i, 2, QTableWidgetItem(f"({nc[0]:.2f}, {nc[1]:.2f})"))
            cnt = QTableWidgetItem(str(c['count']))
            cnt.setForeground(QColor('#c0392b'))
            cnt.setFont(QFont("", -1, QFont.Bold))
            self._tbl_clusters.setItem(i, 3, cnt)

    def _export_report(self):
        if not self._result or not self._worker:
            QMessageBox.information(self, tr("提示"), tr("请先运行分析。"))
            return
        if self._worker.is_busy():
            QMessageBox.information(self, tr("提示"), tr("报告渲染正在进行中，请稍候…"))
            return
        d = QFileDialog.getExistingDirectory(self, tr("选择报告输出目录"))
        if not d:
            return
        try:
            self._btn_export.setEnabled(False)
            self._btn_cancel.setEnabled(True)
            self._log_append("[渲染] 正在导出报告…")
            self._worker.start_render(d)
        except Exception as e:  # noqa: BLE001
            self._btn_export.setEnabled(True)
            self._btn_cancel.setEnabled(False)
            QMessageBox.critical(self, tr("导出失败"), str(e))

    def closeEvent(self, event):
        """关闭时设置取消标志；后台线程为 daemon，随进程退出自然结束。

        不强制 disconnect 信号：PyQt5 会在接收对象销毁时自动移除连接，
        强制 disconnect 反而可能在后台线程 emit 时报错。
        """
        if self._worker is not None:
            self._worker.cancel()
            self._worker.cancel_render()
        super().closeEvent(event)


__all__ = ['HeatmapAnalyzerDialog']
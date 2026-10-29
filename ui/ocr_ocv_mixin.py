# -*- coding: utf-8 -*-
"""OCR 文字识别 + OCV 字符质检 —— 从 MainWindow 拆出的方法集合。

为什么单独放一个 Mixin：
    `ui/main_window.py` 曾经接近 1 万行 / 300 多个方法，从一个按钮找不到
    它到底连到哪段逻辑。OCR 与 OCV 是两条相对独立的功能线（前者跑 PP-OCR，
    后者比对字符特征向量），彼此不依赖，也不参与分割/检测/分类的主流程，
    属于最安全的切分点。

约定：
    - 用 Mixin 而不是独立 Controller，是为了让所有 self.xxx 引用继续成立，
      零改动搬运、行为不变（这是重构里风险最低的做法）。
    - 这些方法仍依赖 MainWindow 上的控件：ocr_status / ocr_progress /
      ocv_status / ocv_threshold / ocv_result_label / canvas 等。
    - 线程内更新 UI 一律走 ui_call（见 ui/qt_thread.py）。
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback

import numpy as np
from PIL import Image
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMenu, QTableWidgetItem

from ui.qt_thread import ui_call


class OcrOcvMixin:
    """OCR / OCV 面板逻辑。由 MainWindow 继承后即可用。"""

    # ===== OCR Methods =====

    def _run_ocr_single(self):
        """Run OCR on the current image."""
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path or not os.path.exists(current_path):
            return
        self.ocr_status.setText("Running...")
        def run():
            try:
                from ocr.engine import get_ocr_engine
                engine = get_ocr_engine()
                results = engine.detect_and_recognize(current_path)
                self._current_ocr_results = results
                # Also cache for image-switch
                if not hasattr(self, "_ocr_results_cache"):
                    self._ocr_results_cache = {}
                self._ocr_results_cache[os.path.basename(current_path)] = results
                # Update image list columns
                if not hasattr(self, "_ocr_predictions"):
                    self._ocr_predictions = {}
                fname = os.path.basename(current_path)
                if results:
                    self._ocr_predictions[fname] = {"text": results[0].get("text", ""), "confidence": results[0].get("score", 0)}
                self.log_signal.emit(f"OCR: {len(results)} text regions found")
                self.refresh_list_signal.emit()
                # 表格填充和画布叠加都是 GUI 操作，回主线程做
                ui_call(self._update_ocr_table, results)
                ui_call(self._update_ocr_overlay)
            except Exception as e:
                self.log_signal.emit(f"OCR error: {e}")
            finally:
                ui_call(self.ocr_status.setText, "Ready")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _run_ocr_batch(self):
        """Run batch OCR on all images."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        out_dir = os.path.join(pd, "outputs", "ocr")
        self.ocr_status.setText("Batch OCR running...")
        self.ocr_progress.setVisible(True)
        def run():
            import time as _time
            try:
                from flow.batch import BatchRunner, build_ocr_flow, list_images
                images = list_images(pd)
                if not images:
                    self.log_signal.emit("No images found")
                    return
                flow = build_ocr_flow(out_dir=out_dir)
                t0 = _time.time()
                results = {}
                def on_progress(i, total, path, frame, err):
                    fname = os.path.basename(path)
                    if err is None:
                        results[fname] = frame.result.get("regions", [])
                        self.log_signal.emit("[%d/%d] %s: %d regions" % (i, total, fname, frame.result.get("count", 0)))
                    else:
                        results[fname] = {"error": str(err)}
                        self.log_signal.emit("[%d/%d] %s: FAILED - %s" % (i, total, fname, err))
                    ui_call(self.ocr_progress.setMaximum, total)
                    ui_call(self.ocr_progress.setValue, i)
                br = BatchRunner(flow, project_dir=pd, stop_check=lambda: self._stop_flag)
                br.run(images, on_progress=on_progress)
                # 保存批量推理结果到 ocr_results.json 文件
                import json as _json
                json_path = os.path.join(out_dir, "ocr_results.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    _json.dump(results, f, ensure_ascii=False, indent=2)
                self._ocr_results_cache = results  # Store for image-switch auto-load
                # Populate image list columns
                self._ocr_predictions = {}
                self._ocr_times = getattr(self, "_ocr_times", {})
                for fname, items in results.items():
                    if isinstance(items, list) and len(items) > 0:
                        self._ocr_predictions[fname] = {
                            "text": items[0].get("text", ""),
                            "confidence": items[0].get("score", 0),
                        }
                total_elapsed = _time.time() - t0
                total = sum(1 for v in results.values() if isinstance(v, list) and len(v) > 0)
                avg_time = total_elapsed / max(total, 1) if total > 0 else 0
                for fname in results:
                    if isinstance(results[fname], list) and len(results[fname]) > 0:
                        if avg_time > 0:
                            self._ocr_times[fname] = f"{avg_time:.2f}s"
                self.log_signal.emit(f"Batch OCR complete: {total} images with text")
                self.refresh_list_signal.emit()
                # Load results for current image
                current_path = getattr(self.canvas, "current_image_path", "")
                if current_path:
                    fname = os.path.basename(current_path)
                    if fname in results and isinstance(results[fname], list):
                        self._current_ocr_results = results[fname]
                        ui_call(self._update_ocr_table, results[fname])
                        ui_call(self._update_ocr_overlay)
            except Exception as e:
                import traceback
                self.log_signal.emit(f"Batch OCR error: {e}")
            finally:
                ui_call(self.ocr_status.setText, "Ready")
                ui_call(self.ocr_progress.setVisible, False)
        import threading
        threading.Thread(target=run, daemon=True).start()
    def _update_ocr_table(self, results):
        """Update OCR result table."""
        self.ocr_result_table.setRowCount(len(results))
        def _ro(txt):
            it = QTableWidgetItem(txt)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)  # 只读
            return it
        for i, r in enumerate(results):
            self.ocr_result_table.setItem(i, 0, _ro(str(i+1)))
            text_item = QTableWidgetItem(r.get("text", ""))
            # Text 列允许选中/编辑，便于鼠标选中文本后 Ctrl+C 或右键复制
            text_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
            self.ocr_result_table.setItem(i, 1, text_item)
            self.ocr_result_table.setItem(i, 2, _ro(f"{r.get('score', 0):.3f}"))
            self.ocr_result_table.setItem(i, 3, _ro(f"({r.get('x',0)},{r.get('y',0)})"))
            self.ocr_result_table.setItem(i, 4, _ro(f"{r.get('w',0)}x{r.get('h',0)}"))
        self.ocr_result_table.resizeColumnsToContents()

    def _on_ocr_result_menu(self, pos):
        """OCR 结果表格右键菜单：复制当前行 Text 列文本到剪贴板。"""
        table = self.ocr_result_table
        item = table.itemAt(pos)
        if item is None:
            return
        row = item.row()
        text_item = table.item(row, 1)
        text = text_item.text() if text_item else ""
        menu = QMenu(self)
        act_copy = menu.addAction("复制 Text")
        act_copy.setEnabled(bool(text))
        chosen = menu.exec_(table.viewport().mapToGlobal(pos))
        if chosen is act_copy and text:
            QApplication.clipboard().setText(text)
            brief = text if len(text) <= 40 else text[:40] + "..."
            self.log_signal.emit(f"已复制 OCR 文本: {brief}")

    def _update_ocr_overlay(self):
        """Draw OCR results on the canvas."""
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path:
            return
        results = getattr(self, "_current_ocr_results", None)
        if not results or not self.ocr_show_check.isChecked():
            self.canvas._ocr_overlay = None
            self.canvas.update()
            return
        from PIL import Image
        from ocr.pipeline import draw_ocr_overlay
        import numpy as np
        from PyQt5.QtGui import QImage
        img = Image.open(current_path).convert("RGB")
        overlay = draw_ocr_overlay(np.array(img).copy(), results)
        h, w = overlay.height, overlay.width
        qimg = QImage(np.array(overlay).data, w, h, 3 * w, QImage.Format_RGB888)
        self.canvas._ocr_overlay = qimg.copy()
        self.canvas.update()

    def _export_ocr_csv(self):
        """Export OCR results to CSV."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        out = os.path.join(pd, "outputs", "ocr", "ocr_results.csv")
        if os.path.exists(out):
            os.startfile(os.path.dirname(out))
            self.log_signal.emit(f"Opened: {out}")
        else:
            self.log_signal.emit("No OCR results yet. Run batch OCR first.")

    # ===== OCV Methods =====

    def _build_ocv_model(self):
        """Build OCV feature bank from OK character images."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        ok_dir = os.path.join(pd, "ocv", "ok_samples")
        if not os.path.exists(ok_dir):
            os.makedirs(ok_dir, exist_ok=True)
            self.log_signal.emit(f"Created {ok_dir}. Place OK character crops there, then retry.")
            return
        out_path = os.path.join(pd, "models", "ocv_model.pkl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.ocv_status.setText("Building OCV model...")
        def run():
            try:
                from ocv.inspector import train_ocv_model
                inspector, n = train_ocv_model(ok_dir, out_path)
                self._ocv_inspector = inspector
                self.log_signal.emit(f"OCV model built: {n} OK samples, threshold={inspector.threshold:.2f}")
                ui_call(self.ocv_threshold.setValue, inspector.threshold)
            except Exception as e:
                import traceback
                self.log_signal.emit(f"OCV build error: {e}")
            finally:
                ui_call(self.ocv_status.setText, "Ready")
        import threading
        threading.Thread(target=run, daemon=True).start()

    def _load_ocv_model(self):
        """Load a saved OCV model."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        model_path = os.path.join(pd, "models", "ocv_model.pkl")
        if not os.path.exists(model_path):
            self.log_signal.emit("No OCV model found. Build one first.")
            return
        try:
            from ocv.inspector import OCVInspector
            self._ocv_inspector = OCVInspector()
            n = self._ocv_inspector.load(model_path)
            self.log_signal.emit(f"OCV model loaded: {n} OK samples")
            self.ocv_threshold.setValue(self._ocv_inspector.threshold)
        except Exception as e:
            self.log_signal.emit(f"OCV load error: {e}")

    def _ocv_inspect_current(self):
        """Inspect current crop/character."""
        current_path = getattr(self.canvas, "current_image_path", None)
        if not current_path:
            return
        inspector = getattr(self, "_ocv_inspector", None)
        if inspector is None:
            self.log_signal.emit("No OCV model loaded. Build or load one first.")
            return
        inspector.threshold = self.ocv_threshold.value()
        try:
            from PIL import Image
            img = Image.open(current_path).convert("RGB")
            result = inspector.inspect(img)
            status = "OK" if result["ok"] else "NG"
            color = "green" if result["ok"] else "red"
            self.ocv_result_label.setText(
                f"<b style='color:{color}'>[{status}]</b> "
                f"Score: {result['score']:.3f} | "
                f"Threshold: {result['threshold']:.2f} | "
                f"Type: {result.get('defect_type', 'N/A')}"
            )
            self.log_signal.emit(f"OCV: {status} (score={result['score']:.3f})")
        except Exception as e:
            self.log_signal.emit(f"OCV inspect error: {e}")

    def _ocv_batch_inspect(self):
        """Batch inspect all images in the project."""
        if not self.current_project:
            return
        pd = str(self.pm.get_project_dir(self.current_project["name"]))
        model_path = os.path.join(pd, "models", "ocv_model.pkl")
        if not os.path.exists(model_path):
            self.log_signal.emit("No OCV model loaded. Build one first.")
            return
        self.ocv_status.setText("Batch inspecting...")
        # run() 是子线程：阈值先在主线程读好，别在线程里碰控件
        _ocv_thr = self.ocv_threshold.value()
        def run():
            from flow.batch import BatchRunner, build_ocv_flow, list_images
            images = list_images(pd)
            if not images:
                self.log_signal.emit("No images found")
                ui_call(self.ocv_status.setText, "Ready")
                return
            out_path = os.path.join(pd, "outputs", "ocv_results.jsonl")
            flow = build_ocv_flow(model_path=model_path, threshold=_ocv_thr, out_path=out_path)
            ok_cnt = ng_cnt = 0
            def on_progress(i, total, path, frame, err):
                nonlocal ok_cnt, ng_cnt
                if err is not None:
                    return
                if frame.result.get("ok"):
                    ok_cnt += 1
                else:
                    ng_cnt += 1
            br = BatchRunner(flow, project_dir=pd, stop_check=lambda: self._stop_flag)
            br.run(images, on_progress=on_progress)
            self.log_signal.emit(f"OCV batch: {ok_cnt} OK, {ng_cnt} NG out of {len(images)}")
            ui_call(self.ocv_status.setText, "Ready")
        import threading
        threading.Thread(target=run, daemon=True).start()

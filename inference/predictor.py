"""Inference engine - PyTorch + TensorRT + ONNX backends"""
import os, sys, numpy as np, torch, subprocess, struct
from PIL import Image
import torchvision.transforms.functional as TF
from training.models import create_model

class Predictor:
    def __init__(self, project_dir, model_filename="best_model.pth", device=None, backend="pytorch"):
        self.project_dir = project_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.classes = ["background"]
        self.ort_session = None
        self.trt_server = None  # Persistent TensorRT subprocess
        self._trt_python = None  # Path to Python 3.12 for TRT
        self.load(model_filename, backend=backend)

    def _find_trt_python(self):
        """Find a Python interpreter that can import tensorrt."""
        if self._trt_python and os.path.exists(self._trt_python):
            return self._trt_python
        candidates = [
            r"C:\Users\Administrator\AppData\Local\Temp\venv312_trt\Scripts\python.exe",
        ]
        for p in candidates:
            if os.path.exists(p):
                self._trt_python = p
                return p
        return None

    def _init_trt(self, model_filename="best_model.pth"):
        """Build TRT engine from ONNX model if needed, spawn persistent TRT server."""
        models_dir = os.path.join(self.project_dir, "models")
        onnx_path = os.path.join(models_dir, "model.onnx")
        engine_path = os.path.join(models_dir, "model_fp16.engine")

        if not os.path.exists(onnx_path):
            print("[TRT] No ONNX model found, please export ONNX first")
            return False

        # Build engine if not exists or ONNX is newer
        need_build = not os.path.exists(engine_path) or (
            os.path.getmtime(onnx_path) > os.path.getmtime(engine_path)
        )
        if need_build:
            print("[TRT] Building FP16 engine from ONNX...")
            import tempfile, shutil
            tmp_dir = tempfile.mkdtemp()
            tmp_onnx = os.path.join(tmp_dir, "model.onnx")
            tmp_engine = os.path.join(tmp_dir, "model_fp16.engine")

            # Copy ONNX with external data to temp (avoid Chinese path issues)
            shutil.copy2(onnx_path, tmp_onnx)
            onnx_data = onnx_path + ".data"
            if os.path.exists(onnx_data):
                shutil.copy2(onnx_data, tmp_onnx + ".data")

            ckpt_path = os.path.join(models_dir, model_filename)
            num_classes = 2
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                num_classes = ckpt.get("num_classes", 2)

            trtexec = r"C:\TensorRT-10.11.0\TensorRT-10.11.0.33\bin\trtexec.exe"
            cmd = [
                trtexec,
                f"--onnx={tmp_onnx}",
                f"--saveEngine={tmp_engine}",
                "--fp16",
                f"--minShapes=input:1x3x512x512",
                f"--optShapes=input:16x3x512x512",
                f"--maxShapes=input:64x3x512x512",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"[TRT] Engine build failed:\n{result.stderr[-500:]}")
                    return False
                shutil.copy2(tmp_engine, engine_path)
                print(f"[TRT] Engine built: {os.path.getsize(engine_path)/1024/1024:.1f} MB")
            except Exception as e:
                print(f"[TRT] Engine build error: {e}")
                return False
            finally:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass

        # Load checkpoint for num_classes
        ckpt_path = os.path.join(models_dir, model_filename)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.num_classes = ckpt["num_classes"]
            self.classes = ckpt["classes"]

        # Also prepare model for _create_overlay
        path = os.path.join(models_dir, model_filename)
        if os.path.exists(path) and self.model is None:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            model_name = checkpoint.get("model_name", "deeplabv3")
            self.model = create_model(self.num_classes, model_name).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

        # Spawn persistent TRT server
        return self._spawn_trt_server(engine_path)

    def _spawn_trt_server(self, engine_path):
        """Spawn persistent TRT server subprocess (Python 3.12)."""
        python312 = self._find_trt_python()
        if not python312:
            print("[TRT] Python 3.12 with tensorrt not found, falling back to PyTorch")
            return False

        server_script = os.path.join(os.environ.get("TEMP", "/tmp"), "trt_server.py")
        if not os.path.exists(server_script):
            print(f"[TRT] Server script not found: {server_script}")
            return False

        try:
            # Kill any existing server
            if self.trt_server is not None:
                try:
                    self.trt_server.stdin.close()
                    self.trt_server.terminate()
                    self.trt_server.wait(timeout=5)
                except Exception:
                    pass
                self.trt_server = None

            self.trt_server = subprocess.Popen(
                [python312, server_script],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=False,
            )
            self.trt_server.stdin.write(engine_path.encode() + b"\n")
            self.trt_server.stdin.flush()

            ready = self.trt_server.stdout.readline().decode().strip()
            if ready != "READY":
                print(f"[TRT] Server unexpected response: {ready}")
                return False

            print("[TRT] Server ready (FP16, persistent)")
            return True
        except Exception as e:
            print(f"[TRT] Failed to spawn server: {e}")
            self.trt_server = None
            return False

    def _trt_infer_batch(self, tiles_tensor):
        """Send batch [B,3,512,512] to TRT server, get back [B,512,512] uint8 argmax."""
        B, C, H, W = tiles_tensor.shape
        data = tiles_tensor.contiguous().cpu().numpy().tobytes()
        nbytes = len(data)

        self.trt_server.stdin.write(f"{B} {C} {H} {W} {nbytes}\n".encode())
        self.trt_server.stdin.write(data)
        self.trt_server.stdin.flush()

        # Read response header: "B H W out_bytes\n"
        resp = b""
        while not resp.endswith(b"\n"):
            ch = os.read(self.trt_server.stdout.fileno(), 1)
            if not ch:
                raise RuntimeError("TRT server closed")
            resp += ch
        bB, bH, bW, out_bytes = map(int, resp.decode().strip().split())

        # Read exact out_bytes of uint8 data
        out = b""
        while len(out) < out_bytes:
            chunk = os.read(self.trt_server.stdout.fileno(), out_bytes - len(out))
            if not chunk:
                break
            out += chunk

        return np.frombuffer(out, dtype=np.uint8).reshape(bB, bH, bW)

    def _predict_tiled_trt(self, image, tile_size=512, overlap=64):
        """Tiled inference via persistent TRT server (FP16)."""
        w, h = image.size
        stride = tile_size - overlap
        tiles = []
        positions = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w)
                y2 = min(y + tile_size, h)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)
                tile = image.crop((x1, y1, x2, y2))
                pw = tile_size - (x2 - x1)
                ph = tile_size - (y2 - y1)
                if pw > 0 or ph > 0:
                    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0))
                    tile = padded
                tiles.append(TF.to_tensor(tile))
                positions.append((x1, y1, x2, y2, ph, pw))

        pred_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        hann_1d = np.hanning(tile_size)
        hann = np.sqrt(hann_1d.reshape(-1, 1) * hann_1d.reshape(1, -1)).astype(np.float32)

        max_batch = 64
        for i in range(0, len(tiles), max_batch):
            batch_tiles = torch.stack(tiles[i:i+max_batch])
            batch_preds = self._trt_infer_batch(batch_tiles).astype(np.float32)

            for j, (x1, y1, x2, y2, ph, pw) in enumerate(positions[i:i+max_batch]):
                tp = batch_preds[j][:tile_size-ph, :tile_size-pw]
                th, tw = tp.shape
                wgt = hann[:th, :tw] if (th != tile_size or tw != tile_size) else hann
                pred_sum[y1:y2, x1:x2] += tp * wgt
                count_map[y1:y2, x1:x2] += wgt

        count_map[count_map == 0] = 1
        return np.round(pred_sum / count_map).astype(np.int64)

    def __del__(self):
        """Cleanup TRT server on deletion."""
        if self.trt_server is not None:
            try:
                self.trt_server.stdin.close()
                self.trt_server.terminate()
                self.trt_server.wait(timeout=5)
            except Exception:
                pass
            self.trt_server = None
    def load(self, filename="best_model.pth", backend="pytorch"):
        self.backend = backend
        models_dir = os.path.join(self.project_dir, "models")

        if backend == "tensorrt":
            if self._init_trt(filename):
                return
            print("[TRT] Falling back to PyTorch")
            backend = "pytorch"
            self.backend = "pytorch"

        self.backend = backend
        models_dir = os.path.join(self.project_dir, "models")

        if backend in ("torchscript",):
            ts_path = os.path.join(models_dir, "model_scripted.pt")
            if os.path.exists(ts_path):
                self.model = torch.jit.load(ts_path, map_location=self.device)
                self.model.eval()
                ckpt_path = os.path.join(models_dir, filename)
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    self.num_classes = ckpt["num_classes"]
                    self.classes = ckpt["classes"]
                return

        if backend.startswith("onnx"):
            onnx_path = os.path.join(models_dir, "model.onnx")
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    # Try CUDA first, fall back to CPU silently
                    self.ort_session = ort.InferenceSession(
                        onnx_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    actual = self.ort_session.get_providers()
                    if "CPUExecutionProvider" in actual and "CUDAExecutionProvider" not in actual:
                        print("[ONNX] CUDA unavailable (need CUDA 12.x, you have 13.x), using CPU - will be slow")
                        # Fall through to PyTorch for GPU speed
                        self.ort_session = None
                    else:
                        print(f"[ONNX] Using providers: {actual}")
                        ckpt_path = os.path.join(models_dir, filename)
                        if os.path.exists(ckpt_path):
                            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                            self.num_classes = ckpt["num_classes"]
                            self.classes = ckpt["classes"]
                        self.model = None
                        return
                except Exception as e:
                    print(f"[ONNX] Failed: {e}, falling back to PyTorch")

        # Default: PyTorch (also used as fallback for all other backends)
        path = os.path.join(models_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.num_classes = checkpoint["num_classes"]
        self.classes = checkpoint["classes"]
        model_name = checkpoint.get("model_name", "deeplabv3")
        self.model = create_model(self.num_classes, model_name).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.ort_session = None

        # torch.compile optimizations (Linux only with Triton; Windows uses cudagraphs)
        if backend in ("compiled", "max-autotune"):
            try:
                import triton  # noqa: F401
                has_triton = True
            except ImportError:
                has_triton = False

            if has_triton:
                mode = "reduce-overhead" if backend == "compiled" else "max-autotune"
                try:
                    self.model = torch.compile(self.model, mode=mode)
                    print(f"[Compiled] torch.compile (mode={mode})")
                    print("[Compiled] Warming up...")
                    dummy = torch.randn(1, 3, 512, 512, device=self.device)
                    with torch.amp.autocast('cuda'):
                        for _ in range(3):
                            _ = self.model(dummy)
                    torch.cuda.synchronize()
                    print("[Compiled] Ready")
                except Exception as e:
                    print(f"[Compiled] Failed: {e}, using eager")
            else:
                # Windows: try cudagraphs backend (no Triton needed)
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead", backend="cudagraphs")
                    print("[Compiled] Using cudagraphs backend (Windows)")
                    print("[Compiled] Warming up...")
                    dummy = torch.randn(1, 3, 512, 512, device=self.device)
                    with torch.amp.autocast('cuda'):
                        for _ in range(5):
                            _ = self.model(dummy)
                    torch.cuda.synchronize()
                    print("[Compiled] Ready")
                except Exception as e:
                    print(f"[Compiled] cudagraphs failed: {e}, using eager mode")

    @torch.inference_mode()
    def predict(self, image, return_overlay=False, tiled=True, scale=1.0):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_w, orig_h = image.size
        if scale < 1.0:
            sw, sh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            downsampled = image.resize((sw, sh), Image.BILINEAR)
        else:
            downsampled = image
            sw, sh = orig_w, orig_h

        w, h = downsampled.size
        if tiled and (w > 1024 or h > 1024):
            pred = self._predict_tiled(downsampled)
        elif not tiled:
            pred = self._predict_full(downsampled, fallback_tiled=True)
        elif self.ort_session is not None:
            # ONNX Runtime
            img_tensor = TF.to_tensor(downsampled).unsqueeze(0).numpy()
            ort_inputs = {self.ort_session.get_inputs()[0].name: img_tensor}
            ort_out = self.ort_session.run(None, ort_inputs)[0]
            pred = ort_out.argmax(1).squeeze(0)
        else:
            img_tensor = TF.to_tensor(downsampled).unsqueeze(0).to(self.device)
            try:
                with torch.amp.autocast('cuda'):
                    output = self.model(img_tensor)
                if isinstance(output, dict):
                    output = output["out"]
                # Keep on GPU until final step
                pred = output.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()
            except RuntimeError:
                pred = self._predict_tiled(downsampled)

        # Upsample mask back to original resolution
        if scale < 1.0:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred = np.array(pred_img.resize((orig_w, orig_h), Image.NEAREST)).astype(pred.dtype)

        if return_overlay:
            overlay = self._create_overlay(image, pred)
            return pred, overlay
        return pred

    @torch.no_grad()
    @torch.no_grad()
    def _predict_tiled(self, image, tile_size=512, overlap=64):
        if self.trt_server is not None:
            return self._predict_tiled_trt(image, tile_size, overlap)
        if self.ort_session is not None:
            return self._predict_tiled_onnx(image, tile_size, overlap)
        return self._predict_tiled_torch(image, tile_size, overlap)
    def _predict_full(self, image, fallback_tiled=True):
        """Inference on the entire image in one forward pass."""
        w, h = image.size
        try:
            img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            with torch.amp.autocast('cuda'):
                output = self.model(img_tensor)
            if isinstance(output, dict):
                output = output["out"]
            pred = output.argmax(1).squeeze(0).cpu().numpy()
            return pred
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and fallback_tiled:
                torch.cuda.empty_cache()
                print(f"[Predictor] Full-image OOM ({w}x{h}), falling back to tiled")
                return self._predict_tiled(image)
            raise

    @torch.inference_mode()
    def _predict_tiled_torch(self, image, tile_size=512, overlap=64):
        """Tiled inference with CPU-side Hanning-window blending.
        GPU scatter is slow; keeping blending on CPU is faster for accumulation."""
        w, h = image.size
        stride = tile_size - overlap
        tiles, positions = [], []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
                x1 = max(0, x2 - tile_size); y1 = max(0, y2 - tile_size)
                tile = image.crop((x1, y1, x2, y2))
                pw = tile_size - (x2 - x1); ph = tile_size - (y2 - y1)
                if pw > 0 or ph > 0:
                    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0)); tile = padded
                tiles.append(TF.to_tensor(tile)); positions.append((x1, y1, x2, y2, ph, pw))

        pred_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # Pre-compute Hanning window (NumPy is fast for this)
        hann = np.sqrt(np.hanning(tile_size).reshape(-1, 1) * np.hanning(tile_size).reshape(1, -1))

        # Dynamic batch size for better GPU utilization
        free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        batch_size = max(16, min(64, int(free_mb * 0.3 / (tile_size * tile_size * 3 * 2 / 1024**2))))

        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[i:i+batch_size]).to(self.device)
            with torch.amp.autocast('cuda'):
                output = self.model(batch)
            if isinstance(output, dict):
                output = output["out"]
            # Move to CPU as uint8 (smaller transfer than float32)
            batch_preds = output.argmax(1).to(torch.uint8).cpu().numpy()

            for j, (x1, y1, x2, y2, ph, pw) in enumerate(positions[i:i+batch_size]):
                tp = batch_preds[j][:tile_size-ph, :tile_size-pw].astype(np.float32)
                th, tw = tp.shape
                wgt = hann[:th, :tw] if (th != tile_size or tw != tile_size) else hann
                pred_sum[y1:y2, x1:x2] += tp * wgt
                count_map[y1:y2, x1:x2] += wgt

        count_map[count_map == 0] = 1
        return np.round(pred_sum / count_map).astype(np.int64)

    def _predict_tiled_onnx(self, image, tile_size=512, overlap=64):
        w, h = image.size
        stride = tile_size - overlap
        tiles = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2 = min(x + tile_size, w); y2 = min(y + tile_size, h)
                x1 = max(0, x2 - tile_size); y1 = max(0, y2 - tile_size)
                tile = image.crop((x1, y1, x2, y2))
                pw = tile_size - (x2 - x1); ph = tile_size - (y2 - y1)
                if pw > 0 or ph > 0:
                    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0)); tile = padded
                tiles.append((TF.to_tensor(tile).numpy(), x1, y1, x2, y2, ph, pw))

        pred_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        batch_size = 16
        input_name = self.ort_session.get_inputs()[0].name
        for i in range(0, len(tiles), batch_size):
            batch = np.stack([t[0] for t in tiles[i:i+batch_size]])
            batch_preds = self.ort_session.run(None, {input_name: batch})[0].argmax(1)
            for j, (_, x1, y1, x2, y2, ph, pw) in enumerate(tiles[i:i+batch_size]):
                tp = batch_preds[j][:tile_size-ph, :tile_size-pw].astype(np.float32)
                gy = np.hanning(tp.shape[0]).reshape(-1, 1)
                gx = np.hanning(tp.shape[1]).reshape(1, -1)
                wgt = np.sqrt(gy * gx)
                pred_sum[y1:y2, x1:x2] += tp * wgt
                count_map[y1:y2, x1:x2] += wgt
        count_map[count_map == 0] = 1
        return np.round(pred_sum / count_map).astype(np.int64)

    @torch.inference_mode()
    def classify(self, image, top_k=5):
        """Classify a single image. Returns list of (class_name, confidence) tuples.
        
        Automatically detects task_type from project.json and loads the
        appropriate classification model.
        """
        import json as _json
        from pathlib import Path as _Path
        
        project_json = _Path(self.project_dir) / "project.json"
        task_type = "语义分割"  # default
        if project_json.exists():
            with open(project_json, "r", encoding="utf-8") as f:
                meta = _json.load(f)
            task_type = meta.get("task_type", "语义分割")
        
        if task_type != "图像分类":
            raise RuntimeError(
                f"classify() is only available for 图像分类 projects, "
                f"current project is '{task_type}'. Use predict() for segmentation."
            )
        
        # Lazy-load classification model if not already loaded
        if not hasattr(self, "_cls_model"):
            from classification.trainer import ClassificationTrainer
            self._cls_trainer = ClassificationTrainer(
                self.project_dir, device=self.device
            )
            self._cls_trainer.load_model("best_model.pth")
            self._cls_model = self._cls_trainer.model
        
        # Prepare image
        import torchvision.transforms as _T
        from classification.dataset import IMAGENET_MEAN, IMAGENET_STD
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        img_size = getattr(self, "_cls_image_size", 224)
        transform = _T.Compose([
            _T.Resize((img_size, img_size)),
            _T.ToTensor(),
            _T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.amp.autocast('cuda'):
            output = self._cls_model(tensor)
        probs = output.softmax(1).squeeze(0)
        
        num_classes = len(self._cls_trainer.class_names)
        topk_probs, topk_ids = probs.topk(min(top_k, num_classes))
        return [
            (self._cls_trainer.class_names[idx], float(conf))
            for conf, idx in zip(topk_probs.cpu().numpy(), topk_ids.cpu().numpy())
        ]

    def _create_overlay(self, image, mask, alpha=0.5):
        import cv2
        from core.config import CLASS_COLORS
        img = np.array(image).astype(np.float32)
        overlay = img.copy()
        for c in range(1, self.num_classes):
            region = (mask == c)
            if not region.any():
                continue
            color = np.array(CLASS_COLORS[c % len(CLASS_COLORS)])
            overlay[region] = (1 - alpha) * overlay[region] + alpha * color
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        for c in range(1, self.num_classes):
            region = (mask == c)
            if not region.any():
                continue
            color = np.array(CLASS_COLORS[c % len(CLASS_COLORS)])
            # Draw class name on each connected component
            binary = region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            label = self.classes[c] if c < len(self.classes) else f"class_{c}"
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    # Font scale adapts to region size: 0.4 min, 2.0 max
                    font_scale = max(0.4, min(2.0, np.sqrt(area) / 80))
                    thickness = max(1, int(font_scale * 2))
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(overlay, (cx - tw // 2 - 2, cy - th - 2), (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
                    cv2.putText(overlay, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return overlay.astype(np.uint8)


"""Inference engine - PyTorch + TensorRT + ONNX backends"""
import os, sys, json, numpy as np, torch, subprocess, struct
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
        # 分类工程通常没有分割模型 (models/best_model.pth)：容错处理，延迟到真正
        # 调用 predict() 时报错；classify()/classify_with_threshold() 会懒加载分类模型。
        self._load_error = None
        try:
            self.load(model_filename, backend=backend)
        except FileNotFoundError as e:
            self._load_error = e
        # ★★ 2026-10-22：推理用的**滑窗块尺寸**必须等于训练时的裁剪尺寸。
        #   原来 `predict()` 里写死 200（注释还写着"模型是 200×200 训的"），
        #   而这次工程是按 256 训的 ⇒ 推理时同一块焊印在块里"变小变糊"，
        #   检测尺度整个错位（用户先问"能训吗"，训完再一推理就会踩这个）。
        #   取值顺序：checkpoint 的 image_size → 工程/平台的 .train_settings.json → 200。
        self._智能块尺寸 = None
        try:
            self._智能块尺寸 = self._resolve_train_tile(model_filename)
        except Exception:
            self._智能块尺寸 = None

    def _resolve_train_tile(self, model_filename="best_model.pth"):
        """训练时的裁剪尺寸（块大小）；读不到就返回 None（沿用老默认 200）。"""
        def _pick(v):
            if isinstance(v, (list, tuple)) and v:
                try:
                    return int(v[0])
                except Exception:
                    return None
            if isinstance(v, (int, float)) and int(v) > 0:
                return int(v)
            return None

        # ① checkpoint（新训练的模型会写进去）
        try:
            p = os.path.join(self.project_dir, "models", model_filename)
            if os.path.exists(p):
                ck = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(ck, dict):
                    got = _pick(ck.get("image_size"))
                    if got:
                        return got
        except Exception:
            pass
        # ② 工程/平台的训练参数文件（老模型没有 image_size 字段，靠这个兜底）
        cands = [
            os.path.join(self.project_dir, ".train_settings.json"),
            os.path.join(self.project_dir, ".cls_train_settings.json"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         ".train_settings.json"),
        ]
        for cp in cands:
            try:
                if os.path.exists(cp):
                    with open(cp, "r", encoding="utf-8") as f:
                        got = _pick(json.load(f).get("image_size"))
                    if got:
                        return got
            except Exception:
                continue
        return None

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
            self.model = self._build_model(model_name, checkpoint).to(self.device)
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
    # ────────── 多视图（三视图 9 通道）支持 ──────────
    def _detect_views(self):
        """读 project.json 判断是不是多视图工程。

        为什么要有这一步：训练走的是 `create_multiview_model`（9 通道，支持 unet），
        而推理原来硬套单视图 `create_model`（3 通道，且**没有 unet**）——
        结果三视图工程一做批量推理，每张图都报一次 `Unknown model: unet`（已踩）。
        """
        self.views = ()
        try:
            with open(os.path.join(self.project_dir, "project.json"), "r", encoding="utf-8") as f:
                meta = json.load(f)
            vs = tuple(meta.get("views") or ())
            if len(vs) > 1:
                self.views = vs
        except Exception:
            pass
        return self.views

    def _build_model(self, model_name, checkpoint=None):
        """按工程类型建模型：多视图 -> create_multiview_model（9 通道），否则单视图。"""
        ck = checkpoint if isinstance(checkpoint, dict) else {}
        if getattr(self, "views", ()):
            from training.multiview import create_multiview_model
            in_ch = len(self.views) * 3
            ck_in = ck.get("in_channels") or ck.get("in_ch")
            if ck_in is not None and int(ck_in) != in_ch:
                raise RuntimeError(
                    f"权重通道数与工程不匹配：权重记的是 {ck_in} 通道，"
                    f"工程 views={self.views} 需要 {in_ch} 通道。"
                    f"（是否用错了工程的模型？）")
            print(f"[Predictor] 多视图工程：views={self.views} -> {in_ch} 通道模型 "
                  f"(model_name={model_name})", flush=True)
            return create_multiview_model(
                model_name, self.num_classes, self.views, pretrained=False,
                unet_base=int(ck.get("unet_base", 16)), unet_depth=int(ck.get("unet_depth", 4)))
        # ★ 2026-10-21：单视图也支持 unet（和 training/models.py 的 create_model 对齐）
        if model_name == "unet":
            from training.multiview import MultiViewUNet
            _in = ck.get("in_channels") or ck.get("in_ch") or 3
            print(f"[Predictor] 单视图 U-Net：in_channels={_in} "
                  f"base={ck.get('unet_base', 16)} depth={ck.get('unet_depth', 4)}", flush=True)
            return MultiViewUNet(int(_in), self.num_classes,
                                 base=int(ck.get("unet_base", 16)),
                                 depth=int(ck.get("unet_depth", 4)))
        if model_name not in ("deeplabv3", "lraspp", "fcn"):
            raise RuntimeError(
                f"Unknown model: {model_name}（单视图工厂只支持 deeplabv3/unet/lraspp/fcn；"
                f"若这是三视图 unet 权重，请确认工程 project.json 里的 views 字段）")
        # ★ 单视图分支也要认权重记的通道数（2026-10-15）：三图工程若是用 GUI 的
        #   单视图 trainer 训的（走的是本分支），权重首层是 9 通道；
        #   这里若还按 3 通道建模，load_state_dict 会因形状不符而失败或被 strict=False 悄悄跳过。
        _ck_in = int(ck.get("in_channels") or 3)
        return create_model(self.num_classes, model_name, in_channels=_ck_in)

    @torch.inference_mode()
    def predict_multiview(self, view_images, tile=256, stride=None, return_overlay=False):
        """三视图推理：按 views 顺序拼成 N*3 通道，**原分辨率分块**跑，再拼回整图。

        view_images: dict{view: PIL.Image 或 ndarray}，必须齐 views 里所有视图（缺就报错，
        绝不拿别的视图顶替 —— 那会让"形状图"被漫反射冒充，结果看似正常其实全错）。
        """
        import cv2
        from PIL import Image as _I
        if not getattr(self, "views", ()):
            raise RuntimeError("不是多视图工程，请用 predict()")
        missing = [v for v in self.views if v not in view_images or view_images[v] is None]
        if missing:
            raise RuntimeError(f"缺少视图 {missing}（三视图必须齐全，不能用其它视图顶替）")
        imgs = []
        for v in self.views:
            im = view_images[v]
            if isinstance(im, np.ndarray):
                im = _I.fromarray(im)
            imgs.append(im.convert("RGB"))
        W, H = imgs[0].size
        for v, im in zip(self.views, imgs):
            if im.size != (W, H):
                raise RuntimeError(f"视图尺寸不一致：{v}={im.size} vs {imgs[0].size}（三视图必须像素对齐）")
        t = int(tile)
        st = int(stride) if stride else max(1, int(t * 0.75))
        pred = np.zeros((H, W), np.uint8)
        # 与训练同一套 ImageNet 归一化（写成常量而不是 import：predictor 顶层不依赖 training 包，
        # 免得推理环境缺训练依赖时直接崩）
        _MEAN = (0.485, 0.456, 0.406)
        _STD = (0.229, 0.224, 0.225)
        mean = torch.tensor(_MEAN * len(self.views)).view(-1, 1, 1)
        std = torch.tensor(_STD * len(self.views)).view(-1, 1, 1)
        for y0 in list(range(0, max(1, H - t + 1), st)) + ([H - t] if H > t and (H - t) % st else []):
            for x0 in list(range(0, max(1, W - t + 1), st)) + ([W - t] if W > t and (W - t) % st else []):
                chans = []
                for im in imgs:
                    box = (max(0, x0), max(0, y0), min(W, x0 + t), min(H, y0 + t))
                    tile_im = im.crop(box)
                    if tile_im.size != (t, t):
                        canvas = _I.new("RGB", (t, t), (0, 0, 0))
                        canvas.paste(tile_im, (0, 0)); tile_im = canvas
                    chans.append(np.array(tile_im, dtype=np.float32) / 255.0)
                x = np.concatenate(chans, axis=2)
                xt = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
                xt = (xt - mean.to(self.device)) / std.to(self.device)
                with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                    out = self.model(xt)
                if isinstance(out, dict):
                    out = out["out"]
                lab = out.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()
                y1, x1 = min(H, y0 + t), min(W, x0 + t)
                pred[max(0, y0):y1, max(0, x0):x1] = lab[: y1 - max(0, y0), : x1 - max(0, x0)]
        if return_overlay:
            return pred, self._create_overlay(imgs[0], pred)
        return pred

    def load(self, filename="best_model.pth", backend="pytorch"):
        self.backend = backend
        self._detect_views()
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
        self.model = self._build_model(model_name, checkpoint).to(self.device)
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
                    # ★ cudagraphs 后端**不接受 mode 参数** —— 原来传了
                    #   mode="reduce-overhead"，于是 torch 直接报
                    #   TypeError: CudagraphsBackend.__call__() got an unexpected keyword argument 'mode'
                    #   → 每次启动都"编译失败、退回 eager"，Compiled 后端等于没用。
                    self.model = torch.compile(self.model, backend="cudagraphs")
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
    def predict(self, image, return_overlay=False, tiled=True, scale=1.0, mode="fine"):
        """mode: "fast" = 缩到 1024 整图一次前向（对标友商 100ms 档）；
                 "fine" = 缩到 1024 后按 200 滑窗拼回（保留细节，慢些）。
        调用方也可以用对象属性 `_智能模式` 预设（界面上的「推理模式」下拉就写它）。"""
        if mode == "fine":
            mode = getattr(self, "_智能模式", "fine") or "fine"
        """预测整图。**顺带把置信度图存到 self.last_prob**（供「推理参数」的分数过滤用）。"""
        self.last_prob = None
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_w, orig_h = image.size
        # 滑窗块尺寸 = **训练时的裁剪尺寸**（`_智能块尺寸`，读 checkpoint 得来；读不到退回 200）
        _块 = int(getattr(self, "_智能块尺寸", 0) or 0) or 200
        _叠 = max(8, _块 // 2)
        # ★★ 智能推理（2026-10-22）：大图默认缩到**长边 1024**再推。
        #   为什么：模型是 200×200 训的，直接喂 5472×3648 会
        #   ① 显存炸（整图激活值上 GB）② 1050ms 以上 ③ 细节本来也被缩放抹掉。
        #   长边 ≤ 1500 的图（如 200×200 小图）**原样不动**。
        #
        # ⚠️ 2026-10-22 补：**分块推理（精细档）不能先缩整图**。
        #   训练是在**原分辨率**上按 256 切块，推理若先把 456×7041 缩到 66×1024
        #   再按 256 切，同一块焊印在块里只剩 1/7 大小 —— 尺度整个错位，
        #   现场表现就是"训练 mIoU 0.95，一推理啥也检不到"。
        #   所以：精细档 + 分块 + 块数可控（≤600）时走**原分辨率滑窗**；
        #   块数太多（38MP 那种）才退回 1024 缩放，避免一次推理几分钟。
        # 大图一律分块：不勾 Tiled 时"整图前向"意味着把整图缩成一个方形喂进去，
        # 对 456×7041 这种细长条等于把焊印压扁 7 倍 —— 实测一检一个 0 块。
        _大图 = (orig_w * orig_h) > (1500 * 1500)
        _要分块 = bool(tiled) or _大图
        _长边 = max(orig_w, orig_h)
        _智能缩放 = 1.0
        _原生滑窗 = False
        if _要分块 and mode != "fast":
            _步 = max(1, _块 - _叠)
            _估块数 = (((orig_w + _步 - 1) // _步) *
                       ((orig_h + _步 - 1) // _步))
            _原生滑窗 = _估块数 <= 600
        if _原生滑窗:
            _智能缩放 = 1.0
        elif _长边 > 1500 and scale >= 1.0:
            _智能缩放 = 1024.0 / float(_长边)
        if _智能缩放 < 1.0:
            scale = _智能缩放
        if scale < 1.0:
            sw, sh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            downsampled = image.resize((sw, sh), Image.BILINEAR)
        else:
            downsampled = image
            sw, sh = orig_w, orig_h

        w, h = downsampled.size
        if mode == "fast":
            # 快速档：缩到 1024 后**整图一次前向**，对标友商
            pred = self._predict_full(downsampled, fallback_tiled=False)
        elif _要分块 and (w > 1024 or h > 1024 or _原生滑窗):
            pred = self._predict_tiled(downsampled, tile_size=_块, overlap=_叠)
        elif not _要分块:
            # 只有小图才会走到这里：整图一次前向
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
                # 置信度图（max softmax）：分数过滤要用；只多一次 softmax，代价可忽略
                with torch.amp.autocast('cuda', enabled=False):
                    prob = output.float().softmax(1).max(1)[0]
                self.last_prob = prob.squeeze(0).to(torch.float16).cpu().numpy().astype(np.float32)
                # Keep on GPU until final step
                pred = output.argmax(1).squeeze(0).to(torch.uint8).cpu().numpy()
            except RuntimeError:
                pred = self._predict_tiled(downsampled)

        # Upsample mask back to original resolution
        if scale < 1.0:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred = np.array(pred_img.resize((orig_w, orig_h), Image.NEAREST)).astype(pred.dtype)
            if self.last_prob is not None and self.last_prob.shape[:2] == (sh, sw):
                p_img = Image.fromarray((self.last_prob * 255).astype(np.uint8))
                self.last_prob = (np.array(p_img.resize((orig_w, orig_h), Image.BILINEAR))
                                  .astype(np.float32) / 255.0)

        # ★★ 2026-10-22：**掩膜一律按 uint8 返回**。
        #   `_predict_tiled*`（滑窗）返回的是 `np.int64`（np.round(...).astype(np.int64)），
        #   而调用方会把它当"类别编号图"直接喂给 PIL / cv2：
        #     mask_editor._build_prediction_overlay → `Image.fromarray(int64)` 直接抛
        #     `TypeError: Cannot handle this data type: (1, 1), <i8>`（用户实拍崩溃）。
        #   以前大图走"整图前向"返回 uint8，所以从没暴露；现在大图强制分块，必踩。
        #   类别数 ≤255 是硬约束（掩膜像素值就是类别 id），uint8 足够。
        pred = np.ascontiguousarray(pred)
        if pred.dtype != np.uint8:
            pred = pred.astype(np.uint8)

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
            # 置信度图：分数过滤要用（非 tiled 路径）
            with torch.amp.autocast('cuda', enabled=False):
                prob = output.float().softmax(1).max(1)[0]
            self.last_prob = prob.squeeze(0).to(torch.float16).cpu().numpy().astype(np.float32)
            pred = output.argmax(1).squeeze(0).cpu().numpy()
            return pred
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and fallback_tiled:
                torch.cuda.empty_cache()
                print(f"[Predictor] Full-image OOM ({w}x{h}), falling back to tiled")
                return self._predict_tiled(image)
            raise

    @staticmethod
    def _tile_strides(w, h, tile_size, overlap):
        """每轴步长 + 该轴是否用 Hann 窗（窄条图横向不重叠）。

        ★★ 2026-10-22 提速（对标友商 ~120ms/张）：
        重叠的唯一目的是"拼缝平滑"。但 456 宽的图用 256 块、stride 128 时，
        横向切 4 列（0/128/256/384）—— 同一段像素被前向算了 2 遍，
        而宽度方向根本没有内容变化，纯浪费。实测 224 块 → 112 块。

        ⚠️ 某轴不重叠时，**该轴窗口必须用 1.0**：Hann 窗两端是 0，
        零重叠直接拼会在缝上得到 count_map≈0，除出来是噪声/黑线。
        用均匀窗时，边缘 clamp 造成的天然重叠会被 count_map 自动平均掉，不会重影。
        """
        ov_x = 0 if w <= tile_size * 2 else overlap
        ov_y = 0 if h <= tile_size * 2 else overlap
        return (max(1, tile_size - ov_x), max(1, tile_size - ov_y),
                ov_x > 0, ov_y > 0)

    @torch.inference_mode()
    def _predict_tiled_torch(self, image, tile_size=512, overlap=64):
        """Tiled inference with CPU-side Hanning-window blending.
        GPU scatter is slow; keeping blending on CPU is faster for accumulation."""
        w, h = image.size
        stride_x, stride_y, use_hx, use_hy = self._tile_strides(w, h, tile_size, overlap)
        tiles, positions = [], []
        for y in range(0, h, stride_y):
            for x in range(0, w, stride_x):
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
        prob_sum = np.zeros((h, w), dtype=np.float32)

        # Pre-compute blending window (NumPy is fast for this)
        # 某轴不重叠时该轴用均匀窗 1.0（Hann 两端为 0，会拼出黑缝）
        _wx = np.hanning(tile_size) if use_hx else np.ones(tile_size)
        _wy = np.hanning(tile_size) if use_hy else np.ones(tile_size)
        hann = np.sqrt(_wy.reshape(-1, 1) * _wx.reshape(1, -1))

        # Dynamic batch size for better GPU utilization
        free_mb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        batch_size = max(16, min(64, int(free_mb * 0.3 / (tile_size * tile_size * 3 * 2 / 1024**2))))

        for i in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[i:i+batch_size]).to(self.device)
            with torch.amp.autocast('cuda'):
                output = self.model(batch)
            if isinstance(output, dict):
                output = output["out"]
            # 置信度：与类别同一套 Hann 窗融合，保证分数过滤与掩膜落在同一块上
            with torch.amp.autocast('cuda', enabled=False):
                prob_batch = output.float().softmax(1).max(1)[0].to(torch.float16).cpu().numpy().astype(np.float32)
            # Move to CPU as uint8 (smaller transfer than float32)
            batch_preds = output.argmax(1).to(torch.uint8).cpu().numpy()

            for j, (x1, y1, x2, y2, ph, pw) in enumerate(positions[i:i+batch_size]):
                tp = batch_preds[j][:tile_size-ph, :tile_size-pw].astype(np.float32)
                pp = prob_batch[j][:tile_size-ph, :tile_size-pw]
                th, tw = tp.shape
                wgt = hann[:th, :tw] if (th != tile_size or tw != tile_size) else hann
                pred_sum[y1:y2, x1:x2] += tp * wgt
                prob_sum[y1:y2, x1:x2] += pp * wgt
                count_map[y1:y2, x1:x2] += wgt

        count_map[count_map == 0] = 1
        self.last_prob = (prob_sum / count_map).astype(np.float32)
        return np.round(pred_sum / count_map).astype(np.int64)

    def _predict_tiled_onnx(self, image, tile_size=512, overlap=64):
        w, h = image.size
        stride_x, stride_y, use_hx, use_hy = self._tile_strides(w, h, tile_size, overlap)
        tiles = []
        for y in range(0, h, stride_y):
            for x in range(0, w, stride_x):
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
                gy = (np.hanning(tp.shape[0]) if use_hy else np.ones(tp.shape[0])).reshape(-1, 1)
                gx = (np.hanning(tp.shape[1]) if use_hx else np.ones(tp.shape[1])).reshape(1, -1)
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
        self._ensure_cls_model()
        
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

    def _ensure_cls_model(self):
        """懒加载分类模型（只加载一次）。"""
        if hasattr(self, "_cls_model"):
            return
        from classification.trainer import ClassificationTrainer
        self._cls_trainer = ClassificationTrainer(self.project_dir, device=self.device)
        self._cls_trainer.load_model("best_model.pth")
        self._cls_model = self._cls_trainer.model

    def _cls_probs(self, image, tta=False):
        """返回 (probs[np, C], class_names)。tta=True 时对 原图/水平翻转/垂直翻转 取平均。"""
        import torchvision.transforms as _T
        from classification.dataset import IMAGENET_MEAN, IMAGENET_STD

        self._ensure_cls_model()
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
        with torch.no_grad():
            probs = self._cls_model(tensor).softmax(1)
            if tta:
                for dim in (2, 3):
                    probs = probs + self._cls_model(torch.flip(tensor, dims=[dim])).softmax(1)
                probs = probs / 3.0
        return probs.squeeze(0).cpu().numpy(), list(self._cls_trainer.class_names)

    def load_cls_threshold(self):
        """读取阈值调优结果 models/classification/threshold_report.json。

        Returns: (pos_class_name, threshold) —— 无则 (None, None)
        """
        import json as _json
        from pathlib import Path as _Path
        p = _Path(self.project_dir) / "models" / "classification" / "threshold_report.json"
        if not p.exists():
            return None, None
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = _json.load(f)
            return d.get("pos_class"), (d.get("best") or {}).get("threshold")
        except Exception:
            return None, None

    def classify_with_threshold(self, image, tta=False):
        """**按调优阈值**判定 OK/NG（而非 argmax），用于把过杀/漏检控制在目标内。

        阈值来自 classification.threshold 生成的 threshold_report.json；
        若未做阈值调优，则退化为 argmax 判定（verdict_by="argmax"）。

        Returns: dict(verdict, ng_class, ng_prob, threshold, verdict_by, topk, class_names)
        """
        probs, names = self._cls_probs(image, tta=tta)
        pos_class, thr = self.load_cls_threshold()

        order = np.argsort(-probs)
        topk = [(names[i], float(probs[i])) for i in order[:min(len(names), 5)]]

        if pos_class in names and thr is not None:
            ng_prob = float(probs[names.index(pos_class)])
            verdict = "NG" if ng_prob >= float(thr) else "OK"
            by = "threshold"
        else:  # 无阈值 -> argmax
            pos_class = names[int(order[0])]
            thr = None
            ng_prob = float(probs[order[0]])
            verdict = "NG" if pos_class not in ("OK", "ok", "良品", "合格") else "OK"
            by = "argmax"

        return {
            "verdict": verdict,
            "ng_class": pos_class,
            "ng_prob": round(ng_prob, 6),
            "threshold": thr,
            "verdict_by": by,
            "topk": [(n, round(c, 6)) for n, c in topk],
            "class_names": names,
            "tta": bool(tta),
        }

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


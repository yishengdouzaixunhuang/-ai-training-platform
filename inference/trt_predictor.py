"""TensorRT inference backend - uses trtexec CLI."""
import os, subprocess, tempfile, time, glob, shutil, numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

TRTEXEC = r"C:\TensorRT-10.11.0\TensorRT-10.11.0.33\bin\trtexec.exe"
TRT_LIB = r"C:\TensorRT-10.11.0\TensorRT-10.11.0.33\lib"

class TRTPredictor:
    def __init__(self, engine_path, num_classes, classes):
        self.engine_path = engine_path
        self.num_classes = num_classes
        self.classes = classes
        self._env = os.environ.copy()
        self._env["PATH"] = TRT_LIB + ";" + self._env.get("PATH", "")
        self._run_trt(np.random.randn(1, 3, 512, 512).astype(np.float32))
    
    def _run_trt(self, batch):
        """Run TRT. batch: float32 [B,3,H,W]. Returns: int64 [B,H,W]."""
        work_dir = tempfile.mkdtemp(prefix="trt_")
        input_file = os.path.join(work_dir, "input.raw")
        try:
            batch.astype(np.float32).tofile(input_file)
            shape_str = "x".join(str(s) for s in batch.shape)
            input_spec = chr(39) + "input" + chr(39) + ":" + input_file
            dump_prefix = os.path.join(work_dir, "out")
            cmd = [
                TRTEXEC,
                "--loadEngine=" + self.engine_path,
                "--iterations=1",
                "--loadInputs=" + input_spec,
                "--dumpRawBindingsToFile=" + dump_prefix,
                "--shapes=input:" + shape_str,
            ]
            subprocess.run(cmd, env=self._env, capture_output=True,
                           timeout=60, check=True, cwd=work_dir)
            B = batch.shape[0]
            pattern = f"output.output.{B}.*.{batch.shape[2]}.{batch.shape[3]}.fp32.raw"
            output_path = os.path.join(work_dir, pattern)
            if not os.path.exists(output_path):
                matches = glob.glob(os.path.join(work_dir, "output.output.*.fp32.raw"))
                output_path = matches[0] if matches else None
            if output_path and os.path.exists(output_path):
                data = np.fromfile(output_path, dtype=np.float32)
                C = data.size // (B * batch.shape[2] * batch.shape[3])
                output = data.reshape(
                    B, C, batch.shape[2], batch.shape[3])
                return output.argmax(1).astype(np.int64)
            raise RuntimeError("TRT output not found in " + work_dir)
        finally:
            try: shutil.rmtree(work_dir)
            except: pass
    
    def predict(self, image, return_overlay=False, tiled=True, scale=1.0):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        orig_w, orig_h = image.size
        if scale < 1.0:
            sw, sh = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
            image = image.resize((sw, sh), Image.BILINEAR)
        w, h = image.size
        if tiled and (w > 1024 or h > 1024):
            pred = self._predict_tiled(image)
        else:
            pred = self._predict_full(image)
        if scale < 1.0:
            pred_img = Image.fromarray(pred.astype(np.uint8))
            pred = np.array(pred_img.resize((orig_w, orig_h), Image.NEAREST))
        if return_overlay:
            overlay = self._create_overlay(image if scale >= 1.0 else Image.fromarray(np.array(image.resize((orig_w, orig_h), Image.BILINEAR))), pred)
            return pred, overlay
        return pred
    
    def _predict_full(self, image):
        t = TF.to_tensor(image).unsqueeze(0).numpy().astype(np.float32)
        return self._run_trt(t).squeeze(0)
    
    def _predict_tiled(self, image, tile_size=512, overlap=64):
        w, h = image.size
        stride = tile_size - overlap
        tiles, positions = [], []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
                x1, y1 = max(0, x2 - tile_size), max(0, y2 - tile_size)
                tile = image.crop((x1, y1, x2, y2))
                pw, ph = tile_size - (x2 - x1), tile_size - (y2 - y1)
                if pw > 0 or ph > 0:
                    padded = Image.new("RGB", (tile_size, tile_size), (0, 0, 0))
                    padded.paste(tile, (0, 0)); tile = padded
                tiles.append(TF.to_tensor(tile).numpy())
                positions.append((x1, y1, x2, y2, ph, pw))
        pred_sum = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        hann = np.sqrt(np.hanning(tile_size).reshape(-1, 1) * np.hanning(tile_size).reshape(1, -1))
        bs = 64
        for i in range(0, len(tiles), bs):
            batch = np.stack(tiles[i:i+bs]).astype(np.float32)
            bp = self._run_trt(batch).astype(np.uint8)
            for j, (x1, y1, x2, y2, ph, pw) in enumerate(positions[i:i+bs]):
                tp = bp[j][:tile_size-ph, :tile_size-pw].astype(np.float32)
                th, tw = tp.shape
                wgt = hann[:th, :tw] if (th != tile_size or tw != tile_size) else hann
                pred_sum[y1:y2, x1:x2] += tp * wgt
                count_map[y1:y2, x1:x2] += wgt
        count_map[count_map == 0] = 1
        return np.round(pred_sum / count_map).astype(np.int64)
    
    def _create_overlay(self, image, mask, alpha=0.5):
        import cv2
        from core.config import CLASS_COLORS
        img = np.array(image).astype(np.float32)
        overlay = img.copy()
        for c in range(1, self.num_classes):
            region = (mask == c)
            if not region.any(): continue
            color = np.array(CLASS_COLORS[c % len(CLASS_COLORS)])
            overlay[region] = (1 - alpha) * overlay[region] + alpha * color
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        for c in range(1, self.num_classes):
            region = (mask == c)
            if not region.any(): continue
            color = np.array(CLASS_COLORS[c % len(CLASS_COLORS)])
            binary = region.astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            label = self.classes[c] if c < len(self.classes) else f"class_{c}"
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 200: continue
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    fs = max(0.4, min(2.0, np.sqrt(area) / 80))
                    thk = max(1, int(fs * 2))
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, thk)
                    cv2.rectangle(overlay, (cx - tw // 2 - 2, cy - th - 2), (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
                    cv2.putText(overlay, label, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), thk, cv2.LINE_AA)
        return overlay.astype(np.uint8)

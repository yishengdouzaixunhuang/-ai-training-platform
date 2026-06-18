"""
OCR Engine wrapping RapidOCR (PP-OCRv4 ONNX Runtime).
Zero PaddlePaddle dependency. Auto-downloads models on first use.
"""
import numpy as np
from PIL import Image


class OCREngine:
    """Text detection + recognition engine."""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._engine = RapidOCR()

    def detect_and_recognize(self, image):
        """
        Run text detection + recognition on an image.

        Args:
            image: PIL Image, numpy array (H,W,3) RGB, or file path string

        Returns:
            list of dicts: [
                {"box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], "text": "...", "score": 0.99},
                ...
            ]
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(image, Image.Image):
            image = np.array(image)

        result, _ = self._engine(image)

        if result is None:
            return []

        boxes = []
        for item in result:
            box = item[0] if isinstance(item[0], list) else item[0].tolist()
            text = item[1]
            score = float(item[2])
            boxes.append({
                "box": box,
                "text": text,
                "score": score,
                # Convert 4-point box to axis-aligned bbox
                "x": int(min(p[0] for p in box)),
                "y": int(min(p[1] for p in box)),
                "w": int(max(p[0] for p in box) - min(p[0] for p in box)),
                "h": int(max(p[1] for p in box) - min(p[1] for p in box)),
            })

        return boxes

    def detect_only(self, image):
        """Detect text regions only (no recognition)."""
        results = self.detect_and_recognize(image)
        return [{"box": r["box"], "x": r["x"], "y": r["y"],
                 "w": r["w"], "h": r["h"]} for r in results]

    def crop_regions(self, image, results=None):
        """Crop text regions from image. If results is None, runs detection first."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        if results is None:
            results = self.detect_and_recognize(image)

        crops = []
        for r in results:
            x, y, w, h = r["x"], r["y"], r["w"], r["h"]
            # Expand slightly (2px margin)
            x1 = max(0, x - 2)
            y1 = max(0, y - 2)
            x2 = min(image.width, x + w + 2)
            y2 = min(image.height, y + h + 2)
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)

        return crops, results


# Singleton for reuse
_engine_instance = None


def get_ocr_engine():
    """Get or create the OCR engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = OCREngine()
    return _engine_instance

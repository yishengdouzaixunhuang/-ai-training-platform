"""
OCR pipeline: batch inference, result export, image overlay.
"""
import os, json, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(size=20):
    """Get a font for text overlay."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
        except Exception:
            return ImageFont.load_default()


def draw_ocr_overlay(image, results, font_size=16):
    """Draw OCR detection boxes and recognized text on an image.

    Args:
        image: PIL Image or numpy array (RGB)
        results: list of dicts from OCREngine.detect_and_recognize()
        font_size: font size for text labels

    Returns:
        PIL Image with overlay drawn
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = _get_font(font_size)

    for r in results:
        box = r["box"]
        text = r.get("text", "")
        score = r.get("score", 0.0)

        # Color by confidence: green > 0.9, yellow > 0.7, red < 0.7
        if score >= 0.9:
            color = (0, 200, 0)    # green
        elif score >= 0.7:
            color = (200, 200, 0)  # yellow
        else:
            color = (200, 0, 0)    # red

        # Draw polygon
        pts = [(p[0], p[1]) for p in box]
        draw.polygon(pts, outline=color, width=2)

        # Draw text label
        label = f"{text} ({score:.2f})"
        x, y = r["x"], max(0, r["y"] - font_size - 4)
        # Background rectangle for text
        try:
            bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
        except Exception:
            pass
        draw.text((x, y), label, fill=color, font=font)

    return image


def run_batch_ocr(image_dir, out_dir):
    """Run OCR on all images in a directory.

    Args:
        image_dir: path to image folder
        out_dir: path to output folder

    Returns:
        dict: {filename: [results]}
    """
    from ocr.engine import get_ocr_engine
    engine = get_ocr_engine()

    os.makedirs(out_dir, exist_ok=True)
    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    ext_set = {".bmp", ".png", ".jpg", ".jpeg"}
    images = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in ext_set
    ])

    all_results = {}
    total = len(images)

    for i, fname in enumerate(images):
        img_path = os.path.join(image_dir, fname)
        try:
            results = engine.detect_and_recognize(img_path)
            all_results[fname] = results

            # Save overlay
            img = Image.open(img_path).convert("RGB")
            overlay = draw_ocr_overlay(img.copy(), results)
            base = os.path.splitext(fname)[0]
            overlay.save(os.path.join(overlay_dir, f"{base}_ocr.jpg"), quality=90)

        except Exception as e:
            all_results[fname] = {"error": str(e)}

    # Save JSON results
    json_path = os.path.join(out_dir, "ocr_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Save CSV
    csv_path = os.path.join(out_dir, "ocr_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("File,BoxIndex,Text,Confidence,X,Y,W,H\n")
        for fname, results in all_results.items():
            if isinstance(results, list):
                for idx, r in enumerate(results):
                    f.write(f"{fname},{idx},{r.get('text','')},{r.get('score',0):.4f},"
                            f"{r.get('x',0)},{r.get('y',0)},{r.get('w',0)},{r.get('h',0)}\n")

    return all_results

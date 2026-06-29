import os
import numpy as np
import torch
from PIL import Image

_sam_predictor = None
_sam_device = None

def _get_device():
    global _sam_device
    if _sam_device is None:
        if torch.cuda.is_available():
            _sam_device = 'cuda'
        else:
            _sam_device = 'cpu'
    return _sam_device

def _ensure_sam():
    global _sam_predictor
    if _sam_predictor is not None:
        return _sam_predictor
    device = _get_device()
    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        raise ImportError(
            'segment-anything not installed. Run: pip install segment-anything'
        )
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'sam_vit_b_01ec64.pth')
    if not os.path.exists(model_path):
        import urllib.request
        url = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        print(f'[SAM] Downloading model to {model_path} (370MB)...')
        urllib.request.urlretrieve(url, model_path)
        print('[SAM] Download complete.')
    print(f'[SAM] Loading model on {device}...')
    sam = sam_model_registry['vit_b'](checkpoint=model_path)
    sam.to(device=device)
    sam.eval()
    _sam_predictor = SamPredictor(sam)
    print('[SAM] Model loaded.')
    return _sam_predictor

def sam_predict(image, points, labels, multimask=False):
    predictor = _ensure_sam()
    if predictor is None:
        return None
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    predictor.set_image(image_np)
    if not points:
        return None
    input_points = np.array(points, dtype=np.float32)
    input_labels = np.array(labels, dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=multimask,
    )
    if multimask:
        return [{'mask': masks[i], 'score': float(scores[i])} for i in range(len(masks))]
    else:
        best_idx = int(np.argmax(scores))
        return {'mask': masks[best_idx], 'score': float(scores[best_idx])}

def sam_predict_box(image, box):
    predictor = _ensure_sam()
    if predictor is None:
        return None
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert('RGB'))
    else:
        image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    predictor.set_image(image_np)
    input_box = np.array(box, dtype=np.float32).reshape(1, 4)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=False,
    )
    return {'mask': masks[0], 'score': float(scores[0])}

def sam_is_available():
    try:
        import segment_anything
        return True
    except ImportError:
        return False

def sam_preload_async(callback=None):
    import threading
    def _load():
        try:
            _ensure_sam()
            if callback:
                callback(True)
        except Exception as e:
            print(f'[SAM] Preload failed: {e}')
            if callback:
                callback(False)
    t = threading.Thread(target=_load, daemon=True)
    t.start()
    return t

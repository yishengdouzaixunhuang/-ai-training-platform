
"""
OCV Inspector: Character quality verification using feature embedding.
Learns "OK" character feature distribution, flags deviations as NG.

Approach: Extract features from a pretrained CNN, build a feature bank
of OK samples, then use nearest-neighbor distance for anomaly scoring.
"""
import os, json, pickle
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms


# ImageNet normalization
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


class OCVInspector:
    """Character quality inspector based on feature embedding."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_bank = None  # [N, D] numpy array of OK features
        self._threshold = None
        self._image_size = 96

    def _get_model(self):
        """Lazy-load a pretrained ResNet18 as feature extractor (no FC)."""
        if self._model is None:
            model = models.resnet18(weights="DEFAULT")
            # Remove FC and avgpool - keep conv features
            self._extractor = torch.nn.Sequential(
                model.conv1, model.bn1, model.relu, model.maxpool,
                model.layer1, model.layer2, model.layer3, model.layer4,
                model.avgpool
            ).to(self.device)
            self._extractor.eval()
        return self._extractor

    @torch.no_grad()
    def extract_features(self, images):
        """Extract feature vectors from character images.

        Args:
            images: list of PIL Images or numpy arrays, or a single image

        Returns:
            np.ndarray of shape [N, 512] (ResNet18 feature dim)
        """
        model = self._get_model()
        if not isinstance(images, list):
            images = [images]

        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(_DEFAULT_TRANSFORM(img))

        batch = torch.stack(tensors).to(self.device)
        features = model(batch).squeeze(-1).squeeze(-1)  # [N, 512]
        return features.cpu().numpy()

    def build_feature_bank(self, ok_dir):
        """Build feature bank from OK character images.

        Args:
            ok_dir: directory containing OK character crop images
        """
        images = []
        ext_set = {".bmp", ".png", ".jpg", ".jpeg"}
        for fname in sorted(os.listdir(ok_dir)):
            if os.path.splitext(fname)[1].lower() in ext_set:
                img = Image.open(os.path.join(ok_dir, fname)).convert("RGB")
                images.append(img)

        if not images:
            raise ValueError(f"No images found in {ok_dir}")

        features = self.extract_features(images)
        self._feature_bank = features
        self._compute_threshold()
        return len(images)

    def _compute_threshold(self, percentile=95):
        """Set anomaly threshold based on OK feature distribution."""
        if self._feature_bank is None or len(self._feature_bank) < 2:
            self._threshold = 5.0
            return

        # Compute pairwise distances among OK samples
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(self._feature_bank)
        # Use nearest neighbor distance (excluding self)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        # Threshold = percentile of OK nearest-neighbor distances * 2
        self._threshold = float(np.percentile(nn_dists, percentile)) * 2.5

    def inspect(self, image):
        """Inspect a single character image.

        Args:
            image: PIL Image or numpy array

        Returns:
            dict: {
                "ok": bool,
                "score": float (anomaly score, higher = more anomalous),
                "threshold": float
            }
        """
        if self._feature_bank is None:
            raise RuntimeError("Feature bank not built. Call build_feature_bank() first.")

        features = self.extract_features([image])  # [1, 512]

        # Nearest neighbor distance to OK feature bank
        from sklearn.metrics.pairwise import euclidean_distances
        dists = euclidean_distances(features, self._feature_bank)[0]
        min_dist = float(dists.min())

        ok = min_dist <= self._threshold
        return {
            "ok": ok,
            "score": min_dist,
            "threshold": self._threshold,
            "defect_type": self._classify_defect(min_dist),
        }

    def _classify_defect(self, score):
        """Heuristic defect classification based on anomaly score."""
        if score <= self._threshold:
            return "OK"
        ratio = score / self._threshold
        if ratio > 3.0:
            return "break"      # severe deviation ? broken stroke
        elif ratio > 2.0:
            return "blur"       # moderate ? blurry
        elif ratio > 1.5:
            return "skew"       # mild ? skewed
        else:
            return "dirt"       # borderline ? dirt/speck

    def inspect_batch(self, images):
        """Inspect multiple character images."""
        return [self.inspect(img) for img in images]

    def save(self, path):
        """Save feature bank and threshold to disk."""
        data = {
            "feature_bank": self._feature_bank,
            "threshold": self._threshold,
            "image_size": self._image_size,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load feature bank and threshold from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._feature_bank = data["feature_bank"]
        self._threshold = data["threshold"]
        self._image_size = data.get("image_size", 96)
        return len(self._feature_bank) if self._feature_bank is not None else 0

    @property
    def is_ready(self):
        return self._feature_bank is not None and self._threshold is not None

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = float(value)

    @property
    def num_ok_samples(self):
        return len(self._feature_bank) if self._feature_bank is not None else 0


def train_ocv_model(ok_dir, output_path):
    """Train an OCV model from OK character images.

    Args:
        ok_dir: directory of OK character crops
        output_path: path to save the model (.pkl)

    Returns:
        OCVInspector instance
    """
    inspector = OCVInspector()
    n = inspector.build_feature_bank(ok_dir)
    inspector.save(output_path)
    return inspector, n

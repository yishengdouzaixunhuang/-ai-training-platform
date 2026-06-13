# Detection package
from detection.box_manager import BoxManager
from detection.box_overlay import DetectionOverlay
from detection.coco_io import load_coco_json, save_coco_json, get_det_json_path
from detection.dataset import build_yolo_dataset, get_annotated_count
from detection.trainer import DetectionTrainer, DET_MODELS
from detection.eval import evaluate_predictions

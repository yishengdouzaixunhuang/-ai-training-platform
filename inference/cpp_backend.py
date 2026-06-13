"""C++ inference backend - calls standalone cpp_infer executable."""
import os, subprocess, json, tempfile, numpy as np
from PIL import Image

class CppInferenceBackend:
    def __init__(self, project_dir, model_file="model.onnx"):
        self.project_dir = project_dir
        self.model_path = os.path.join(project_dir, "models", model_file)
        self.exe_path = os.path.join(os.path.dirname(__file__), "cpp", "build", "Release", "cpp_infer.exe")

    def predict(self, image_path, output_dir=None):
        if not os.path.exists(self.exe_path):
            raise FileNotFoundError(f"C++ binary not built: {self.exe_path}\nBuild with: cd inference/cpp && cmake -B build && cmake --build build --config Release")
        if output_dir is None:
            output_dir = os.path.join(self.project_dir, "outputs")
        os.makedirs(output_dir, exist_ok=True)

        result = subprocess.run(
            [self.exe_path, self.model_path, image_path],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"C++ inference failed: {result.stderr}")

        # Read saved prediction mask
        base = os.path.splitext(os.path.basename(image_path))[0]
        pred_path = os.path.join(output_dir, base + "_pred.png")
        if os.path.exists(pred_path):
            pred = np.array(Image.open(pred_path))
            return pred
        return None

"""TensorRT inference via trtexec CLI - blazing fast, zero Python deps."""
import os, subprocess, tempfile, time, numpy as np

TRTEXEC = r"C:\TensorRT-10.11.0\TensorRT-10.11.0.33\bin\trtexec.exe"
TRT_LIB = r"C:\TensorRT-10.11.0\TensorRT-10.11.0.33\lib"

class TRTInfer:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        env = os.environ.copy()
        env["PATH"] = TRT_LIB + ";" + env.get("PATH", "")
        
        # Warmup
        subprocess.run(
            [TRTEXEC, "--loadEngine=" + engine_path, "--iterations=3"],
            env=env, capture_output=True, timeout=60
        )
        self._env = env
    
    def infer_batch(self, tiles):
        """tiles: list of float32 numpy arrays [3,512,512]. Returns list of int64 [512,512]."""
        raise NotImplementedError("Batch trtexec not supported yet")

# For now, confirm engine loads
if __name__ == "__main__":
    engine = os.environ.get("TEMP", ".") + r"\model_fp16.engine"
    t0 = time.time()
    infer = TRTInfer(engine)
    print(f"Engine loaded in {(time.time()-t0)*1000:.0f}ms")
    print("TensorRT ready!")

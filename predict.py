"""
Minimal test predictor — verify Cog pipeline works before adding FLUX
"""

import time
import torch
from PIL import Image, ImageDraw, ImageFont
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        import sys
        print("=== Minimal test starting ===", flush=True)
        print(f"Python: {sys.version}", flush=True)
        print(f"torch: {torch.__version__}", flush=True)
        print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB", flush=True)
        print("=== Setup complete ===", flush=True)

    def predict(self, text: str = Input(default="Hello!", description="Text to draw")) -> Path:
        print(f"Predict called with: {text}", flush=True)
        img = Image.new("RGB", (512, 512), color=(30, 30, 60))
        draw = ImageDraw.Draw(img)
        draw.text((100, 240), text, fill=(255, 255, 255))
        output = Path("/tmp/test.png")
        img.save(str(output))
        return output

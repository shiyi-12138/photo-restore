"""
Absolute minimal test - no imports beyond cog + PIL
"""

import torch
from PIL import Image, ImageDraw
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("SETUP OK", flush=True)

    def predict(self) -> Path:
        print("PREDICT OK", flush=True)
        img = Image.new("RGB", (256, 256), color=(0, 128, 0))
        draw = ImageDraw.Draw(img)
        draw.text((80, 120), "OK!", fill=(255, 255, 255))
        out = Path("/tmp/test.png")
        img.save(str(out))
        return out

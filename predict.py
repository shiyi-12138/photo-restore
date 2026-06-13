"""
Minimal test with Cog V2 API (run instead of predict)
"""

from PIL import Image, ImageDraw
from cog import BasePredictor, Path, Input


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("SETUP OK", flush=True)

    def run(self, text: str = Input(default="OK!", description="Text to draw")) -> Path:
        print("RUN OK", flush=True)
        img = Image.new("RGB", (256, 256), color=(0, 128, 0))
        draw = ImageDraw.Draw(img)
        draw.text((80, 120), text, fill=(255, 255, 255))
        out = Path("/tmp/test.png")
        img.save(str(out))
        return out

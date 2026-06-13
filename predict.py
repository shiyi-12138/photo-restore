"""
Photo Restore - Old photo restoration using FLUX.1 Kontext
All weights pre-downloaded at build time to avoid 10GB runtime timeout.
"""

import os
import time
import sys
import torch
from PIL import Image
from cog import BasePredictor, Path, Input

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import configs, load_clip, load_t5
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from util import generate_compute_step_map

# Weights pre-downloaded at build time
KONTEXT_WEIGHTS_PATH = "./models/kontext/kontext-dev.sft"
AE_WEIGHTS_PATH = "./models/flux-dev/ae.safetensors"
T5_PATH = "./models/t5"
CLIP_PATH = "./models/clip"


def build_restore_prompt(auto_colorize: bool) -> str:
    base = (
        "Restore this old photograph. "
        "Fix all scratches, tears, dust spots, creases, and physical damage. "
        "Remove noise and grain. "
        "Enhance facial details, sharpen edges, and improve overall clarity. "
        "Preserve the original composition, subjects, and lighting direction exactly. "
    )
    if auto_colorize:
        base += (
            "Colorize the image naturally with realistic skin tones, "
            "natural fabric colors, and appropriate background hues. "
            "The colorization should look like an authentic color photograph, not artificial."
        )
    else:
        base += "Keep the original black-and-white tones."
    return base


class Predictor(BasePredictor):
    """Photo Restore - Old photo restoration with FLUX.1 Kontext"""

    def setup(self) -> None:
        print("=== Photo Restore setup ===", flush=True)
        self.device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB", flush=True)

        st = time.time()
        print("Loading T5...", flush=True)
        self.t5 = load_t5(self.device, max_length=512, t5_path=T5_PATH)
        print(f"T5 loaded ({time.time()-st:.1f}s), GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        st = time.time()
        print("Loading CLIP...", flush=True)
        self.clip = load_clip(self.device, clip_path=CLIP_PATH)
        print(f"CLIP loaded ({time.time()-st:.1f}s), GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        st = time.time()
        print("Loading FLUX Kontext...", flush=True)
        self.model = self._load_kontext()
        print(f"Kontext loaded ({time.time()-st:.1f}s), GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        st = time.time()
        print("Loading AE...", flush=True)
        self.ae = self._load_ae()
        print(f"AE loaded ({time.time()-st:.1f}s), GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

        self.model.eval()
        print("=== Setup complete ===", flush=True)

    def _load_kontext(self):
        config = configs["flux-dev"]
        with torch.device("meta"):
            model = Flux(config.params).to(torch.bfloat16)
        sd = load_sft(KONTEXT_WEIGHTS_PATH, device=str(self.device))
        model.load_state_dict(sd, strict=False, assign=True)
        return model

    def _load_ae(self):
        config = configs["flux-dev"]
        with torch.device("meta"):
            ae = AutoEncoder(config.ae_params)
        sd = load_sft(AE_WEIGHTS_PATH, device=str(self.device))
        ae.load_state_dict(sd, strict=False, assign=True)
        return ae

    def predict(
        self,
        image: Path = Input(description="Old photo to restore"),
        auto_colorize: bool = Input(default=True, description="Colorize B&W photos"),
        guidance: float = Input(default=3.0, ge=1.0, le=10.0, description="Restoration strength"),
        seed: int = Input(default=42, description="Random seed"),
    ) -> Path:
        prompt = build_restore_prompt(auto_colorize=auto_colorize)

        with torch.inference_mode():
            print(f"Restoring with guidance={guidance}, colorize={auto_colorize}", flush=True)
            inp, h, w = prepare_kontext(
                t5=self.t5, clip=self.clip, prompt=prompt, ae=self.ae,
                img_cond_path=str(image), target_width=None, target_height=None,
                bs=1, seed=seed, device=self.device,
            )
            inp.pop("img_cond_orig", None)

            compute_map = generate_compute_step_map("go really fast", 28)
            timesteps = get_schedule(28, inp["img"].shape[1], shift=True)

            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guidance, compute_step_map=compute_map)
            x = unpack(x.float(), h, w)

            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                x = self.ae.decode(x)

            x = x.clamp(-1, 1)
            x = (x + 1) / 2
            x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
            result = Image.fromarray(x[0])

        output = "/tmp/restored.png"
        result.save(output, format="PNG")
        print("Done!", flush=True)
        return Path(output)

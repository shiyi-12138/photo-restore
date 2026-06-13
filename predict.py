"""
Photo Restore - Old photo restoration using FLUX.1 Kontext
Fixes scratches, damage, and colorizes old photos.
"""

import os
import time
import torch
from PIL import Image
from cog import BasePredictor, Path, Input

from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.util import configs, load_clip, load_t5
from flux.model import Flux
from flux.modules.autoencoder import AutoEncoder
from safetensors.torch import load_file as load_sft
from safety_checker import SafetyChecker
from util import print_timing, generate_compute_step_map
from weights import download_weights

from flux.util import ASPECT_RATIOS

KONTEXT_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/kontext/release-candidate/kontext-dev.sft"
KONTEXT_WEIGHTS_PATH = "./models/kontext/kontext-dev.sft"
AE_WEIGHTS_URL = "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/safetensors/ae.safetensors"
AE_WEIGHTS_PATH = "./models/flux-dev/ae.safetensors"
T5_WEIGHTS_URL = (
    "https://weights.replicate.delivery/default/official-models/flux/t5/t5-v1_1-xxl.tar"
)
T5_WEIGHTS_PATH = "./models/t5"
CLIP_URL = "https://weights.replicate.delivery/default/official-models/flux/clip/clip-vit-large-patch14.tar"
CLIP_PATH = "./models/clip"


# Optimized prompt for photo restoration
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
        self.device = torch.device("cuda")
        download_model_weights()

        st = time.time()
        print("Loading T5...")
        self.t5 = load_t5(self.device, max_length=512, t5_path=T5_WEIGHTS_PATH)
        print(f"T5 loaded in {time.time() - st:.1f}s")

        st = time.time()
        print("Loading CLIP...")
        self.clip = load_clip(self.device, clip_path=CLIP_PATH)
        print(f"CLIP loaded in {time.time() - st:.1f}s")

        st = time.time()
        print("Loading Kontext model...")
        self.model = _load_kontext_model(device=self.device)
        print(f"Kontext model loaded in {time.time() - st:.1f}s")

        st = time.time()
        print("Loading autoencoder...")
        self.ae = _load_ae_local(device=self.device)
        print(f"AE loaded in {time.time() - st:.1f}s")

        st = time.time()
        print("Compiling with torch.compile...")
        self.model = torch.compile(self.model, dynamic=True)

        # Warmup
        self._warmup()
        print(f"Setup complete in {time.time() - st:.1f}s")

    def _warmup(self):
        warmup_img = Path(__file__).parent / "lady.png"
        if warmup_img.exists():
            self._run_inference(
                prompt=build_restore_prompt(auto_colorize=False),
                input_image=warmup_img,
                num_inference_steps=20,
                guidance=2.5,
                seed=42,
                go_fast=True,
            )

    def predict(
        self,
        image: Path = Input(description="Old photo to restore. JPEG/PNG/WebP supported."),
        auto_colorize: bool = Input(
            default=True,
            description="Automatically colorize black-and-white photos. Turn off to keep B&W.",
        ),
        guidance: float = Input(
            default=3.0,
            ge=1.0,
            le=10.0,
            description="How strongly to follow the restoration instruction. Higher = more aggressive restoration.",
        ),
        seed: int = Input(
            default=None,
            description="Random seed for reproducible results.",
        ),
    ) -> Path:
        """Restore an old photo: fix damage, enhance details, colorize."""
        prompt = build_restore_prompt(auto_colorize=auto_colorize)

        with torch.inference_mode():
            image = self._run_inference(
                prompt=prompt,
                input_image=image,
                num_inference_steps=28,
                guidance=guidance,
                seed=seed or int.from_bytes(os.urandom(2), "big"),
                go_fast=True,
            )

        output_path = "/tmp/restored.png"
        image.save(output_path, format="PNG")
        return Path(output_path)

    def _run_inference(
        self,
        prompt: str,
        input_image: Path,
        num_inference_steps: int,
        guidance: float,
        seed: int,
        go_fast: bool,
    ) -> Image.Image:
        inp, final_height, final_width = prepare_kontext(
            t5=self.t5,
            clip=self.clip,
            prompt=prompt,
            ae=self.ae,
            img_cond_path=str(input_image),
            target_width=None,
            target_height=None,
            bs=1,
            seed=seed,
            device=self.device,
        )

        inp.pop("img_cond_orig", None)

        if go_fast:
            compute_step_map = generate_compute_step_map("go really fast", num_inference_steps)
        else:
            compute_step_map = generate_compute_step_map("none", num_inference_steps)

        timesteps = get_schedule(num_inference_steps, inp["img"].shape[1], shift=True)

        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            compute_step_map=compute_step_map,
        )

        x = unpack(x.float(), final_height, final_width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        x = (x.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return Image.fromarray(x[0])


def download_model_weights():
    """Download all required model weights"""
    if not os.path.exists(KONTEXT_WEIGHTS_PATH):
        print("Downloading Kontext weights...")
        download_weights(KONTEXT_WEIGHTS_URL, Path(KONTEXT_WEIGHTS_PATH))
    if not os.path.exists(AE_WEIGHTS_PATH):
        print("Downloading AE weights...")
        download_weights(AE_WEIGHTS_URL, Path(AE_WEIGHTS_PATH))
    if not os.path.exists(T5_WEIGHTS_PATH):
        print("Downloading T5 weights...")
        download_weights(T5_WEIGHTS_URL, Path(T5_WEIGHTS_PATH))
    if not os.path.exists(CLIP_PATH):
        print("Downloading CLIP weights...")
        download_weights(CLIP_URL, Path(CLIP_PATH))


def _load_kontext_model(device: str | torch.device = "cuda"):
    config = configs["flux-dev"]
    with torch.device("meta"):
        model = Flux(config.params).to(torch.bfloat16)

    sd = load_sft(KONTEXT_WEIGHTS_PATH, device=str(device))
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        print(f"Missing keys: {missing}")
    return model


def _load_ae_local(device: str | torch.device = "cuda"):
    config = configs["flux-dev"]
    with torch.device("meta"):
        ae = AutoEncoder(config.ae_params)

    sd = load_sft(AE_WEIGHTS_PATH, device=str(device))
    ae.load_state_dict(sd, strict=False, assign=True)
    return ae

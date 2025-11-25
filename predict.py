from typing import Any
from pathlib import Path as SysPath
import torch

from cog import BasePredictor, Input, Path

from stages.sdxl import load_sdxl_pipeline, generate_character_image
from stages.faceswap import load_faceswap_models, faceswap_apply
from stages.wan_remix import load_svd_pipeline, generate_video
from stages.vfi_post import upscale_fps, film_effect


class Predictor(BasePredictor):
    def setup(self):
        print("MPS available:", torch.backends.mps.is_available())
        self.sdxl = load_sdxl_pipeline()
        self.face_app, self.face_swapper = load_faceswap_models()
        self.svd = load_svd_pipeline()

    def predict(
        self,
        face_img: str = Input(description="URL to face image"),
        character_prompt: str = Input(description="Character description (SDXL prompt)"),
        scene_prompt: str = Input(
            description="Scene description"
        ),
    ) -> Any:
        out_dir = SysPath("/tmp/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)

        img1_path = out_dir / "stage1_character.png"
        img2_path = out_dir / "stage2_faceswapped.png"
        video_raw_path = out_dir / "stage3_raw.mp4"
        video_60_path = out_dir / "stage4_60fps.mp4"
        video_final_path = out_dir / "stage4_final_film.mp4"

        generate_character_image(self.sdxl, character_prompt, str(img1_path))

        faceswap_apply(
            self.face_app,
            self.face_swapper,
            str(img1_path),
            face_img,
            str(img2_path),
        )

        generate_video(self.svd, str(img2_path), scene_prompt, str(video_raw_path))

        upscale_fps(str(video_raw_path), str(video_60_path))
        film_effect(str(video_60_path), str(video_final_path))

        return {
            "image_raw": Path(str(img1_path)),
            "image_faceswapped": Path(str(img2_path)),
            "video_raw": Path(str(video_raw_path)),
            "video_final": Path(str(video_final_path)),
        }
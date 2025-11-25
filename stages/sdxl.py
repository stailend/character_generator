# stages/sdxl.py

import torch
from diffusers import AutoPipelineForText2Image

pipe = None

def load_sdxl_pipeline():
    global pipe
    if pipe is None:
        pipe = AutoPipelineForText2Image.from_pretrained(
            "segmind/tiny-sd",
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
        )

        if torch.backends.mps.is_available():
            pipe.to("mps")
            print("✅ Using MPS on macOS")
        else:
            pipe.to("cpu")
            print("✅ Using CPU")

    return pipe


def generate_character_image(prompt: str, out_path="stage1.png"):
    pipe = load_sdxl_pipeline()
    result = pipe(prompt)
    image = result.images[0]
    image.save(out_path)
    return out_path
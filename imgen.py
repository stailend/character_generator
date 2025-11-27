# imgen.py

import torch
from diffusers import AutoPipelineForText2Image

pipe = None

def load_pipeline():
    global pipe
    if pipe is None:

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            dtype = torch.float16
        else:
            dtype = torch.float32

        pipe = AutoPipelineForText2Image.from_pretrained(
            "segmind/tiny-sd",
            torch_dtype=dtype
        )

        if torch.cuda.is_available():
            pipe.to("cuda")
            print("== Using CUDA ==")
        elif torch.backends.mps.is_available():
            pipe.to("mps")
            print("== Using MPS ==")
        else:
            pipe.to("cpu")
            print("== Using CPU ==")

    return pipe

def generate_image(prompt: str, out_path="generated_image.png"):
    pipe = load_pipeline()
    result = pipe(prompt)
    img = result.images[0]
    img.save(out_path)
    return out_path
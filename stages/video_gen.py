# stages/video_gen.py

import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

pipe = None

def load_video_pipeline():
    global pipe
    if pipe is None:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32
        )

        if torch.backends.mps.is_available():
            pipe.to("mps")
            print("Using MPS for video generation")
        else:
            pipe.to("cpu")
            print("Using CPU")

    return pipe


def generate_video(image_path: str, prompt: str, out_path="stage3.mp4"):
    pipe = load_video_pipeline()

    image = Image.open(image_path).convert("RGB")

    result = pipe(
        image=image,
        prompt=prompt,
        num_frames=30
    )

    video_frames = result.frames[0] 

    import cv2
    import numpy as np

    h, w = video_frames[0].size
    video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (h, w))

    for frame in video_frames:
        frame_np = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(frame_np)

    video.release()
    return out_path
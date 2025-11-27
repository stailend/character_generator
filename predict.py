# predict.py

import sys
import os
from imgen import generate_image
from faceswap import faceswap_from_url, faceswap_from_local

def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

def main():

    if len(sys.argv) < 3:
        print("Usage:")
        print(" - python predict.py \"prompt text\" FACE_URL")
        print(" - python predict.py \"prompt text\" FACE_PATH")
        return

    prompt = sys.argv[1]
    face_input = sys.argv[2]

    print("== Stage 1: Generating image ==")
    stage1_path = generate_character_image(prompt, out_path="generated_image.png")

    if is_url(face_input):
        print("== Stage 2: FaceSwap from URL ==")
        stage2_path = faceswap_from_url(stage1_path, face_input)
    else:
        if not os.path.exists(face_input):
            raise FileNotFoundError(f"Local face file not found: {face_input}")

        print("== Stage 2: FaceSwap from local ==")
        stage2_path = faceswap_from_local(stage1_path, face_input)

    print("== Done! Result saved:", stage2_path, ' ==')


if __name__ == "__main__":
    main()
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from huggingface_hub import hf_hub_download
import urllib.request
import requests
app = None
swapper = None

def setup_faceswap():
    global app, swapper

    if app is None:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1)  
        
    if swapper is None:
        model_path = hf_hub_download(
            repo_id="ezioruan/inswapper_128.onnx",
            filename="inswapper_128.onnx",
            cache_dir="models"
        )
        swapper = get_model(
            model_path,
            providers=["CPUExecutionProvider"]
        )

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_image(url: str, save_path: str):
    r = requests.get(url, timeout=10, verify=False) 
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    return save_path

def faceswap(base_image_path: str, face_url: str) -> str:
    setup_faceswap()

    face_path = "input_face.jpg"
    download_image(face_url, face_path)

    base_img = cv2.imread(base_image_path)
    source_img = cv2.imread(face_path)

    base_faces = app.get(base_img)
    source_faces = app.get(source_img)

    if len(base_faces) == 0:
        raise RuntimeError("No face detected in generated image")
    if len(source_faces) == 0:
        raise RuntimeError("No face detected in input face image")

    swapped = swapper.get(base_img, base_faces[0], source_faces[0])

    out_path = "output_faceswap.png"
    cv2.imwrite(out_path, swapped)
    return out_path
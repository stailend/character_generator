# faceswap.py

import cv2
import requests
import urllib3
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from huggingface_hub import hf_hub_download


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = None
swapper = None


def setup_faceswap():
    global app, swapper

    if app is None:
        app = FaceAnalysis(name="buffalo_l")
        app.prepare(ctx_id=-1)  # CPU only

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


def download_image(url: str, save_path: str):
    r = requests.get(url, timeout=10, verify=False)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(r.content)
    return save_path


def _faceswap_internal(base_path: str, face_path: str) -> str:
    setup_faceswap()

    base_img = cv2.imread(base_path)
    source_img = cv2.imread(face_path)

    base_faces = app.get(base_img)
    source_faces = app.get(source_img)

    if len(base_faces) == 0:
        raise RuntimeError("No face detected in base image")
    if len(source_faces) == 0:
        raise RuntimeError("No face detected in face image")

    result = swapper.get(base_img, base_faces[0], source_faces[0])

    out_path = "final_image.png"
    cv2.imwrite(out_path, result)
    return out_path


def faceswap_from_url(base_path: str, face_url: str) -> str:
    face_path = "input_face.jpg"
    download_image(face_url, face_path)
    return _faceswap_internal(base_path, face_path)


def faceswap_from_local(base_path: str, face_path: str) -> str:
    return _faceswap_internal(base_path, face_path)
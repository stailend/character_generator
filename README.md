**Foto2Video – Multi-Stage AI Video Generator (Cog + Replicate)**

Foto2Video is a multi-stage AI pipeline that:
	1.	Generates a character image from text
	2.	Performs face swap with the user’s face
	3.	Creates a short video based on a scene prompt
	4.	Upscales and enhances the final video with film-style effects

The project is packaged with Cog and ready for deployment to Replicate.

⸻

🚀 Features
	•	Text-to-image character generation
	•	High-quality face swapping
	•	Image-to-video scene creation
	•	1080p video upscale + 60 FPS
	•	Film-style noise/sharpening via FFmpeg
	•	JSON output with all intermediate stages

⸻

🧠 Pipeline Overview

Stage 1 — Character Image
Model: black-forest-labs/flux-canny-dev

Stage 2 — Face Swap
Model: easel/advanced-face-swap

Stage 3 — Image → Video
Model: wan-video/wan-2.1-1.3b

Stage 4 — Video Enhancement
Model: topazlabs/video-upscale + FFmpeg

⸻

📂 Project Structure
```
.
├── stages/
│   ├── sdxl.py
│   ├── faceswap.py
│   ├── wan_remix.py
│   └── vfi_post.py
├── predict.py
├── cog.yaml
└── README.md
```

⸻

⚙️ Local Run

Install Cog:
```
pip install cog
```
Set your Replicate API token:
```
export REPLICATE_API_TOKEN="r8_xxxxx"
```
Run prediction:
```
cog predict \
  --env REPLICATE_API_TOKEN=$REPLICATE_API_TOKEN \
  -i face_img="https://example.com/face.jpg" \
  -i character_prompt="a medieval knight" \
  -i scene_prompt="cinematic forest battle"
```

⸻

📤 Output Format
```
{
  "image_raw": "https://...",
  "image_faceswapped": "https://...",
  "video_raw": "https://...",
  "video_final": "https://..."
}
```

⸻

📦 Deploy to Replicate
```
cog push r8-username/foto2video
```

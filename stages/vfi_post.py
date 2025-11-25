import subprocess


def upscale_fps(input_video: str, output_path: str) -> str:
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", "minterpolate=fps=60",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path


def film_effect(input_video: str, output_path: str) -> str:
    vf = (
        "format=yuv420p,"
        "noise=alls=8:allf=t+u,"
        "eq=contrast=1.1:brightness=0.02,"
        "unsharp=5:5:0.8:5:5:0.0"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    subprocess.run(cmd, check=True)
    return output_path
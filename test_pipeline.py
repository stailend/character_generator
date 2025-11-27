from stages.sdxl import generate_character_image
from stages.faceswap import faceswap
from stages.video_gen import generate_video

def main():
    img1 = generate_character_image("a woman with blue eyes in knight armor")
    img2 = faceswap(img1, "https://img.freepik.com/free-photo/portrait-white-man-isolated_53876-40306.jpg")

    print("== DONE == ")

if __name__ == "__main__":
    main()
    

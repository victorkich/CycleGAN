from cv2 import VideoCapture, imwrite
from tqdm import tqdm
import os

path = os.path.abspath(os.path.dirname(__file__))
images = [f"{path}/data/images_A", f"{path}/data/images_B"]
videos = f"{path}/data/videos"
listdir = os.listdir(videos)

# Parameter
interval = 1

for v, rep in enumerate(listdir):
    cap = VideoCapture(videos+'/'+rep)

    if not cap.isOpened():
        print("Error opening video stream or file")
        continue

    print(f"Extracting frames from video {rep}")
    for i in tqdm(range(0, int(cap.get(7)), interval)):
        ret, frame = cap.read()
        if ret:
            imwrite(f"{images[v]}/{int(i/interval)}.jpg", frame)
        else:
            break

    cap.release()

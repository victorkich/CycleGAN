from torchvision.utils import make_grid
from torch import load, cuda, device
from torchvision import transforms
from models import GeneratorResNet
from tqdm import tqdm
from PIL import Image
import torch
import cv2
import os

path = os.path.abspath(os.path.dirname(__file__))
path_video = f"{path}/data/videos/simulated_moto.mp4"
path_G_AB = f"{path}/saved_models/G_AB_300.pth"
path_G_BA = f"{path}/saved_models/G_BA_300.pth"

# Parameters
input_shape = (3, 196, 196)  # [c, h, w]
n_residual_blocks = 9  # number of residual blocks in generator
A2B = True  # filter direction

# Create a VideoCapture object
cap = cv2.VideoCapture(path_video)

# Use gpu if available
cuda_available = cuda.is_available()
device = device('cuda' if cuda_available else 'cpu')
print("PyTorch CUDA:", cuda_available)

# Load generator and discriminator model
model = GeneratorResNet(input_shape, n_residual_blocks).to(device)
model.load_state_dict(load(path_G_AB if A2B else path_G_BA, map_location=device))
model.eval()

# Define the codec and create VideoWriter object. The output is stored in 'test_model.mp4' file.
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('data/tests/model_test_2.mp4', fourcc, 24.0, (input_shape[2]*2, input_shape[1]), True)

transform = transforms.Compose([
    transforms.Resize((input_shape[1], input_shape[2])),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Recording video...")
for _ in tqdm(range(int(cap.get(7)))):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        real = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
        fake = model(real)

        # Arange images along x-axis
        real = make_grid(real, nrow=1, normalize=True)
        fake = make_grid(fake, nrow=1, normalize=True)

        # Arange images along y-axis
        image_grid = torch.cat((real, fake), 2).permute(1, 2, 0).cpu().numpy()

        # Converting to BGR and uint8
        image_grid = cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR)
        image_grid = cv2.normalize(image_grid, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Show and record video
        cv2.imshow("Frame", image_grid)
        out.write(image_grid)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break
print("Video recorded!")

# When everything done, release the video capture and video write objects
cap.release()
out.release()

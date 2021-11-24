from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.utils import save_image, make_grid
from torchvision import datasets, transforms
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from models import *
from utils import *
import numpy as np
import itertools
import datetime
import torch
import math
import time
import os

# Check GPU usage
cuda = torch.cuda.is_available()
device = torch.cuda.device('cuda' if cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
print("Using CUDA:", cuda)

# Parameters
input_shape = (3, 196, 196)  # [c, h, w]
batch_size = 4  # size of the batches
n_residual_blocks = 9  # number of residual blocks in generator
epoch = 0  # epoch to start training from
n_epochs = 1000  # number of epochs of training
n_workers = 8  # number of cpu threads to use during batch generation
decay_epoch = 50  # epoch from which to start lr decay
lr = 0.0002  # learning rate
b1 = 0.5  # decay of first order momentum of gradient
b2 = 0.999  # decay of first order momentum of gradient
lambda_cyc = 10.0  # cycle loss weight
lambda_id = 5.0  # identity loss weight
checkpoint_interval = 25  # interval between saving model checkpoints
sample_interval = 200  # interval between saving generator outputs
load_model = False  # load weights from a current pre trainded model

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, n_residual_blocks)
G_BA = GeneratorResNet(input_shape, n_residual_blocks)
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

# Load model or initialize weights from normal distribution
if load_model:
    G_AB.load_state_dict(torch.load(os.getcwd() + "/saved_models/G_AB_58.pth"))
    G_BA.load_state_dict(torch.load(os.getcwd() + "/saved_models/G_BA_58.pth"))
    D_A.load_state_dict(torch.load(os.getcwd() + "/saved_models/D_A_58.pth"))
    D_B.load_state_dict(torch.load(os.getcwd() + "/saved_models/D_B_58.pth"))
else:
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(b1, b2))
optimizer_D_A = Adam(D_A.parameters(), lr=lr, betas=(b1, b2))
optimizer_D_B = Adam(D_B.parameters(), lr=lr, betas=(b1, b2))

# Learning rate update schedulers
lr_scheduler_G = lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_A = lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_D_B = lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transform = [
    transforms.Resize((input_shape[1], input_shape[2])),
    transforms.RandomCrop((input_shape[1], input_shape[2])),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


class RiGANDataset(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.cwd = os.getcwd()
        self.files_A = os.listdir(self.cwd + "/data/images_A/")
        self.files_B = os.listdir(self.cwd + "/data/images_B/")
        self.transform = transforms.Compose(transform)

    def __repr__(self):
        return f"Dataset class with {self.__len__()} files"

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):
        img_A = Image.open(self.cwd + "/data/images_A/" + self.files_A[idx % len(self.files_A)])
        img_B = Image.open(self.cwd + "/data/images_B/" + self.files_B[np.random.randint(0, len(self.files_B) - 1)])
        item_A = self.transform(img_A)
        item_B = self.transform(img_B)
        return {"A": item_A, "B": item_B}


# Loading the train and val dataset using data loader
dataset = RiGANDataset(transform=transform)
lengths = [round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
train_data = Subset(dataset, range(0, lengths[0]))
val_data = Subset(dataset, range(lengths[0], sum(lengths)))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
val_loader = DataLoader(val_data, batch_size=5, shuffle=True, num_workers=1)


def sample_images(batches_done):
    """ Saves a generated sample from the test set """
    imgs = next(iter(val_loader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)

    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)

    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, os.getcwd() + "/data/outputs/%s.png" % batches_done, normalize=False)


#  Training
prev_time = time.time()
for epoch in range(epoch, n_epochs):
    for i, batch in enumerate(train_loader):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # Train Generators
        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator A
        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)

        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)

        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # Train Discriminator B
        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)

        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)

        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # Log Progress
        # Determine approximate time left
        batches_done = epoch * len(train_loader) + i
        batches_left = n_epochs * len(train_loader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
            % (
                epoch,
                n_epochs,
                i,
                len(train_loader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), os.getcwd() + "/saved_models/G_AB_%d.pth" % epoch)
        torch.save(G_BA.state_dict(), os.getcwd() + "/saved_models/G_BA_%d.pth" % epoch)
        torch.save(D_A.state_dict(), os.getcwd() + "/saved_models/D_A_%d.pth" % epoch)
        torch.save(D_B.state_dict(), os.getcwd() + "/saved_models/D_B_%d.pth" % epoch)

# Utils
from __future__ import print_function

# %matplotlib inline
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
from torchvision.utils import make_grid
from keras.models import load_model
from DistributedGanGPUTensorflow import C


class Util:
    def __init__(self, dataset, dataloader):
        self.dataset = dataset
        self.dataloader = dataloader
        self.device = self.get_default_device()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def show_example(self, img, label):
        print('Label: ', self.dataset.classes[label], "(" + str(label) + ")")
        plt.imshow(self.denorm(img)[0], cmap='gray')

    def save_samples(self, index, latent_tensors, sample_dir, generator, show=True):
        fake_images = generator(latent_tensors)
        fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
        vutils.save_image(self.denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(self.denorm(fake_images).cpu().detach(), nrow=8).permute(1, 2, 0))

    def show_batch(self, dl):
        for images, labels in dl:
            print(images, flush=True)
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=10, padding=2, normalize=True).permute(1, 2, 0))
            break

    def show_batch_2(self, dl):
        print("asdasd")
        real_batch = next(iter(self.dataloader))
        print(real_batch, flush=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(self.device)[:64],
                                                 padding=2, normalize=True).cpu(), (1, 2, 0)))





    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield self.to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    def to_device(self, data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list, tuple)):
            return [self.to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)


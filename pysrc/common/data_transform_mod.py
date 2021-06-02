from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms

class DataTransform():
    def __init__(self, resize, mean, std):
        self.mean = mean
        self.std = std
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img_pil, acc_numpy, phase="train"):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        ## acc: numpy -> tensor
        acc_numpy = acc_numpy.astype(np.float32)
        acc_numpy = acc_numpy / np.linalg.norm(acc_numpy)
        acc_tensor = torch.from_numpy(acc_numpy)
        return img_tensor, acc_tensor
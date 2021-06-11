from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import random
import math

import torch
from torchvision import transforms
from torchvision.transforms.transforms import Resize

class DataTransform():
    def __init__(self, resize, mean, std):
        self.mean = mean
        self.std = std
        size = (resize, resize)
        self.img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    def __call__(self, img_pil, roll_numpy, pitch_numpy, phase="train"):
        ## img: numpy -> tensor
        img_tensor = self.img_transform(img_pil)
        
        ## roll: numpy -> tensor
        roll_numpy = roll_numpy.astype(np.float32)
        roll_numpy = roll_numpy / np.linalg.norm(roll_numpy)
        #roll_tensor = torch.from_numpy(roll_numpy)
        roll_tensor = torch.tensor(roll_numpy, requires_grad=True)
        #roll_tensor = roll_tensor.long()

        # pitch: numpy -> tensor
        pitch_numpy = pitch_numpy.astype(np.float32)
        pitch_numpy = pitch_numpy / np.linalg.norm(pitch_numpy)
        #pitch_tensor = torch.from_numpy(pitch_numpy)
        pitch_tensor = torch.tensor(pitch_numpy, requires_grad = True)
        #pitch_tensor = pitch_tensor.long()

        return img_tensor, roll_tensor, pitch_tensor
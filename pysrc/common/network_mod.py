from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False):
        super(Network, self).__init__()

        self.kernel_size = 3
        self.padding = [self.kernel_size // 2, self.kernel_size//2]

        self.cnn = nn.Sequential(
            nn.Conv2d(   1,  64, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(  64, 128, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 128, 256, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 256, 512, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 512, 1024, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.dim_fc_in = (1024)*(resize//32)*(resize//32)
        self.dim_fc_out = dim_fc_out

        self.roll_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 3000),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 3000, 1000),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 1000, self.dim_fc_out)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 3000),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 3000, 1000),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 1000, self.dim_fc_out)
        )

    def getParamValueList(self):
        list_cnn_param_value = []
        list_roll_fc_param_value = []
        list_pitch_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                list_cnn_param_value.append(param_value)
            if "roll_fc" in param_name:
                list_roll_fc_param_value.append(param_value)
            if "pitch_fc" in param_name:
                list_pitch_fc_param_value.append(param_value)
            
        return list_cnn_param_value, list_roll_fc_param_value, list_pitch_fc_param_value
    
    def forward(self, x):
        feature = self.cnn(x)

        feature = torch.flatten(feature)

        roll = self.roll_fc(feature)
        pitch = self.pitch_fc(feature)

        l2norm = torch.norm( roll[:, :self.dim_fc_out], p=2, dim=1, keepdims=True)
        roll[: , :self.dim_fc_out] = torch.div( roll[: , :self.dim_fc_out].clone(), l2norm)

        l2norm = torch.norm( pitch[:, :self.dim_fc_out], p=2, dim=1, keepdims=True)
        pitch[: , :self.dim_fc_out] = torch.div( pitch[: , :self.dim_fc_out].clone(), l2norm)
        return roll, pitch
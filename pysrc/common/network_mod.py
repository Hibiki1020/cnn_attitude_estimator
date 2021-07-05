from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as nn_functional

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False):
        super(Network, self).__init__()

        self.kernel_size = 3
        self.padding = [self.kernel_size // 2, self.kernel_size//2]

        self.cnn = nn.Sequential(
            nn.Conv2d(   1,  64, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(  64, 128, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 128, 256, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 256, 512, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 512, 1024, self.kernel_size, padding=self.padding, bias=False, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.dim_fc_in = (1024)*(7)*(7)
        self.dim_fc_out = dim_fc_out

        self.roll_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 3000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 3000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 1000, self.dim_fc_out)
        )

        self.pitch_fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, 3000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 3000, 1000),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear( 1000, self.dim_fc_out)
        )

        self.initializeWeights()#no need?

    def initializeWeights(self):
        for m in self.roll_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

        for m in self.pitch_fc.children():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
    
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

        feature = torch.flatten(feature, 1)

        roll = self.roll_fc(feature)
        pitch = self.pitch_fc(feature)

        #roll = nn_functional.softmax(roll, dim=0)
        #pitch = nn_functional.softmax(pitch, dim=0)

        logged_roll = nn_functional.log_softmax(roll, dim=0)
        logged_pitch = nn_functional.log_softmax(pitch, dim=0)

        print(logged_roll.dim())

        #l2norm = torch.norm( roll[:, :self.dim_fc_out], p=2, dim=1, keepdim=True)
        #roll[: , :self.dim_fc_out] = torch.div( roll[: , :self.dim_fc_out].clone(), l2norm)

        #l2norm = torch.norm( pitch[:, :self.dim_fc_out], p=2, dim=1, keepdim=True)
        #pitch[: , :self.dim_fc_out] = torch.div( pitch[: , :self.dim_fc_out].clone(), l2norm)
        return logged_roll, logged_pitch
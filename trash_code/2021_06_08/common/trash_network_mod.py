from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False):
        super(Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(   1,  64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(  64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 128, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 256, 512, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d( 512, 1024, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.dim_fc_in = 1024*(resize//32)*(resize//32)
        self.dim_fc_out = dim_fc_out

        self.fc = nn.Sequential(
            nn.Linear(self.dim_fc_in, self.dim_fc_out),
            nn.Dropout(p=dropout_rate),
            nn.Linear( self.dim_fc_out, self.dim_fc_out),
            nn.Dropout(p=dropout_rate),
            nn.Linear( self.dim_fc_out, self.dim_fc_out)
        )

    def getParamValueList(self):
        list_cnn_param_value = []
        list_fc_param_value = []
        for param_name, param_value in self.named_parameters():
            param_value.requires_grad = True
            if "cnn" in param_name:
                # print("cnn: ", param_name)
                list_cnn_param_value.append(param_value)
            if "fc" in param_name:
                # print("fc: ", param_name)
                list_fc_param_value.append(param_value)
        # print("list_cnn_param_value: ",list_cnn_param_value)
        # print("list_fc_param_value: ",list_fc_param_value)
        return list_cnn_param_value, list_fc_param_value
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)

        l2norm = torch.norm( x[:, :self.dim_fc_out], p=2, dim=1, keepdims=True)
        x[: , :self.dim_fc_out] = torch.div( x[: , :self.dim_fc_out].clone(), l2norm)
        return x

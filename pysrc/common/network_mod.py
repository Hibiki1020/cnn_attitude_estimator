from PIL import Image
import numpy as np

import torch
from torchvision import models
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, resize, dim_fc_out, dropout_rate):
        super(Network, self).__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(   1,  64, 3) #Input Image is 672*672*1
        self.conv2 = nn.Conv2d(  64, 128, 4)
        self.conv3 = nn.Conv2d( 128, 256, 3)
        self.conv4 = nn.Conv2d( 256, 512, 3)
        self.conv5 = nn.Conv2d( 512, 1024, 3)

        self.dim_fc_in = 1024
        self.dim_fc_out = dim_fc_out

        #test code
        self.fc1 = nn.Linear(self.dim_fc_in, 100)
        self.fc2 = nn.Linear( 100, dim_fc_out)
        self.fc3 = nn.Linear( dim_fc_out, dim_fc_out)

        #self.fc1 = nn.Linear(dim_fc_in, dim_fc_out)
        #self.fc2 = nn.Linear( dim_fc_out, dim_fc_out)
        #self.fc3 = nn.Linear( dim_fc_out, dim_fc_out)

        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x1)
        x3 = self.pool(x2)

        x4 = self.conv2(x3)
        x5 = self.relu(x4)
        x6 = self.pool(x5)

        x7 = self.conv3(x6)
        x8 = self.relu(x7)
        x9 = self.pool(x8)

        x10 = self.conv4(x9)
        x11 = self.relu(x10)
        x12 = self.pool(x11)

        x13 = self.conv5(x12)
        x14 = self.relu(x13)
        x15 = self.pool(x14)

        x16 = self.fc1(x15)
        x17 = self.dropout(x16)

        x18 = self.fc2(x17)
        x19 = self.dropout(x18)

        x20 = self.fc3(x19)

        l2norm = torch.norm( x20[:, :self.dim_fc_out], p=2, dim=1, keepdims=True)
        x20[: , :self.dim_fc_out] = torch.div( x20[: , :self.dim_fc_out].clone(), l2norm)

        return x

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

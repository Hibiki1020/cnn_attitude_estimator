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

        dim_fc_in = 1024*(resize//32)*(resize//32)

        self.fc1 = nn.Linear(dim_fc_in, dim_fc_out)
        self.fc2 = nn.Linear( dim_fc_out, dim_fc_out)
        self.fc3 = nn.Linear( dim_fc_out, dim_fc_out)

        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.dropout(x)

        x = self.fc3(x)

        l2norm = torch.norm( x[:, :32400], p=2, dim=1, keepdims=True)
        x[: , :32400] = torch.div( x[: , :32400].clone(), l2norm)

        return x
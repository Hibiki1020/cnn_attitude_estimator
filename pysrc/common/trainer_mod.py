from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

class Trainer:
    def __init__(self,
    method_name,
    train_dataset,
    valid_dataset,
    net,
    criterion,
    optimizer_name,
    lr_cnn,
    lr_fc,
    batch_size,
    num_epochs,
    weights_path,
    log_path,
    graph_path):

    self.weights_path = weights_path
    self.log_path = log_path
    self.graph_path = graph_path

    self.setRandomCondition()
    self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    print("Training Device: ", self.device)
    


def setRandomCondition(self, keep_reproducibility=False): #Random Training Environment

        #Refer https://nuka137.hatenablog.com/entry/2020/09/01/080038

        if keep_reproducibility:
            torch.manual_seed(19981020)
            np.random.seed(19981020)
            random.seed(19981020)
            torch.backends.cudnn.deterministic = True #https://qiita.com/chat-flip/items/c2e983b7f30ef10b91f6
            torch.backends.cudnn.benchmark = False
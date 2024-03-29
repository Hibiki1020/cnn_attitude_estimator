from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod
from common import make_datalist_mod

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default="../../pyyaml/train_config.yaml",
        help='Train hyperparameter config file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    method_name = CFG["method_name"]
    save_top_path = CFG["save_top_path"]
    weights_path = CFG["save_top_path"] + CFG["weights_path"]
    log_path = CFG["save_top_path"] + CFG["log_path"]
    graph_path = CFG["save_top_path"] + CFG["graph_path"]
    csv_name = CFG["csv_name"]
    index_csv_path = CFG["index_csv_path"]

    train_sequences = CFG["train"]
    valid_sequences = CFG["valid"]

    #get hyperparameter for learning
    original_size = CFG["hyperparameter"]["original_size"]
    resize = CFG["hyperparameter"]["resize"]
    mean_element = CFG["hyperparameter"]["mean_element"]
    std_element = CFG["hyperparameter"]["std_element"]
    hor_fov_deg = CFG["hyperparameter"]["hor_fov_deg"]
    optimizer_name = CFG["hyperparameter"]["optimizer_name"]
    lr_cnn = float(CFG["hyperparameter"]["lr_cnn"])
    lr_fc = float(CFG["hyperparameter"]["lr_fc"])
    batch_size = CFG["hyperparameter"]["batch_size"]
    num_epochs = CFG["hyperparameter"]["num_epochs"]
    dropout_rate = float(CFG["hyperparameter"]["dropout_rate"])
    dim_fc_out = int(CFG["hyperparameter"]["dim_fc_out"])

    try:
        print("Copy files to %s for further reference." % log_path)
        copyfile(FLAGS.train_cfg, log_path + "/train_config.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting....")
        quit()

    train_dataset = dataset_mod.ClassOriginaldataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "train",
        index_dict_path = index_csv_path,
    )

    valid_dataset = dataset_mod.ClassOriginaldataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "valid",
        index_dict_path = index_csv_path,
    )

    ##Network
    net = network_mod.Network(resize, dim_fc_out, dropout_rate)


    ##Criterion
    criterion = nn.MSELoss()

    #train
    trainer = trainer_mod.Trainer(
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
        graph_path
    )

    trainer.train()

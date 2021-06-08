import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
import yaml
import os
import csv

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms


import sys
sys.path.append('../')
from common import network_mod

class CNNAttitudeEstimator:
    def __init__(self, CFG):
        print("CNN Attitude Estimator")
        print("Frame Infer")

        self..CFG = CFG
        self.method_name = CFG["method_name"]
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]
        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]
        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)
        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]
        self.frame_id = CFG["frame_id"]
        self.window_size = CFG["window_size"]
        self.mean_element = CFG["mean_element"]
        self.std_element = CFG["std_element"]
        self.dropout_rate = CFG["dropout_rate"]

        #OpenCV
        self.color_img_cv = np.empty(0)

        #DNN
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)
        self.net = self.getNetwork(self.window_size, self.weights_path, self.dropout_rate)

    def getNetwork(self, window_size, weights_path, dropout_rate):
        net = network_mod.Network(dim_out=32761, dropout_rate=dropout_rate)
        print(net)

        net.to(self.device)
        net.eval()

        #load
        if torch.cuda.is_available():
            loaded_weights = torch.load(weights_path)
            print("GPU  ==>  GPU")
        else:
            loaded_weights = torch.load(weights_path, map_location={"cuda:0": "cpu"})
            print("GPU  ==>  CPU")

        net.load_state_dict(loaded_weights)
        return net

    def getImageTransform(self, resize, mean_element, std_element):

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        return img_transform

    def spin():
        print("Start Infer")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./frame_infer.py")
    parser.add_argument(
        '--frame_infer_config', '-fic',
        type=str,
        required=False,
        default="../../pyyaml/frame_infer_config.yaml"
    )

    FLAGS, unparsed = parser.parse_args()

    #Load .yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.frame_infer_config)
        quit()

    cnn_attitude_estimator = CNNAttitudeEstimator(CFG)
    cnn_attitude_estimator.spin()
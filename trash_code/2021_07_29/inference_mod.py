import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
from numpy.core.fromnumeric import argmin
import yaml
import os
import csv
import random
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.stats as stats

from sklearn.mixture import GaussianMixture

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as nn_functional

from collections import OrderedDict

class InferenceMod:
    def __init__(self, CFG, net):
        self.CFG = CFG
        self.method_name = CFG["method_name"]
        
        self.infer_dataset_top_directory = CFG["infer_dataset_top_directory"]
        self.csv_name = CFG["csv_name"]

        self.weights_top_directory = CFG["weights_top_directory"]
        self.weights_file_name = CFG["weights_file_name"]

        self.weights_path = os.path.join(self.weights_top_directory, self.weights_file_name)

        self.infer_log_top_directory = CFG["infer_log_top_directory"]
        self.infer_log_file_name = CFG["infer_log_file_name"]

        self.index_dict_name = CFG["index_dict_name"]
        self.index_dict_path = "../../index_dict/" + self.index_dict_name

        self.window_original_size = int(CFG["window_original_size"])
        self.window_num = int(CFG["window_num"])
        self.resize = int(CFG["resize"])
        self.mean_element = float(CFG["mean_element"])
        self.std_element = float(CFG["std_element"])
        self.dim_fc_out = int(CFG["dim_fc_out"])
        self.dropout_rate = float(CFG["dropout_rate"])
        self.corner_threshold = int(CFG["corner_threshold"])

        self.color_img_cv = np.empty(0)

        self.image_data_list = []
        self.ground_truth_list = []

        #Using only 1 GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)
        self.net = self.getNetwork(net)

        self.value_dict = []

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)
    
    def getImageTransform(self,resize,mean_element,std_element):

        mean = mean_element
        std = std_element
        size = (resize, resize)

        '''
        img_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((mean_element,), (std_element,))
        ])

        '''
        img_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

        return img_transform

    def getNetwork(self, net):
        print(net)

        net.to(self.device)
        net.eval()
        #net.train()

        #load
        if torch.cuda.is_available():
            state_dict = torch.load(self.weights_path, map_location=lambda storage, loc: storage)
            print("GPU  ==>  GPU")
        else:
            state_dict = torch.load(self.weights_path, map_location={"cuda:0": "cpu"})
            print("GPU  ==>  CPU")
        

        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' in k:
                k = k.replace('module.', '')
            new_state_dict[k] = v

        net.load_state_dict(new_state_dict)
        return net

    def spin(self):
        self.image_data_list, self.ground_truth_list = self.get_data()

    def get_data(self):
        image_data_list = []
        data_list = []
        
        csv_path = os.path.join(self.infer_dataset_top_directory, self.csv_name)

        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img_path = os.path.join(self.infer_dataset_top_directory, row[0])
                gt_roll = float(row[4])/3.141592*180.0
                gt_pitch = float(row[5])/3.141592*180.0

                image_data_list.append(img_path)
                tmp_row = [row[0], gt_roll, gt_pitch]
                data_list.append(tmp_row)

        return image_data_list, data_list
    
    def check_window(self, window): #Bottle Neck
        #fast = cv2.FastFeatureDetector()
        detector = cv2.ORB_create()
        #print("Detect corner")
        #keypoints = fast.detect(window, None)
        keypoints = detector.detect(window)

        window_checker = False

        if len(keypoints) > self.corner_threshold:
            window_checker = True

        return window_checker

    def extract_window(self, image_original):
        height = image_original.shape[0]
        width = image_original.shape[1]

        windows = []
        correct_windows = []
        tmp_windows = []        

        total_window_checker = False
        window_count = 0
        error_count = 0

        while total_window_checker==False:
            width_start = random.randint(0, int(width)-self.window_original_size)
            height_start = random.randint(0, int(height)-self.window_original_size)

            window = image_original[height_start:(height_start + self.window_original_size), width_start:(width_start + self.window_original_size)]
            #cv2.imshow('window',window)
            #tmp_window_checker = self.check_window(window)
            tmp_window_checker = True

            if tmp_window_checker == True:
                window_count += 1
                correct_windows.append(window)
                tmp_windows.append(window)

                if window_count >= self.window_num:
                    total_window_checker = True
                    windows = correct_windows
            else:
                error_count += 1
                tmp_windows.append(window)

                if error_count >=self.window_num:
                    print("Less Feature Point...")
                    total_window_checker = True
                    windows = tmp_windows

        return windows

    def cvToPIL(self, img_cv):
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_cv)
        return img_pil

    def transformImage(self, inference_image):
        ## color
        img_pil = self.cvToPIL(inference_image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        #print(inputs)
        return inputs

    def prediction(self, input_image):
        logged_output_roll_array, logged_output_pitch_array, roll_array, pitch_array = self.net(input_image)

        output_roll_array = roll_array.to('cpu').detach().numpy().copy()
        output_pitch_array = pitch_array.to('cpu').detach().numpy().copy()

        #print(output_roll_array)

        return np.array(output_roll_array), np.array(output_pitch_array)

    def normalize(self, v):
        l2 = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
        l2[l2==0] = 1
        return v/l2

    def array_to_value_simple(self, output_array):
        max_index = int(np.argmax(output_array))
        plus_index = max_index + 1
        minus_index = max_index - 1

        value = 0.0

        if max_index == 0:
            value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][max_index+1]*self.value_dict[max_index+1]
        elif max_index == 360: #361
            value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][max_index-1]*self.value_dict[max_index-1]
        else:
            if output_array[0][minus_index] > output_array[0][plus_index]: #一つ前のインデックスを採用
                value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][minus_index]*self.value_dict[minus_index]
            elif output_array[0][minus_index] < output_array[0][plus_index]: #一つ後のインデックスを採用
                value = output_array[0][max_index]*self.value_dict[max_index] + output_array[0][plus_index]*self.value_dict[plus_index]

        return value

    def inference(self):
        print("Start Inference")

        result_csv = []

        infer_count = 0

        for (img_path, ground_truth) in zip(self.image_data_list, self.ground_truth_list):
            print("---------Inference at " + str(infer_count) + "---------")
            infer_count += 1
            #image_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Load Image
            image_original = cv2.imread(img_path)
            #cv2.imshow('image',image_original)

            windows = self.extract_window(image_original)
            print("Transform input image")
            print("---------------------")

            start_clock = time.time()

            result = []

            roll_result_list = []
            pitch_result_list = []

            roll_hist_array = []
            pitch_hist_array = []

            for i in range(361):
                tmp = 0.0
                roll_hist_array.append(tmp)
                pitch_hist_array.append(tmp)

            for window in windows:
                inference_image = window
                input_image = self.transformImage(inference_image)

                roll_output_array, pitch_output_array = self.prediction(input_image)

                tmp_roll = self.array_to_value_simple(roll_output_array)
                tmp_pitch = self.array_to_value_simple(pitch_output_array)

                roll_hist_array += roll_output_array[0]
                pitch_hist_array += pitch_output_array[0]

                tmp_result = [tmp_roll, tmp_pitch]
                
                roll_result_list.append(tmp_roll)
                pitch_result_list.append(tmp_pitch)

                result.append(tmp_result)

            roll_hist_array /= float(len(windows))
            pitch_hist_array /= float(len(windows))

            np_result = np.array(result)

            roll = np.mean(tmp_roll)
            pitch = np.mean(tmp_pitch)

            print("Infered Roll:  " + str(roll) +  "[deg]")
            print("GT Roll:       " + str(ground_truth[1]) + "[deg]")
            print("Infered Pitch: " + str(pitch) + "[deg]")
            print("GT Pitch:      " + str(ground_truth[2]) + "[deg]")

            tmp_result_csv = [roll, pitch, ground_truth[1], ground_truth[2]]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")

        return result_csv

    def save_csv(self, result_csv):
        result_csv_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")
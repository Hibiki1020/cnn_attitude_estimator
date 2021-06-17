import cv2
import PIL.Image as Image
import math
import numpy as np
import time
import argparse
import yaml
import os
import csv
import random

import torch
from torchvision import models
import torch.nn as nn
from torchvision import transforms

import sys
sys.path.append('../')
from common import network_mod

class CNNAttitudeEstimator:
    def __init__(self, CFG):
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

        #Using only 1 GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device ==> ", self.device)

        self.img_transform = self.getImageTransform(self.resize, self.mean_element, self.std_element)
        self.net = self.getNetwork(self.resize, self.weights_path, self.dim_fc_out, self.dropout_rate)

        self.value_dict = []

        with open(self.index_dict_path) as fd:
            reader = csv.reader(fd)
            for row in reader:
                num = float(row[0])
                self.value_dict.append(num)
            reader.close()
        

    def getImageTransform(self,resize,mean_element,std_element):

        img_transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((mean_element,), (std_element,))
        ])

        return img_transform

    def getNetwork(self, resize, weights_path, dim_fc_out, dropout_rate):
        net = network_mod.Network(resize, dim_fc_out, dropout_rate, use_pretrained_vgg=False)

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

    def spin(self):
        self.image_data_list, self.ground_truth_list = self.get_data()
        self.result_csv = self.frame_infer(self.image_data_list, self.ground_truth_list)
        self.save_csv(self.result_csv)

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
            reader.close()

        return image_data_list, data_list

    def check_window(self, window):
        fast = cv2.FastFeatureDetector()
        keypoints = fast.detect(window, None)

        window_checker = False

        if keypoints.size() > self.corner_threshold:
            window_checker = True

        return window_checker

    def extract_window(self, image_original):
        height, width, channels = image_original.shape[ :3]

        windows = []
        correct_windows = []
        tmp_windows = []        

        total_window_checker = False
        window_count = 0
        error_count = 0

        while total_window_checker==False:
            width_start = random.randint(0, width-self.window_original_size)
            height_start = random.randint(0, height-self.window_original_size)

            window = image_original[height_start:self.window_original_size, width_start:self.window_original_size]
            tmp_window_checker = self.check_window(window)

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

    def transformImage(self):
        ## color
        img_pil = self.cvToPIL(self.inference_image)
        img_tensor = self.img_transform(img_pil)
        inputs = img_tensor.unsqueeze_(0)
        inputs = inputs.to(self.device)
        return inputs

    def prediction(self, input_image):
        output_roll_array, output_pitch_array = self.net(input_image)
        output_roll_array = output_roll_array.cpu().detach().numpy()[0]
        output_pitch_array = output_pitch_array.cpu().detach().numpy()[0]

        return np.array(output_roll_array), np.array(output_pitch_array)

    def normalize(self, v):
        l2 = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
        l2[l2==0] = 1
        return v/l2

    def array_to_value_simple(self, output_array):
        max_index = np.argmax(output_array)

        value = 0.0

        if max_index == 0:
            value = output_array[max_index]*self.value_dict[max_index] + output_array[max_index+1]*self.value_dict[max_index+1]
        elif max_index == int(self.dim_fc_out): #361
            value = output_array[max_index]*self.value_dict[max_index] + output_array[max_index-1]*self.value_dict[max_index-1]
        else:
            if output_array[max_index-1] > output_array[max_index+1]: #一つ前のインデックスを採用
                value = output_array[max_index]*self.value_dict[max_index] + output_array[max_index-1]*self.value_dict[max_index-1]
            elif output_array[max_index-1] < output_array[max_index+1]: #一つ後のインデックスを採用
                value = output_array[max_index]*self.value_dict[max_index] + output_array[max_index+1]*self.value_dict[max_index+1]
        
        return value

    def save_csv(self, result_csv):
        result_csv_path = os.path.join(self.infer_log_top_directory, self.infer_log_file_name)
        csv_file = open(result_csv_path, 'w')
        csv_w = csv.writer(csv_file)
        for row in result_csv:
            csv_w.writerow(row)
        csv_file.close()
        print("Save Inference Data")

    def frame_infer(self, image_data_list, ground_truth_list):
        print("Start Inference")

        result_csv = []

        for img_path, ground_truth in image_data_list, ground_truth_list:
            print("---------------------")
            image_original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) #Load Image
            windows = self.extract_window(image_original)

            print("Transform input image")
            print("---------------------")

            start_clock = time.time()

            result = []      

            for window in windows:
                self.inference_image = window
                input_image = self.transformImage()

                roll_output_array, pitch_output_array = self.prediction(input_image)
                roll_output_array = self.normalize(roll_output_array)
                pitch_output_array = self.normalize(pitch_output_array)

                tmp_roll = self.array_to_value_simple(roll_output_array)
                tmp_pitch = self.array_to_value_simple(pitch_output_array)

                tmp_result = [tmp_roll, tmp_pitch]
                result.append(tmp_result)

            np_result = np.array(result)

            roll = np.mean(np_result, axis=0)
            pitch = np.mean(np_result, axis=1)

            cov = np.cov(np_result)

            tmp_result_csv = [roll, pitch, ground_truth[1], ground_truth[2]]
            result_csv.append(tmp_result_csv)

            print("Period [s]: ", time.time() - start_clock)
            print("---------------------")

            print("\n")
            print("\n")

        return result_csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./frame_infer.py")
    parser.add_argument(
        '--frame_infer_config', '-fic',
        type=str,
        required=False,
        default='../../pyyaml/frame_infer_config.yaml',
        help='Frame Infer Config'
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening frame infer config file %s", FLAGS.frame_infer_config)
        CFG = yaml.safe_load(open(FLAGS.frame_infer_config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening frame infer config file %s", FLAGS.frame_infer_config)
        quit()
    
    cnn_attitude_estimator = CNNAttitudeEstimator(CFG)
    cnn_attitude_estimator.spin()
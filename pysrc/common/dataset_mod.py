 
import torch.utils.data as data
from PIL import Image
import numpy as np
import math
import csv

class ClassOriginaldataset(data.Dataset):
    def __init__(self, data_list, transform, phase, index_dict_path):
        self.data_list = data_list
        self.transform = transform
        self.phase = phase
        self.index_dict_path = index_dict_path

        self.index_dict = []

        with open(index_dict_path) as f:
            reader = csv.reader(f)
            for row in reader:
                self.index_dict.append(row)

    def search_index(self, number):
        index = int(1000000000)
        for row in self.index_dict:
            if float(number) == float(row[0]):
                index = int(row[1])
                break
        
        return index

    def float_to_array(self, num_float):
        num_deg = float(num_float / 3.141592 * 180.0)

        num_upper = 0.0
        num_lower = 0.0

        tmp_deg = float(int(num_deg))
        if tmp_deg < num_deg: # 0 < num_deg
            num_lower = tmp_deg
            num_upper = num_lower + 1.0
        elif num_deg < tmp_deg: # tmp_deg < 0
            num_lower = tmp_deg - 1.0
            num_upper = tmp_deg
        
        dist_low = math.sqrt(num_deg - num_lower)
        dist_high = math.sqrt(num_deg - num_upper)

        lower_ind = self.search_index(num_lower)
        upper_ind = self.search_index(num_upper)

        array = np.zeros(len(self.index_dict))
        for i in range(len(array)):
            array[i] = 0.0
        
        array[lower_ind] = dist_high
        array[upper_ind] = dist_low

        return array

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]

        roll_str = self.data_list[index][4]
        pitch_str = self.data_list[index][5]

        roll_float = [float(num) for num in roll_str]
        pitch_float = [float(num) for num in pitch_str]

        roll_list = self.float_to_array(roll_float)
        pitch_list = self.float_to_array(pitch_float)

        img_pil = Image.open(img_path)
        roll_numpy = np.array(roll_list)
        pitch_numpy = np.array(pitch_list)

        roll_img_trans, roll_trans = self.transform(img_pil, roll_numpy, phase=self.phase)
        pitch_img_trans, pitch_trans = self.transform(img_pil, pitch_numpy, phase=self.phase)

        return roll_img_trans, roll_trans, pitch_trans
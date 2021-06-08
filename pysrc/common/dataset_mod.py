 
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

    def float_to_array(self, list_float):
        print("a")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]

        roll_list_str = self.data_list[index][4]
        pitch_list_str = self.data_list[index][5]

        roll_list_float = [float(num) for num in roll_list_str]
        pitch_list_float = [float(num) for num in pitch_list_str]

        roll_list = self.float_to_array(roll_list_float)
        pitch_list = self.float_to_array(pitch_list_float)

        img_pil = Image.open(img_path)
        roll_numpy = np.array(roll_list)
        pitch_numpy = np.array(pitch_list)

        roll_img_trans, roll_trans = self.transform(img_pil, roll_numpy, phase=self.phase)
        pitch_img_trans, pitch_trans = self.transform(img_pil, pitch_numpy, phase=self.phase)

        return roll_img_trans, roll_trans, pitch_img_trans, pitch_trans



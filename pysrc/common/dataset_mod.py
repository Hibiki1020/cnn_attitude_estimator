import torch.utils.data as data
from PIL import Image
import numpy as np

import array_generator

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

    def float_to_array(self, rp_list_float):
        roll_deg = rp_list_float[0]/M_PI*180.0
        pitch_deg = rp_list_float[1]/M_PI*180.0

        upper_roll = 0.0
        lower_roll = 0.0

        upper_pitch = 0.0
        lower_pitch = 0.0

        tmp_roll = float(int(roll_deg))
        tmp_pitch = float(int(pitch_deg))

        if( tmp_roll < roll_deg ): # roll > 0.0[deg]
            if tmp_roll%2 != 0:
                tmp_roll = tmp_roll - 1.0
                lower_roll = tmp_roll
                upper_roll = lower_roll + 2.0
            else:
                lower_roll = tmp_roll
                upper_roll = lower_roll + 2.0
        elif( tmp_roll > roll_deg):# roll < 0.0[deg]
            if tmp_roll%2 != 0:
                tmp_roll = tmp_roll + 1.0
                upper_roll = tmp_roll
                lower_roll = upper_roll - 2.0
            else:
                upper_roll = tmp_roll
                lower_roll = upper_roll - 2.0

        if( tmp_pitch < pitch_deg ): # pitch > 0.0[deg]
            if tmp_pitch%2 != 0:
                tmp_pitch = tmp_pitch - 1.0
                lower_pitch = tmp_pitch
                upper_pitch = lower_pitch + 2.0
            else:
                lower_pitch = tmp_pitch
                upper_pitch = lower_pitch + 2.0
        elif( tmp_pitch > pitch_deg):# roll < 0.0[deg]
            if tmp_pitch%2 != 0:
                tmp_pitch = tmp_pitch + 1.0
                upper_pitch = tmp_pitch
                lower_pitch = upper_pitch - 2.0
            else:
                upper_pitch = tmp_pitch
                lower_pitch = upper_pitch - 2.0



    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        rp_list = self.data[index][3:6]
        rp_list_float = [float(num) for num in rp_list]
        rp_list = float_to_array(rp_list_float) #Convert to 32400 array

        img_pil = Image.open(img_path)
        rp_numpy = np.array(rp_list)

        img_trans, rp_trans = self.transform(img_pil, rp_numpy, phase=self.phase)
        return img_trans, rp_trans

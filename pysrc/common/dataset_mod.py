import torch.utils.data as data
from PIL import Image
import numpy as np
import math
import csv

#バグが一切存在してはいけないプログラム。要確認

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

    def get_distance(self, x1, y1, x2, y2):
        d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return d
    
    def norm(self, X, M, m, max, min):
        y = (X-min)/(max-min)*(M-m) + m
        return y

    def search_index(self, roll, pitch):
        index = -1
        for row in self.index_dict:
            if roll==row[0] and pitch==row[1]:
                index = row[4]
                break
        
        return index

    def float_to_array(self, rp_list_float):
        roll_deg = rp_list_float[0]/3.141592*180.0
        pitch_deg = rp_list_float[1]/3.141592*180.0

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

        r_ud = [upper_roll, upper_pitch]
        r_ud_dist = self.get_distance(roll_deg, pitch_deg, r_ud[0], r_ud[1])

        r_up = [lower_roll, upper_pitch]
        r_up_dist = self.get_distance(roll_deg, pitch_deg, r_up[0], r_up[1])
        
        l_ud = [upper_roll, lower_pitch]
        l_ud_dist = self.get_distance(roll_deg, pitch_deg, l_ud[0], l_ud[1])
        
        l_up = [lower_roll, lower_pitch]
        l_up_dist = self.get_distance(roll_deg, pitch_deg, l_up[0], l_up[1])

        values_tmp = [1.0/r_ud_dist, 1.0/r_up_dist, 1.0/l_ud_dist, 1.0/r_up_dist]
        tmp_min = min(values_tmp)
        tmp_max = max(values_tmp)

        values = [self.norm(values_tmp[0], 1.0, 0.0, tmp_max, tmp_min), self.norm(values_tmp[1], 1.0, 0.0, tmp_max, tmp_min), self.norm(values_tmp[2], 1.0, 0.0, tmp_max, tmp_min), self.norm(values_tmp[3], 1.0, 0.0, tmp_max, tmp_min)]
        #values = [values_tmp[0]/np.average(values_tmp), values_tmp[1]/np.average(values_tmp), values_tmp[2]/np.average(values_tmp), values_tmp[3]/np.average(values_tmp),]
        #r_ud r_up l_ud l_up
        print(values)
        array = [32761]
        for i in range(len(array)):
            array[i] = 0.0
        
        tmp_index = self.search_index( str(int(r_ud[0])), str(int(r_ud[1])) )
        array[int(tmp_index)] = values[0]

        tmp_index = self.search_index( str(int(r_up[0])), str(int(r_up[1])) )
        array[int(tmp_index)] = values[1]

        tmp_index = self.search_index( str(int(l_ud[0])), str(int(l_ud[1])) )
        array[int(tmp_index)] = values[2]

        tmp_index = self.search_index( str(int(l_up[0])), str(int(l_up[1])) )
        array[int(tmp_index)] = values[3]

        return array

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index][0]
        rp_list = self.data_list[index][5:7]
        rp_list_float = [float(num) for num in rp_list]
        rp_list = self.float_to_array(rp_list_float) #Convert to 32400 array

        img_pil = Image.open(img_path)
        rp_numpy = np.array(rp_list)

        img_trans, rp_trans = self.transform(img_pil, rp_numpy, phase=self.phase)
        return img_trans, rp_trans
import rospy
from sensor_msgs.msg import Imu

import math
import numpy as np
import argparse
import yaml
import os
import time
import csv
import sys
import tf
from geometry_msgs.msg import Quaternion

class QuaternionToRpy:
    def __init__(self):
        self.imu_data = Imu()
        self.imu_subscriber = rospy.Subscriber('/imu/data', Imu, self.callbackImuMsg, queue_size=1)
        self.init_csv('/media/amsl/96fde31e-3b9b-4160-8d8a-a4b913579ca21/tmp_save/imu_attitude', '/imu_attitude.csv')
        self.csv_path = 'init_path'
        self.count = 0
        self.init_sec = 0
    
    def init_csv(self, top_path, filename):
        self.csv_path = os.path.join(top_path, filename)
        self.csv_file = open(self.csv_path, 'w')
        self.csv_w = csv.writer(self.csv_file)

    def callbackImuMsg(self, msg):
        print("Catch IMU Data")
        self.imu_data = msg
        eular = tf.transformations.eular_from_quaternion(self.imu_data.orientation.x, self.imu_data.orientation.y, self.imu_data.orientation.z, self.imu_data.orientation.w)
        time = rospy.to_sec(self.imu_data.header.stamp)
        
        if self.count == 0:
            self.init_sec = time
        
        array = [str(time), str(eular[0]), str(eular[1]), str(eular[2])]
        self.count += 1




if __name__ == "__main__":
    rospy.init_node('quaternion_to_rpy', anonymous=True)
    quartenion_to_rpy = QuaternionToRpy()
    rospy.spin()

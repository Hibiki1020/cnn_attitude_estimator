import yaml
import csv
import argparse
import os

class VerifyInfer:
    def __init__(self, CFG):
        self.CFG = CFG
        self.infer_log_top_path = CFG["infer_log_top_path"]
        self.infer_log_file_name = CFG["infer_log_file_name"]

        self.data_list = self.data_load()

    def data_load(self):
        csv_path = os.path.join(self.infer_log_top_path, self.infer_log_file_name)
        data_list = []
        
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data_list.append(row)

        return data_list

    def spin(self):
        acc_roll_diff = 0.0
        acc_pitch_diff = 0.0
        size_of_csv = float(len(self.data_list))

        for row in self.data_list:
            tmp_roll_diff = abs(float(row[1]) - float(row[3]))
            tmp_pitch_diff = abs(float(row[2]) - float(row[4]))

            acc_roll_diff += tmp_roll_diff
            acc_pitch_diff += tmp_pitch_diff
        
        acc_roll_diff = acc_roll_diff/size_of_csv
        acc_pitch_diff = acc_pitch_diff/size_of_csv

        print("Average Roll Difference  :" + str(acc_roll_diff) + " [deg]")
        print("Average Pitch Difference :" + str(acc_pitch_diff) + " [deg]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./verify_infer.py")

    parser.add_argument(
        '--verify_infer', '-vi',
        type=str,
        required=False,
        default='../pyyaml/verify_infer.yaml',
    )

    FLAGS, unparsed = parser.parse_known_args()
    #Load yaml file
    try:
        print("Opening yaml file %s", FLAGS.verify_infer)
        CFG = yaml.safe_load(open(FLAGS.verify_infer, 'r'))
    except Exception as e:
        print(e)
        print("Error yaml file %s", FLAGS.verify_infer)
        quit()

    verify_infer = VerifyInfer(CFG)
    verify_infer.spin()
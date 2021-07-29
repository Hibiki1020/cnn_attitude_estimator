import yaml
import argparse

import sys
sys.path.append('../')
from common import network_mod
from common import vgg_network_mod
from common import inference_mod

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
    
    attitude_estimator = inference_mod.InferenceMod(CFG)
    attitude_estimator.spin()
    result_csv = attitude_estimator.inference()
    attitude_estimator.save_csv(result_csv)

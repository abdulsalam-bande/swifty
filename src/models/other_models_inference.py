import argparse

from ml_models import OtherModels
from swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="Predict docking scores of your target",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_file", type=str, help="Path to the directory of your file containing the molecules")
parser.add_argument("--output_dir", type=str, help="Where to save the results")
parser.add_argument("--model_name", type=str, help="model_name is the serialized model such as"
                                                 "sgdreg_test_500_mac_50_model.pkl")
args = parser.parse_args()

if __name__ == '__main__':
    OtherModels.inference(args.input_file, args.output_dir, args.model_name)

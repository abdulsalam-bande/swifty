import argparse
from core.lstm import SwiftDock
from utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="Predict docking scores of your target",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--input_file", type=str, help="Path to the directory of your file containing the molecules")
parser.add_argument("--output_dir", type=str, help="Path for the results")
parser.add_argument("--model_name", type=str, help="model_name is the serialized model such as lstm_test_mac_10_model")
args = parser.parse_args()

if __name__ == '__main__':
    SwiftDock.inference(args.input_file, args.output_dir, args.model_name)

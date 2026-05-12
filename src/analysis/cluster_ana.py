import warnings

from core.lstm import SwiftDock
from config.paths import RESULTS_DIR, LSTM_PATHS

warnings.filterwarnings("ignore")

ALL_TARGETS = [
    "Drp1_GTPase",
    "RyR2",
    "Drp1_MiD49",
    "ace",
    "spike",
    "nsp",
    "nsp_sam",
    "5ht1b",
    "fimh",
    "adeR",
    "adeS"
]

MODELS = ["lstm"]
DESCRIPTORS = ["morgan_onehot_mac"]

TRAINING_SIZE = 7000

RESULTS_DIR_CLUSTER = RESULTS_DIR / "cluster_ana_results"
RESULTS_DIR_CLUSTER.mkdir(parents=True, exist_ok=True)

TRAINING_TESTING_DATA_DIR = LSTM_PATHS["training_and_testing_data"]
SERIALIZED_MODELS_DIR = LSTM_PATHS["serialized_models_path"]


if __name__ == "__main__":
    for target in ALL_TARGETS:
        for model in MODELS:
            for descriptor in DESCRIPTORS:
                identifier = f"{model}_{target}_{descriptor}_{TRAINING_SIZE}"

                input_path = TRAINING_TESTING_DATA_DIR / f"{identifier}_test_data.csv"
                model_path = SERIALIZED_MODELS_DIR / f"{identifier}_model.pt"

                if input_path.exists() and model_path.exists():
                    SwiftDock.inference(
                        input_path=input_path,
                        output_path=RESULTS_DIR_CLUSTER,
                        model_path=model_path,
                        cluster_analyses=True
                    )
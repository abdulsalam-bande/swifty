import argparse

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from config.paths import DATASET_DIR, ML_PATHS, create_directories
from config.settings import ML_CONFIG
from core.ml_models import OtherModels
from utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

REGRESSORS = {
    "decision_tree": DecisionTreeRegressor,
    "xgboost": XGBRegressor,
    "sgdreg": SGDRegressor
}

DESCRIPTOR_DIMS = {
    "onehot": 3500 + 1,
    "morgan_onehot_mac": 4691 + 1,
    "mac": 167 + 1
}


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()

    if value in ("yes", "true", "t", "y", "1"):
        return True

    if value in ("no", "false", "f", "n", "0"):
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_selected_descriptors(descriptors):
    selected_descriptors = {}

    for descriptor in descriptors:
        if descriptor not in DESCRIPTOR_DIMS:
            logger.warning(f"Skipping unknown descriptor: {descriptor}")
            continue

        selected_descriptors[descriptor] = DESCRIPTOR_DIMS[descriptor]

    return selected_descriptors


def get_selected_regressors(regressors):
    selected_regressors = {}

    for regressor in regressors:
        if regressor not in REGRESSORS:
            logger.warning(f"Skipping unknown regressor: {regressor}")
            continue

        selected_regressors[regressor] = REGRESSORS[regressor]

    return selected_regressors


def load_memmap_data(data_set_path, data_dim):
    if not data_set_path.exists():
        raise FileNotFoundError(f"Feature file not found: {data_set_path}")

    data = np.memmap(data_set_path, dtype=np.float32)
    target_length = data.shape[0] // data_dim

    return data.reshape((target_length, data_dim))


def train_ml(args, all_data, train_size, test_size, val_size, identifier, number_of_folds,
             regressor, serialized_models_path, descriptor, data_csv, use_cross_validation):

    model = OtherModels(
        training_metrics_dir=ML_PATHS["training_metrics_dir"],
        testing_metrics_dir=ML_PATHS["testing_metrics_dir"],
        test_predictions_dir=ML_PATHS["test_predictions_dir"],
        project_info_dir=ML_PATHS["project_info_dir"],
        shap_analyses_dir=ML_PATHS["shap_analyses_dir"],
        all_data=all_data,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        identifier=identifier,
        number_of_folds=number_of_folds,
        regressor=regressor,
        serialized_models_path=serialized_models_path,
        descriptor=descriptor,
        data_csv=data_csv,
        cross_validate=use_cross_validation,
        config=ML_CONFIG
    )

    model.split_data()
    model.train()

    if use_cross_validation:
        model.diagnose()

    model.test()
    model.shap_analyses()
    model.save_results()


def parse_args():
    parser = argparse.ArgumentParser(description="train code for fast docking",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input", type=str, help="specify the target protein", nargs="+", required=True)
    parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs="+", required=True)
    parser.add_argument("--training_sizes", type=int, help="training and cross validation size", nargs="+", required=True)
    parser.add_argument("--regressors", type=str, help="regressors to train", nargs="+", required=True)
    parser.add_argument("--cross_validate", type=str2bool, help="if to use cross validation")

    return parser.parse_args()


if __name__ == "__main__":
    create_directories(ML_PATHS)
    args = parse_args()

    selected_descriptors = get_selected_descriptors(args.descriptors)
    selected_regressors = get_selected_regressors(args.regressors)

    if not selected_descriptors:
        raise ValueError("No valid descriptors were selected.")

    if not selected_regressors:
        raise ValueError("No valid regressors were selected.")

    number_of_folds = ML_CONFIG["number_of_folds"] if args.cross_validate else 1
    use_cross_validation = bool(args.cross_validate and number_of_folds > 1)

    for target in args.input:
        for regressor_id, regressor in selected_regressors.items():
            for descriptor, data_dim in selected_descriptors.items():
                for size in args.training_sizes:
                    data_set_path = DATASET_DIR / f"{target}_{descriptor}.dat"
                    data_csv = DATASET_DIR / f"{target}.csv"

                    if not data_csv.exists():
                        raise FileNotFoundError(f"Dataset not found: {data_csv}")

                    identifier = f"{regressor_id}_{target}_{descriptor}_{size}"
                    data = load_memmap_data(data_set_path, data_dim)

                    train_size = size
                    val_size = size * number_of_folds if use_cross_validation else 0
                    test_size = len(data) - (train_size + val_size)

                    if test_size <= 0:
                        raise ValueError(f"Invalid split sizes for {identifier}. train_size={train_size}, "
                                         f"val_size={val_size}, test_size={test_size}")

                    train_ml(args=args, all_data=data, train_size=train_size, test_size=test_size,
                             val_size=val_size, identifier=identifier, number_of_folds=number_of_folds,
                             regressor=regressor, serialized_models_path=ML_PATHS["serialized_models_path"],
                             descriptor=descriptor, data_csv=data_csv, use_cross_validation=use_cross_validation)
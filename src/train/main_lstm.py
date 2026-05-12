import argparse

import pandas as pd

from config.paths import DATASET_DIR, LSTM_PATHS, create_directories
from config.settings import DEFAULT_CONFIG
from core.lstm import logger, SwiftDock
from features.smiles_featurizers import mac_keys_fingerprints, one_hot_encode, morgan_fingerprints_mac_and_one_hot
from utils.utils import get_target_seq


DESCRIPTORS = {
    "mac": {
        "feature_dim": 167,
        "function": mac_keys_fingerprints
    },
    "onehot": {
        "feature_dim": 3500,
        "function": one_hot_encode
    },
    "morgan_onehot_mac": {
        "feature_dim": 4691,
        "function": morgan_fingerprints_mac_and_one_hot
    }
}


def get_descriptor_data(descriptor):
    return DESCRIPTORS.get(descriptor)


def get_descriptor_name(descriptor_function):
    for name, descriptor_data in DESCRIPTORS.items():
        if descriptor_data["function"].__name__ == descriptor_function.__name__:
            return name

    return None


def get_cross_validation_folds(args):
    if not args.cross_validate:
        return 1

    return int(args.cross_validate)


def train_models(args, target, descriptor_data, size):
    number_of_folds = get_cross_validation_folds(args)
    use_cross_validation = number_of_folds > 1

    descriptor_name = get_descriptor_name(descriptor_data["function"])
    identifier = f"lstm_{target}_{descriptor_name}_{size}"

    logger.info(f"Identifier {identifier}")

    data_csv = DATASET_DIR / f"{target}.csv"
    identifier_data = LSTM_PATHS["tsne_analyses_dir"] / f"{identifier}_data.csv"

    if not data_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")

    data_all = pd.read_csv(data_csv).dropna().reset_index(drop=True)
    data_all.to_csv(identifier_data, index=False)

    train_size = size
    val_size = size * number_of_folds if use_cross_validation else 0
    test_size = len(data_all) - (train_size + val_size)

    if test_size <= 0:
        raise ValueError(f"Invalid split sizes. train_size={train_size}, val_size={val_size}, test_size={test_size}")

    sequence = get_target_seq(target)

    model = SwiftDock(
        training_and_testing_data=LSTM_PATHS["training_and_testing_data"],
        training_metrics_dir=LSTM_PATHS["training_metrics_dir"],
        testing_metrics_dir=LSTM_PATHS["testing_metrics_dir"],
        test_predictions_dir=LSTM_PATHS["test_predictions_dir"],
        project_info_dir=LSTM_PATHS["project_info_dir"],
        target_path=data_all,
        train_size=train_size,
        test_size=test_size,
        val_size=val_size,
        identifier=identifier,
        number_of_folds=number_of_folds,
        descriptor=descriptor_data["function"],
        feature_dim=descriptor_data["feature_dim"],
        serialized_models_path=LSTM_PATHS["serialized_models_path"],
        cross_validate=use_cross_validation,
        shap_analyses_dir=LSTM_PATHS["shap_analyses_dir"],
        tsne_analyses_dir=LSTM_PATHS["tsne_analyses_dir"],
        data_csv=data_csv,
        batch_size=args.batch_size or DEFAULT_CONFIG["batch_size"],
        number_of_workers=args.number_of_workers or DEFAULT_CONFIG["number_of_workers"],
        sequence=sequence,
        split_base_on_clustering=args.split_base_on_clustering,
        config=DEFAULT_CONFIG
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
    parser.add_argument("--cross_validate", type=int, help="number of folds to use for cross validation")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--number_of_workers", type=int, help="number of workers")
    parser.add_argument("--split_base_on_clustering", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    create_directories(LSTM_PATHS)
    args = parse_args()

    for target in args.input:
        for descriptor in args.descriptors:
            descriptor_data = get_descriptor_data(descriptor)

            if descriptor_data is None:
                logger.warning(f"Skipping unknown descriptor: {descriptor}")
                continue

            for size in args.training_sizes:
                train_models(args, target, descriptor_data, size)
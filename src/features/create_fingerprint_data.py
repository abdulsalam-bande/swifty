import argparse
import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config.paths import DATASET_DIR
from config.settings import FEATURE_CONFIG
from features.smiles_featurizers import one_hot_encode, morgan_fingerprints_mac_and_one_hot, mac_keys_fingerprints
from utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

DESCRIPTORS = {
    "onehot": {
        "feature_dim": 3500,
        "function": one_hot_encode
    },
    "morgan_onehot_mac": {
        "feature_dim": 4691,
        "function": morgan_fingerprints_mac_and_one_hot
    },
    "mac": {
        "feature_dim": 167,
        "function": mac_keys_fingerprints
    }
}


class FeatureGenerator(Dataset):
    def __init__(self, data, descriptor):
        self.data = data.reset_index(drop=True)
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]

        smile = str(data["smile"])
        score = np.array(data["docking_score"], dtype=np.float32).reshape(1,)

        features = self.descriptor(smile)
        features = features.astype(np.float32)

        return np.concatenate((features, score))


def get_selected_descriptors(descriptors):
    selected_descriptors = {}

    for descriptor in descriptors:
        if descriptor not in DESCRIPTORS:
            logger.warning(f"Skipping unknown descriptor: {descriptor}")
            continue

        selected_descriptors[descriptor] = DESCRIPTORS[descriptor]

    return selected_descriptors


def load_target_data(target):
    csv_path = DATASET_DIR / f"{target}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    if "smile" not in data.columns or "docking_score" not in data.columns:
        raise ValueError(f"{csv_path} must contain 'smile' and 'docking_score' columns.")

    return data


def create_feature_file(target, descriptor_name, descriptor_data, batch_size, num_workers):
    data = load_target_data(target)

    descriptor = descriptor_data["function"]
    feature_dim = descriptor_data["feature_dim"]

    output_path = DATASET_DIR / f"{target}_{descriptor_name}.dat"

    dataset = FeatureGenerator(data, descriptor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_set = np.memmap(output_path, dtype=np.float32, mode="w+", shape=(len(data), feature_dim + 1))

    start_time = time.time()
    start_index = 0

    for batch in tqdm(dataloader):
        batch = batch.numpy().astype(np.float32)
        end_index = start_index + batch.shape[0]

        data_set[start_index:end_index, :] = batch
        start_index = end_index

    data_set.flush()
    del data_set

    create_time = (time.time() - start_time) / 60

    logger.info(f"Created {output_path}")
    logger.info(f"Creating Time: {create_time:.4f} Minutes")


def create_features(targets, descriptors, batch_size=None, num_workers=None):
    batch_size = batch_size or FEATURE_CONFIG["batch_size"]
    num_workers = FEATURE_CONFIG["num_workers"] if num_workers is None else num_workers

    if not descriptors:
        raise ValueError("No valid descriptors were selected.")

    for target in targets:
        for descriptor_name, descriptor_data in descriptors.items():
            create_feature_file(target, descriptor_name, descriptor_data, batch_size, num_workers)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Featurize molecules for sklearn models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--input", type=str, help="targets to create binary files for", nargs="+", required=True)
    parser.add_argument("--descriptors", type=str, help="descriptor names", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--num_workers", type=int, help="number of dataloader workers")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selected_descriptors = get_selected_descriptors(args.descriptors)

    create_features(
        targets=args.input,
        descriptors=selected_descriptors,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
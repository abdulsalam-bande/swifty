import argparse
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from smiles_featurizers import morgan_fingerprints_mac_and_one_hot, mac_keys_fingerprints, one_hot_encode

from swift_dock_logger import swift_dock_logger

info = {
    'onehot': [3500, one_hot_encode],
    'morgan_onehot_mac': [4691, morgan_fingerprints_mac_and_one_hot],
    'mac': [167, mac_keys_fingerprints]
}

dataset_dir = "../../datasets"

logger = swift_dock_logger()


class FeatureGenerator(Dataset):
    def __init__(self, data_dict, descriptor):
        self.data_dict = data_dict
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        score = data['docking_score']
        score = np.array(score, dtype='float32').reshape(1, )
        features = self.descriptor(smile)
        X_Y = np.concatenate((features, score))
        return X_Y


def create_features(targets, info):
    for target in targets:
        for key, item in info.items():
            fingerprint_name = key
            descriptor = item[1]
            path_to_csv_file = f"{dataset_dir}/{target}.csv"
            data = pd.read_csv(path_to_csv_file)
            data = data.dropna()
            smiles_data_train = FeatureGenerator(data, descriptor)
            directory = f"{dataset_dir}/{target}_{fingerprint_name}.dat"
            data_set = np.memmap(directory, dtype=np.float32,
                                 mode='w+', shape=(len(data), item[0] + 1))
            start_time_test = time.time()
            init = 0
            batch_size = 128
            train_dataloader = DataLoader(smiles_data_train, batch_size=batch_size, shuffle=True, num_workers=128)
            for i, data in enumerate(tqdm(train_dataloader)):
                numpy_data = data.numpy()
                data_set[init:init + numpy_data.shape[0], :] = numpy_data
                init = init + numpy_data.shape[0]
            create_time = (time.time() - start_time_test) / 60
            logger.info(f"Creating Time : {create_time} Minutes")
            del data_set
            time_dict = {'creation_time': create_time}
            # saving_dir = f"{dataset_dir}/{target}_{fingerprint_name}_creation_time.txt"
            # with open(saving_dir, 'w+') as file:
            #     file.write(str(time_dict))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Featurize molecules for the targets that will be trained with sklearn "
                    "models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="to create the binary files for", nargs='+')
    parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
    args = parser.parse_args()
    descriptors_dictionary_command_line = {}
    for desc in args.descriptors:
        if desc in info:
            descriptors_dictionary_command_line[desc] = [info[desc][0],
                                                         info[desc][1]]
    create_features(args.input, descriptors_dictionary_command_line)

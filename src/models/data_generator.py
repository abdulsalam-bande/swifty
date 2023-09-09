import torch
from torch.utils.data import Dataset
from smiles_featurizers import mac_keys_fingerprints, one_hot_encode, morgan_fingerprints_mac_and_one_hot


class DataGenerator(Dataset):
    def __init__(self, data_dict, descriptor):
        self.data_dict = data_dict
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        score = data['docking_score']
        features = self.descriptor(smile)
        features = torch.from_numpy(features.reshape(features.shape[0], 1))
        score = torch.tensor([score])
        return features, score


class ShapAnalysesDataGenerator(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        return torch.from_numpy(x_sample).to(dtype=torch.float32).reshape(-1, 1), torch.tensor([y_sample]).to(
            dtype=torch.float64)


class InferenceDataGenerator(Dataset):
    def __init__(self, data_dict, descriptor):
        self.data_dict = data_dict
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        features = self.descriptor(smile)
        features = torch.from_numpy(features.reshape(features.shape[0], 1))
        return features

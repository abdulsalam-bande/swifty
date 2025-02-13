import copy
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ExplicitBitVect
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from smiles_featurizers import one_hot_encode, mac_keys_fingerprints
from seq_dict import sequence_dict

mpl.rcParams['figure.dpi'] = 300

warnings.filterwarnings("ignore")


def test_model(test_dataloader, net):
    smiles_prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            features, _ = data
            if features.shape[0] == 1:
                features = features.squeeze(-1)
            else:
                features = features.squeeze()
            features = features.unsqueeze(0)
            outputs = net(features)
            if outputs.squeeze().dim() == 0:
                smiles_prediction.append(outputs.squeeze().item())
            else:
                smiles_prediction.extend(outputs.squeeze().tolist())
    return smiles_prediction


def inference(test_dataloader, net):
    smiles_prediction = []
    with torch.no_grad():
        for i, features in enumerate(test_dataloader):
            features = features.squeeze()
            features = features.unsqueeze(0)
            outputs = net(features)
            smiles_prediction.extend(outputs.squeeze().tolist())
    return smiles_prediction


def create_fold_predictions_and_target_df(fold_predictions, smiles_target, number_of_folds, test_size):
    all_predictions = np.zeros((test_size, number_of_folds + 1))
    for i in range(number_of_folds):
        all_predictions[:, i] = fold_predictions[i]

    all_predictions[:, -1] = smiles_target
    columns = ['f' + str(i) for i in range(number_of_folds)]
    columns.append('target')
    predictions_and_target_df = pd.DataFrame(all_predictions, columns=columns)
    return predictions_and_target_df


def create_test_metrics(fold_predictions, smiles_target, number_of_folds):
    all_folds_mse = 0
    all_folds_mae = 0
    all_folds_rsquared = 0
    for i in range(number_of_folds):
        mse, mae, rsquared = calculate_metrics(
            fold_predictions[i], smiles_target)
        all_folds_mse = all_folds_mse + mse
        all_folds_mae = all_folds_mae + mae
        all_folds_rsquared = all_folds_rsquared + rsquared
    metrics_dict_test = {"test_mse": [], "test_mae": [], "test_rsquared": []}
    metrics_dict_test["test_mse"].append((all_folds_mse / number_of_folds))
    metrics_dict_test["test_mae"].append((all_folds_mae / number_of_folds))
    metrics_dict_test["test_rsquared"].append((all_folds_rsquared / number_of_folds) * 100)
    return metrics_dict_test


def calculate_metrics(predictions, target):
    mse = mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    rsquared = r2_score(target, predictions)
    return mse, mae, rsquared


def get_training_and_test_data(data, training_size, testing_size):
    x = data['smile']
    y = data['docking_score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=training_size, test_size=testing_size,
                                                        random_state=42)
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    return train, test


def get_data_splits(data, training_count, testing_count, validation_count):
    x = data['smile']
    y = data['docking_score']
    total_data = data.shape[0]
    assert training_count + testing_count + validation_count == total_data, "The counts must sum up to the total number of data points"

    # Splitting the data into test and temp (temp contains both training and validation data)
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=testing_count, random_state=42)

    # Remaining data for training and validation
    remaining_data = total_data - testing_count

    # Calculate proportions for training and validation sets
    train_proportion = training_count / remaining_data
    val_proportion = validation_count / remaining_data

    # Split the temp data into train and validation
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, train_size=train_proportion, random_state=42)

    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    validation = pd.concat([x_val, y_val], axis=1)

    return train, test, validation


def get_data_splits_clustering(data, training_count, testing_count, validation_count):
    sorted_data = data.sort_values('cluster')
    total_data = sorted_data.shape[0]
    assert training_count + testing_count + validation_count == total_data, "The counts must sum up to the total number of data points"

    # Indices for splitting data into train, test, and validation
    train_end_idx = training_count
    test_end_idx = train_end_idx + validation_count

    # Splitting the data
    train = sorted_data.iloc[:train_end_idx].reset_index(drop=True)
    validation = sorted_data.iloc[train_end_idx:test_end_idx].reset_index(drop=True)
    test = sorted_data.iloc[test_end_idx:].reset_index(drop=True)

    return train, test, validation


def save_dict(history, identifier):
    result_df = pd.DataFrame.from_dict(history)
    result_df.index.name = 'index'
    result_df.to_csv(identifier, index=False)


def get_smiles_dict(path_to_all_smiles):
    all_strings = ""
    f = open(path_to_all_smiles, "r")
    for x in f:
        all_strings = all_strings + x

    chars = tuple(set(all_strings))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    dict_size = len(char2int)
    return int2char, char2int, dict_size


class TanimotoDataGenerator(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        smile = data[0]
        fingerprint = data[1]
        data_copy = copy.deepcopy(self.data)
        del data_copy[idx]
        all_distance = []
        for key, value in data_copy.items():
            dis = calculate_tanimoto_distance(fingerprint, value[1])
            all_distance.append(dis)

        result = {'avg': sum(all_distance) / len(all_distance), 'max': max(all_distance), 'min': min(all_distance)}
        return result


def plot_docking_scores_hist(data, directory):
    plt.hist(data, bins=30)  # density=False would make counts
    plt.ylabel('Number Of Smiles')
    plt.xlabel('Docking Score')
    plt.savefig(directory)
    plt.show()


def plot_tanimoto_hist(data, directory, Info_type):
    plt.hist(data, bins=30)  # density=False would make counts
    plt.ylabel('Frequency')
    plt.xlabel(Info_type)
    plt.savefig(directory)
    plt.show()


def save_dict_with_one_index(history, identifier):
    result_df = pd.DataFrame(history, index=[0])
    result_df.to_csv(identifier, index=False)


def calculate_tanimoto_distance(smile1, smile2):
    return DataStructs.FingerprintSimilarity(smile1, smile2)


def morgan_fingerprints_mac_and_one_hot_bitvect(smile):
    # Get the Morgan fingerprint
    m1 = Chem.MolFromSmiles(smile)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(m1, 1, nBits=1024).ToBitString()

    # Get the one-hot encoding
    one_hot = one_hot_encode(smile).astype(int).tolist()

    # Get the MACCS keys
    mac_keys = mac_keys_fingerprints(smile).astype(int).tolist()

    # Combine them in the order: fingerprint, one-hot, MACCS
    combined = fingerprint + ''.join(map(str, one_hot)) + ''.join(map(str, mac_keys))

    # Convert to ExplicitBitVect for RDKit compatibility
    combined_bitvect = ExplicitBitVect(len(combined))
    for i, bit in enumerate(combined):
        combined_bitvect.SetBit(i) if int(bit) else combined_bitvect.UnSetBit(i)

    return combined_bitvect


def calculate_tanimoto_similarity(smile1, smile2):
    fp1 = morgan_fingerprints_mac_and_one_hot_bitvect(smile1)
    fp2 = morgan_fingerprints_mac_and_one_hot_bitvect(smile2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_most_similar_structure(test_data, training_data):
    max_similarity = -float('inf')
    most_similar_structure = None
    for index, train_fp in training_data.iterrows():
        train_smile = train_fp['smile']
        test_smile = test_data['smile']
        similarity = calculate_tanimoto_similarity(test_smile, train_smile)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_structure = train_fp
    return most_similar_structure


def correlate_predictions(test_data, training_data, model):
    predictions = []
    actual_values = []
    for test_fp in test_data:
        most_similar_structure = get_most_similar_structure(test_fp, training_data)
        prediction = model.predict([test_fp])[0]
        actual_value = most_similar_structure['target']
        predictions.append(prediction)
        actual_values.append(actual_value)
    correlation_coefficient = np.corrcoef(predictions, actual_values)[0, 1]
    return correlation_coefficient


def get_fingerprint(smiles_string):
    """Generate Morgan fingerprint for a given SMILES string."""
    molecule = Chem.MolFromSmiles(smiles_string)
    return AllChem.GetMorganFingerprintAsBitVect(molecule, 2)


def prepare_llm_instruction(smile, smiles_properties, predicted_docking_score):
    instruction = f"""
    Given the SMILE representation "{smile}" and its predicted docking score of {predicted_docking_score}, 
    alongside the following molecular properties: 
    Molecular Weight: {smiles_properties['mol_weight']}, 
    Number of Atoms: {smiles_properties['num_atoms']}, 
    Number of Bonds: {smiles_properties['num_bonds']}, 
    Number of Rotatable Bonds: {smiles_properties['num_rotatable_bonds']}, 
    Number of Hydrogen Donors: {smiles_properties['num_h_donors']}, 
    Number of Hydrogen Acceptors: {smiles_properties['num_h_acceptors']}, 
    LogP: {smiles_properties['logp']}, 
    Molar Refractivity: {smiles_properties['mr']}, 
    Topological Polar Surface Area (TPSA): {smiles_properties['tpsa']}, 
    Number of Rings: {smiles_properties['num_rings']}, 
    Number of Aromatic Rings: {smiles_properties['num_aromatic_rings']}, 
    Hall-Kier Alpha: {smiles_properties['hall_kier_alpha']}, 
    Fraction Csp3: {smiles_properties['fraction_csp3']}, 
    Number of Nitrogens: {smiles_properties['num_nitrogens']}, 
    Number of Oxygens: {smiles_properties['num_oxygens']}, 
    Number of Sulphurs: {smiles_properties['num_sulphurs']},

    provide an explanation of how these specific properties may influence the docking score. 
    Consider the significance of the SMILE representation in terms of structural features that could 
    affect the molecule's interaction with a biological target, including aspects such as hydrophobic 
    interactions indicated by LogP, molecular flexibility suggested by the number of rotatable bonds, 
    and other relevant molecular features.
    """

    return instruction


def get_target_seq(target_name):
    default = "MEALIPVINKLQDVFNTVGADIIQLPQIVVVGTQSSGKSSVLESLVGRDLLPRGTGIVTRRPLILQLVHVSQEDKRKTTGEENGVEAEEWGKFLHTKNKLYTDFDEIRQEIENETERISGNNKGVSPEPIHLKIFSPNVVNLTLVDLPGMTKVPVGDQPKDIELQIRELILRFISNPNSIILAVTAANTDMATSEALKISREVDPDGRRTLAVITKLDLMDAGTDAMDVLMGRVIPVKLGIIGVVNRSQLDINNKKSVTDSIRDEYAFLQKKYPSLANRNGTKYLARTLNRLLMHHIRDCLPELKTRINVLAAQYQSLLNSYGEPVDGSGSGSGSKEAADMLKALQGASQIIAEIRETHLWLEHHHHHH"
    return sequence_dict.get(target_name,default)

import copy
import os
import pickle
import time

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from create_fingerprint_data import create_features
from smiles_featurizers import morgan_fingerprints_mac_and_one_hot, mac_keys_fingerprints, one_hot_encode, \
    compute_descriptors
from swift_dock_logger import swift_dock_logger
from utils import calculate_metrics, create_test_metrics, create_fold_predictions_and_target_df, save_dict

logger = swift_dock_logger()


class OtherModels:
    def __init__(self, training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 shap_analyses_dir,
                 all_data, train_size, test_size, val_size, identifier, number_of_folds, regressor,
                 serialized_models_path, descriptor, data_csv):
        self.all_data = all_data
        self.training_metrics_dir = training_metrics_dir
        self.testing_metrics_dir = testing_metrics_dir
        self.test_predictions_dir = test_predictions_dir
        self.project_info_dir = project_info_dir
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.regressor = regressor
        self.serialized_models_path = serialized_models_path
        self.descriptor = descriptor
        self.shap_analyses_dir = shap_analyses_dir
        self.data_csv = data_csv
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.cross_validation_metrics = None
        self.all_regressors = None
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.train_time = None
        self.single_regressor = None
        self.test_for_shap_analyses = None
        self.model_for_shap_analyses = copy.deepcopy(regressor())
        self.scaler = StandardScaler()
        self.train_for_shap_analyses = None

    def split_data(self, cross_validate):
        if cross_validate:
            self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]

            self.x_train = self.x[:self.train_size]
            self.y_train = self.y[:self.train_size]

            self.x_val = self.x[self.train_size:self.train_size + self.val_size]
            self.y_val = self.y[self.train_size:self.train_size + self.val_size]

            self.x_test = self.x[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]
            self.y_test = self.y[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]
        else:
            self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]
            self.x_train = self.x[0:self.train_size]
            self.y_train = self.y[0:self.train_size]
            self.x_test = self.x[self.train_size:self.train_size + self.test_size]
            self.y_test = self.y[self.train_size:self.train_size + self.test_size]

    def train(self):
        logger.info(f"Training has started for {self.identifier}")
        start_time_train = time.time()
        rg = self.regressor()  # Create an instance of the regressor
        rg.fit(self.x_train, self.y_train)
        self.train_time = (time.time() - start_time_train) / 60
        self.single_regressor = rg
        identifier_model_path = f"{self.serialized_models_path}{self.identifier}_model.pkl"
        descriptor_dict = {'descriptor': self.descriptor}
        with open(identifier_model_path, 'wb') as file:
            pickle.dump((rg, descriptor_dict), file)
        logger.info(f"Training is Done! {self.identifier}")

        sample_size = 50000
        if len(pd.read_csv(self.data_csv)) > 4500000:
            sample_size = 100000

        shap_test_size = sample_size * 0.8
        data_df = pd.read_csv(self.data_csv).sample(sample_size)
        self.train_for_shap_analyses, self.test_for_shap_analyses = train_test_split(data_df, test_size=shap_test_size,
                                                                                     random_state=42)
        train_smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values()) for smile in
                        self.train_for_shap_analyses['smile']]
        train_docking_scores = self.train_for_shap_analyses['docking_score'].tolist()
        normalized_descriptors = self.scaler.fit_transform(train_smiles)
        self.model_for_shap_analyses.fit(normalized_descriptors, train_docking_scores)

    def diagnose(self):
        logger.info(f"Validation has started for {self.identifier}")
        kf = KFold(n_splits=self.number_of_folds)
        kf.get_n_splits(self.x_val)
        regressors_list = []
        train_metrics = {'average_fold_mse': [], 'average_fold_mae': [], 'average_fold_rsquared': []}
        start_time_train = time.time()
        for big_index, small_index in kf.split(self.x_val):
            x_train_fold, x_test_fold = self.x_val[small_index], self.x_val[big_index]
            y_train_fold, y_test_fold = self.y_val[small_index], self.y_val[big_index]
            rg = self.regressor()  # Create an instance of the regressor
            rg.fit(x_train_fold, y_train_fold)
            regressors_list.append(rg)
            predictions = rg.predict(x_test_fold)
            mse, mae, rsquared = calculate_metrics(predictions, y_test_fold)
            train_metrics['average_fold_mse'].append(mse)
            train_metrics['average_fold_mae'].append(mae)
            train_metrics['average_fold_rsquared'].append(rsquared)
        self.cross_validation_time = (time.time() - start_time_train) / 60
        average_fold_mse = sum(train_metrics['average_fold_mse']) / len(train_metrics['average_fold_mse'])
        average_fold_mae = sum(train_metrics['average_fold_mae']) / len(train_metrics['average_fold_mae'])
        average_fold_r2 = sum(train_metrics['average_fold_rsquared']) / len(train_metrics['average_fold_rsquared'])
        train_metrics = {'average_fold_mse': [average_fold_mse], 'average_fold_mae': [average_fold_mae],
                         'average_fold_rsquared': [average_fold_r2]}
        self.cross_validation_metrics = train_metrics
        self.all_regressors = regressors_list
        identifier_train_val_metrics = f"{self.training_metrics_dir}{self.identifier}_cross_validation_metrics.csv"
        save_dict(self.cross_validation_metrics, identifier_train_val_metrics)

    def test(self):
        logger.info(f"Testing has started for {self.identifier}")
        all_models_predictions = []
        start_time_test = time.time()
        fold_predictions = self.single_regressor.predict(self.x_test)
        all_models_predictions.append(fold_predictions)
        self.test_time = (time.time() - start_time_test) / 60
        metrics_dict_test = create_test_metrics(all_models_predictions, self.y_test, 1)
        predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, self.y_test,
                                                                          1, self.test_size)
        self.test_metrics = metrics_dict_test
        self.test_predictions_and_target_df = predictions_and_target_df
        logger.info(f"Testing is Done! {self.identifier}")
        return all_models_predictions

    def shap_analyses(self):
        logger.info('Starting Shap Analyses.')
        smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values()) for smile in
                  self.test_for_shap_analyses['smile']]
        normalized_descriptors = self.scaler.fit_transform(smiles)

        def model_predict(smiles):
            return self.model_for_shap_analyses.predict(smiles)

        # Create a masker for the dataset
        masker = shap.maskers.Independent(data=normalized_descriptors)
        explainer = shap.explainers.Permutation(model_predict, masker)
        shap_values = explainer.shap_values(normalized_descriptors)

        # File paths for output
        shap_analyses_csv_dir = f"{self.shap_analyses_dir}{self.identifier}_shap_analyses.csv"
        shap_analyses_summary_plot = f"{self.shap_analyses_dir}{self.identifier}_shap_summary_plot.png"
        shap_analyses_feature_importance = f"{self.shap_analyses_dir}{self.identifier}_shap_feature_importance.png"
        feature_names = [
            "mol_weight",
            "num_atoms",
            "num_bonds",
            "num_rotatable_bonds",
            "num_h_donors",
            "num_h_acceptors",
            "logp",
            "mr",
            "tpsa",
            "num_rings",
            "num_aromatic_rings",
            "hall_kier_alpha",
            "fraction_csp3",
            "num_nitrogens",
            "num_oxygens",
            "num_sulphurs"
        ]

        # Convert normalized_descriptors to DataFrame with feature names
        normalized_descriptors_df = pd.DataFrame(normalized_descriptors, columns=feature_names)

        # SHAP DataFrame
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        avg_shap = shap_df.abs().mean().sort_values(ascending=False)
        avg_shap.to_csv(shap_analyses_csv_dir)

        # Generate and save SHAP plots
        shap.summary_plot(shap_values, normalized_descriptors_df, plot_type="dot", show=False)
        plt.gcf().tight_layout()
        plt.gcf().savefig(shap_analyses_summary_plot)
        plt.close()

        shap.summary_plot(shap_values, normalized_descriptors_df, plot_type="bar", show=False)
        plt.gcf().tight_layout()
        plt.gcf().savefig(shap_analyses_feature_importance)
        plt.close()

    def evaluate_structural_diversity(self):
        logger.info('Starting Structural Diversity Analyses...')
        tsne_visualization_dir = f"{self.shap_analyses_dir}{self.identifier}_tsne_visualization.png"
        tsne_dir = f"{self.shap_analyses_dir}{self.identifier}_tsne_data.csv"

        def __get_circular_fingerprints(data):
            fps = []
            for smile in data['smile']:
                mol = Chem.MolFromSmiles(smile)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                arr = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, arr)
                fps.append(arr)
            return np.array(fps)

        def __get_mac_fingerprints(data):
            fps = []
            for smile in data['smile']:
                mol = Chem.MolFromSmiles(smile)
                if mol is None:  # Invalid SMILES string
                    print(f"Warning: Invalid SMILES skipped: {smile}")
                    continue
                maccs_key = MACCSkeys.GenMACCSKeys(mol)
                arr = np.zeros((1,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(maccs_key, arr)
                fps.append(arr)
            return np.array(fps)

        # Assuming train_for_shap_analyses and test_for_shap_analyses are your training and test DataFrames
        train_fps = __get_mac_fingerprints(self.train_for_shap_analyses)
        test_fps = __get_mac_fingerprints(self.test_for_shap_analyses)

        # 2. t-SNE Dimensionality Reduction
        all_fps = np.vstack([train_fps, test_fps])
        tsne = TSNE(n_components=2, random_state=42).fit_transform(all_fps)

        # Save t-SNE data to the directory specified in 'tsne_dir' as a CSV file
        pd.DataFrame(tsne, columns=['Dimension_1', 'Dimension_2']).to_csv(tsne_dir, index=False)

        # 3. Visualization
        plt.figure(figsize=(10, 7))
        plt.scatter(tsne[:len(train_fps), 0], tsne[:len(train_fps), 1], color='blue', label='Training Data', alpha=0.5)
        plt.scatter(tsne[len(train_fps):, 0], tsne[len(train_fps):, 1], color='red', label='Test Data', alpha=0.5)
        plt.legend(loc='upper right')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE Visualization of Training vs Test Data')

        # Save dat
        plt.savefig(tsne_visualization_dir, dpi=300, bbox_inches='tight')
        pd.DataFrame(tsne, columns=['Dimension_1', 'Dimension_2']).to_csv(tsne_dir, index=False)

    def plot_docking_vs_mol_weight(self):
        logger.info(f"Started creating plot for mol weights of {self.identifier}")
        size_cor_plot_dir = f"{self.shap_analyses_dir}{self.identifier}_mol_weight.png"
        df = pd.read_csv(self.data_csv)
        mol_weights = []
        for smile in df['smile']:
            mol = Chem.MolFromSmiles(smile)
            mol_weight = Descriptors.MolWt(mol)
            mol_weights.append(mol_weight)

            # Add the molecular weights as a new column to the DataFrame
        df['mol_weight'] = mol_weights

        # Create a scatter plot
        plt.figure(figsize=(10, 6))

        # Uncomment the following lines for a 2D density plot
        sns.kdeplot(x=df['mol_weight'], y=df['docking_score'], cmap='inferno', fill=True)

        plt.title('Docking Score vs Molecular Weight')
        plt.xlabel('Molecular Weight')
        plt.ylabel('Docking Score')
        plt.grid(True)
        plt.savefig(size_cor_plot_dir)

    def save_results(self):
        identifier_test_metrics = f"{self.testing_metrics_dir}{self.identifier}_test_metrics.csv"
        save_dict(self.test_metrics, identifier_test_metrics)
        identifier_test_pred_target_df = f"{self.test_predictions_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
        project_info_dict = {"training_size": [self.train_size], "testing_size": [self.test_size],
                             str(self.number_of_folds) + " fold_validation_time": [self.cross_validation_time],
                             'training_time': self.train_time,
                             "testing_time": [self.test_time]}
        identifier_project_info = f"{self.project_info_dir}{self.identifier}_project_info.csv"
        save_dict(project_info_dict, identifier_project_info)
        logger.info(f"Saving done started for {self.identifier}")

    @staticmethod
    def inference(input_path, output_path, model_path):
        smiles = pd.read_csv(input_path)['smile'].tolist()
        tmp_path = '../../datasets/tmp.csv'
        logger.info('Inference has started...')
        df = pd.read_csv(input_path)
        df['docking_score'] = 0
        df.to_csv(tmp_path, index=False)
        # Load the model
        with open(model_path, 'rb') as file:
            pickle_model, descriptor_dict = pickle.load(file)
        descriptor = descriptor_dict['descriptor']
        info = {
            'onehot': [3500, one_hot_encode],
            'morgan_onehot_mac': [4691, morgan_fingerprints_mac_and_one_hot],
            'mac': [167, mac_keys_fingerprints]
        }
        dimensions_ml_models = {'onehot': 3500 + 1, 'morgan_onehot_mac': 4691 + 1,
                                'mac': 167 + 1}
        new_dict = {descriptor: info[descriptor]}
        create_features(['tmp'], new_dict)
        os.remove(tmp_path)
        data_set_path = f'../../datasets/tmp_{descriptor}.dat'
        data = np.memmap(data_set_path, dtype=np.float32)
        target_length = data.shape[0] // dimensions_ml_models[descriptor]
        data = data.reshape((target_length, dimensions_ml_models[descriptor]))
        x, y = data[:, :-1], data[:, -1]
        predictions = pickle_model.predict(x)
        results_dict = {"smile": smiles, "docking_score": predictions}
        identifier_project_info = f"{output_path}/results.csv"
        save_dict(results_dict, identifier_project_info)
        logger.info('Inference is finished')

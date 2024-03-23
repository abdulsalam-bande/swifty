import copy
import os
import time
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from data_generator import DataGenerator, InferenceDataGenerator, ShapAnalysesDataGenerator
from model import AttentionNetwork
from smiles_featurizers import compute_descriptors
from swift_dock_logger import swift_dock_logger
from trainer import train_model
from utils import get_training_and_test_data, test_model, calculate_metrics, create_test_metrics, \
    create_fold_predictions_and_target_df, save_dict, inference, get_data_splits_clustering

logger = swift_dock_logger()
warnings.filterwarnings("ignore")

class SwiftDock:
    def __init__(self, training_and_testing_data, training_metrics_dir, testing_metrics_dir, test_predictions_dir,
                 project_info_dir, target_path,
                 train_size, test_size, val_size, identifier, number_of_folds, descriptor, feature_dim,
                 serialized_models_path, cross_validate, shap_analyses_dir, tsne_analyses_dir, data_csv, batch_size,
                 number_of_workers, sequence):
        self.target_path = target_path
        self.training_metrics_dir = training_metrics_dir
        self.training_and_testing_data = training_and_testing_data
        self.testing_metrics_dir = testing_metrics_dir
        self.test_predictions_dir = test_predictions_dir
        self.project_info_dir = project_info_dir
        self.train_size = train_size
        self.test_size = test_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.feature_dim = feature_dim
        self.descriptor = descriptor
        self.serialized_models_path = serialized_models_path
        self.train_data = None
        self.test_data = None
        self.val_data = val_size
        self.cross_validation_metrics = None
        self.all_networks = None
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.single_mode_time = None
        self.single_model = None
        self.cross_validate = cross_validate
        self.shap_analyses_dir = shap_analyses_dir
        self.model_for_shap_analyses = None
        self.data_csv = data_csv
        self.scaler = StandardScaler()
        self.tsne_metrics_dir = tsne_analyses_dir
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        self.sequence = sequence

    def split_data(self, cross_validate):
        train_name = f"{self.training_and_testing_data}{self.identifier}_train_data.csv"
        val_name = f"{self.training_and_testing_data}{self.identifier}_val_data.csv"
        test_name = f"{self.training_and_testing_data}{self.identifier}_test_data.csv"

        if cross_validate:
            self.train_data, self.test_data, self.val_data = get_data_splits_clustering(self.target_path,
                                                                                        self.train_size,
                                                                                        self.test_size, self.val_data)
            self.val_data.to_csv(val_name, index=False)
        else:
            self.train_data, self.test_data = get_training_and_test_data(self.target_path, self.train_size,
                                                                         self.test_size)

        self.train_data.to_csv(train_name, index=False)
        self.test_data.to_csv(test_name, index=False)

    def train(self):
        logger.info('Starting training...')
        identifier_model_path = f"{self.serialized_models_path}{self.identifier}_model.pt"
        train_data_identifier = f"{self.tsne_metrics_dir}{self.identifier}_train_data.csv"
        self.train_data.to_csv(train_data_identifier, index=False)
        smiles_data_train = DataGenerator(self.train_data, descriptor=self.descriptor)  # train
        train_dataloader = DataLoader(smiles_data_train, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.number_of_workers)
        criterion = nn.MSELoss()
        net = AttentionNetwork(self.feature_dim, self.sequence)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        number_of_epochs = 7
        start = time.time()
        model, _ = train_model(train_dataloader, net, criterion,
                               optimizer, number_of_epochs)
        self.single_mode_time = (time.time() - start) / 60
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'descriptor': self.descriptor,
                    'num_of_features': self.feature_dim}, identifier_model_path)
        self.single_model = model


        # sample_size = 80000
        # shap_test_size = int(sample_size * 0.8)
        # shap_number_of_epochs = 7
        # # shap model
        # data_df = pd.read_csv(self.data_csv).sample(sample_size)
        # self.train_for_shap_analyses, self.test_for_shap_analyses = train_test_split(data_df, test_size=shap_test_size,
        #                                                                              random_state=42)
        # train_smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values()) for smile in
        #                 self.train_for_shap_analyses['smile']]
        # train_docking_scores = self.train_for_shap_analyses['docking_score'].tolist()
        # normalized_descriptors = self.scaler.fit_transform(train_smiles)
        # self.model_for_shap_analyses = AttentionNetwork(16, self.sequence)
        # shap_model_optimizer = torch.optim.Adam(self.model_for_shap_analyses.parameters(), lr=0.001)
        #
        # shap_data_gen = ShapAnalysesDataGenerator(normalized_descriptors, train_docking_scores)  # train
        # shap_dataloader = DataLoader(shap_data_gen, batch_size=self.batch_size, shuffle=True,
        #                              num_workers=self.number_of_workers)
        # self.model_for_shap_analyses, _ = train_model(shap_dataloader, self.model_for_shap_analyses, criterion,
        #                                               shap_model_optimizer, shap_number_of_epochs)

    def diagnose(self):
        logger.info('Starting diagnosis...')
        all_train_metrics = []
        df_split = np.array_split(self.val_data, self.number_of_folds)
        all_networks = []
        fold_mse, fold_mae, fold_rsquared = 0, 0, 0
        number_of_epochs = 7
        start_time_train_val = time.time()
        for fold in range(self.number_of_folds):
            net = AttentionNetwork(self.feature_dim, self.sequence)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            temp_data = copy.deepcopy(df_split)
            temp_data.pop(fold)
            temp_data = pd.concat(temp_data)
            smiles_data_train = DataGenerator(df_split[fold], descriptor=self.descriptor)  # train
            train_dataloader = DataLoader(smiles_data_train, batch_size=self.batch_size, shuffle=True,
                                          num_workers=self.number_of_workers)
            fold_test_dataloader_class = DataGenerator(temp_data, descriptor=self.descriptor)
            fold_test_dataloader = DataLoader(fold_test_dataloader_class, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.number_of_workers)
            criterion = nn.MSELoss()
            # training
            model, metrics_dict = train_model(train_dataloader, net, criterion,
                                              optimizer, number_of_epochs)
            all_networks.append(model)
            all_train_metrics.append(metrics_dict)

            # Validate
            fold_predictions = test_model(fold_test_dataloader, model)
            test_smiles_target = temp_data['docking_score'].tolist()
            mse, mae, rsquared = calculate_metrics(fold_predictions, test_smiles_target)
            fold_mse = fold_mse + mse
            fold_mae = fold_mae + mae
            fold_rsquared = fold_rsquared + rsquared
        self.cross_validation_time = (time.time() - start_time_train_val) / 60
        cross_validation_metrics = {"average_fold_mse": fold_mse / self.number_of_folds,
                                    "average_fold_mae": fold_mae / self.number_of_folds,
                                    "average_fold_rsquared": fold_rsquared / self.number_of_folds}

        final_dict = {}
        for i in range(self.number_of_folds):
            final_dict['fold ' + str(i) + ' mse'] = all_train_metrics[i]['training_mse']
        f = pd.DataFrame.from_dict(final_dict)
        f['fold mean'] = f.mean(axis=1)
        average_mse = f['fold mean'].tolist()
        cross_validation_metrics['average_epoch_mse'] = average_mse
        self.cross_validation_metrics = cross_validation_metrics
        self.all_networks = all_networks

    def test(self):
        logger.info('Starting testing...')
        all_models_predictions = []
        smiles_data_test = DataGenerator(self.test_data, descriptor=self.descriptor)
        test_dataloader = DataLoader(smiles_data_test, batch_size=self.batch_size, shuffle=False,
                                     num_workers=self.number_of_workers)
        start_time_test = time.time()
        for fold in range(self.number_of_folds):
            logger.info(f"making fold {fold} predictions")
            test_predictions = test_model(test_dataloader, self.all_networks[fold])
            all_models_predictions.append(test_predictions)
        self.test_time = (time.time() - start_time_test) / 60
        smiles_target = self.test_data['docking_score'].tolist()
        smiles_data = self.test_data['smile'].tolist()
        metrics_dict_test = create_test_metrics(all_models_predictions, smiles_target, self.cross_validate)
        predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, smiles_target,
                                                                          self.cross_validate, self.test_size)
        predictions_and_target_df['smile'] = smiles_data
        self.test_metrics = metrics_dict_test
        self.test_predictions_and_target_df = predictions_and_target_df
        identifier_test_pred_target_df = f"{self.tsne_metrics_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)

    def shap_analyses(self):
        logger.info('Starting Shap Analyses...')
        smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values()) for smile in
                  self.test_for_shap_analyses['smile']]
        normalized_descriptors = self.scaler.fit_transform(smiles)

        def model_predict(smiles):
            smiles_tensor = torch.tensor(smiles, dtype=torch.float32).unsqueeze(0)
            self.model_for_shap_analyses.eval()
            with torch.no_grad():
                outputs = self.model_for_shap_analyses(smiles_tensor)
            outputs = outputs.cpu().numpy()

            return outputs

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
        if self.cross_validate:
            identifier_train_val_metrics = f"{self.training_metrics_dir}{self.identifier}_cross_validation_metrics.csv"
            save_dict(self.cross_validation_metrics, identifier_train_val_metrics)
        identifier_test_metrics = f"{self.testing_metrics_dir}{self.identifier}_test_metrics.csv"
        save_dict(self.test_metrics, identifier_test_metrics)
        identifier_test_pred_target_df = f"{self.test_predictions_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
        project_info_dict = {"training_size": [self.train_size], "testing_size": [self.test_size],
                             'training_time': self.single_mode_time,
                             str(self.number_of_folds) + " fold_validation_time": [self.cross_validation_time],
                             "testing_time": [self.test_time]}
        identifier_project_info = f"{self.project_info_dir}{self.identifier}_project_info.csv"
        save_dict(project_info_dict, identifier_project_info)
        logger.info('Training and Testing information has been saved.')

    @staticmethod
    def inference(input_path, output_path, model_path, batch_size=32, num_workers=10,cluster_analyses=False):
        logger.info('Inference has started...')
        data = pd.read_csv(input_path)
        smiles = data['smile'].tolist()
        docking_score = []
        cluster = []
        if 'docking_score' in data.columns:
            docking_score = data['docking_score'].tolist()
        if 'cluster' in data.columns:
            cluster = data['cluster'].tolist()
        # Load the model
        checkpoint = torch.load(model_path)
        descriptor = checkpoint['descriptor']
        num_of_features = checkpoint['num_of_features']
        model = AttentionNetwork(num_of_features) # TODO: Update this to use sequecnce
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        smiles_data_train = InferenceDataGenerator(pd.read_csv(input_path), descriptor=descriptor)  # train
        torch.set_num_threads(6)
        inference_dataloader = DataLoader(smiles_data_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        predictions = inference(inference_dataloader, model)
        if 'docking_score' in data.columns:
            results_dict = {"smile": smiles, "predicted_docking_score": predictions, "docking_score": docking_score,
                            'cluster': cluster}
        else:
            results_dict = {"smile": smiles, "predicted_docking_scores": predictions}
        file_name = os.path.basename(input_path)
        file_name_without_extension = os.path.splitext(file_name)[0]

        # Construct the new output file path
        new_output_file_path = os.path.join(output_path, f"{file_name_without_extension}_predictions.csv")
        save_dict(results_dict, new_output_file_path)
        if cluster_analyses:
            cluster_errors = {}
            for predicted, actual, cluster in zip(results_dict["predicted_docking_score"], results_dict["docking_score"],
                                                  results_dict["cluster"]):
                # Calculate the absolute error
                error = abs(predicted - actual)

                # Update the cluster_errors dictionary
                if cluster not in cluster_errors:
                    cluster_errors[cluster] = {'total_error': 0, 'count': 0}
                cluster_errors[cluster]['total_error'] += error
                cluster_errors[cluster]['count'] += 1

            # Calculate the average MAE for each cluster
            cluster_average_mae = {str(cluster): round(errors['total_error'] / errors['count'] ,4)for cluster, errors in
                                   cluster_errors.items()}
            cluster_output_file_path = os.path.join(output_path, f"{file_name_without_extension}_cluster_results.csv")
            save_dict({key: [value] for key, value in cluster_average_mae.items()}, cluster_output_file_path)

        logger.info('Inference is Done')

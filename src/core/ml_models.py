import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from config.paths import DATASET_DIR
from config.settings import ML_CONFIG
from features.create_fingerprint_data import create_features
from features.smiles_featurizers import morgan_fingerprints_mac_and_one_hot, mac_keys_fingerprints, one_hot_encode, \
    compute_descriptors
from utils.swift_dock_logger import swift_dock_logger
from utils.utils import calculate_metrics, create_test_metrics, create_fold_predictions_and_target_df, save_dict

logger = swift_dock_logger()


class OtherModels:
    def __init__(self, training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 shap_analyses_dir, all_data, train_size, test_size, val_size, identifier, number_of_folds,
                 regressor, serialized_models_path, descriptor, data_csv, cross_validate, config=None):

        self.all_data = all_data
        self.training_metrics_dir = Path(training_metrics_dir)
        self.testing_metrics_dir = Path(testing_metrics_dir)
        self.test_predictions_dir = Path(test_predictions_dir)
        self.project_info_dir = Path(project_info_dir)
        self.shap_analyses_dir = Path(shap_analyses_dir)
        self.serialized_models_path = Path(serialized_models_path)

        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.regressor = regressor
        self.descriptor = descriptor
        self.data_csv = Path(data_csv)
        self.cross_validate = cross_validate

        self.config = config or ML_CONFIG
        self.random_state = self.config.get("random_state", 42)
        self.shap_sample_size = self.config.get("shap_sample_size", 50)
        self.shap_test_size = self.config.get("shap_test_size", 0.8)

        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

        self.cross_validation_metrics = None
        self.all_regressors = []
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.train_time = None
        self.single_regressor = None

        self.train_for_shap_analyses = None
        self.test_for_shap_analyses = None
        self.model_for_shap_analyses = None
        self.scaler = StandardScaler()

    def get_file_path(self, directory, suffix):
        return Path(directory) / f"{self.identifier}_{suffix}"

    def create_regressor(self):
        return self.regressor()

    def split_data(self):
        self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]

        if self.cross_validate:
            self.x_train = self.x[:self.train_size]
            self.y_train = self.y[:self.train_size]

            self.x_val = self.x[self.train_size:self.train_size + self.val_size]
            self.y_val = self.y[self.train_size:self.train_size + self.val_size]

            self.x_test = self.x[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]
            self.y_test = self.y[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]

        else:
            self.x_train = self.x[:self.train_size]
            self.y_train = self.y[:self.train_size]

            self.x_test = self.x[self.train_size:self.train_size + self.test_size]
            self.y_test = self.y[self.train_size:self.train_size + self.test_size]

    def train(self):
        logger.info(f"Training has started for {self.identifier}")

        start_time_train = time.time()

        regressor = self.create_regressor()
        regressor.fit(self.x_train, self.y_train)

        self.train_time = (time.time() - start_time_train) / 60
        self.single_regressor = regressor

        identifier_model_path = self.get_file_path(self.serialized_models_path, "model.pkl")
        descriptor_dict = {"descriptor": self.descriptor}

        with open(identifier_model_path, "wb") as file:
            pickle.dump((regressor, descriptor_dict), file)

        logger.info(f"Training is Done! {self.identifier}")

        self.train_shap_model()

    def train_shap_model(self):
        data_df = pd.read_csv(self.data_csv).dropna().reset_index(drop=True)
        sample_size = min(self.shap_sample_size, len(data_df))

        if sample_size < 2:
            logger.warning("Skipping SHAP model training because sample size is too small.")
            return

        data_df = data_df.sample(sample_size, random_state=self.random_state)

        self.train_for_shap_analyses, self.test_for_shap_analyses = train_test_split(
            data_df, test_size=self.shap_test_size, random_state=self.random_state)

        train_smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values())
                        for smile in self.train_for_shap_analyses["smile"]]

        train_docking_scores = self.train_for_shap_analyses["docking_score"].tolist()
        normalized_descriptors = self.scaler.fit_transform(train_smiles)

        self.model_for_shap_analyses = self.create_regressor()
        self.model_for_shap_analyses.fit(normalized_descriptors, train_docking_scores)

    def diagnose(self):
        if not self.cross_validate or self.number_of_folds <= 1:
            logger.info("Skipping validation because cross validation is disabled.")
            return

        logger.info(f"Validation has started for {self.identifier}")

        kf = KFold(n_splits=self.number_of_folds, shuffle=True, random_state=self.random_state)
        regressors_list = []

        fold_mse = []
        fold_mae = []
        fold_rsquared = []

        start_time_train = time.time()

        for train_index, test_index in kf.split(self.x_val):
            x_train_fold, x_test_fold = self.x_val[train_index], self.x_val[test_index]
            y_train_fold, y_test_fold = self.y_val[train_index], self.y_val[test_index]

            regressor = self.create_regressor()
            regressor.fit(x_train_fold, y_train_fold)

            regressors_list.append(regressor)

            predictions = regressor.predict(x_test_fold)
            mse, mae, rsquared = calculate_metrics(predictions, y_test_fold)

            fold_mse.append(mse)
            fold_mae.append(mae)
            fold_rsquared.append(rsquared)

        self.cross_validation_time = (time.time() - start_time_train) / 60

        self.cross_validation_metrics = {
            "average_fold_mse": [float(np.mean(fold_mse))],
            "average_fold_mae": [float(np.mean(fold_mae))],
            "average_fold_rsquared": [float(np.mean(fold_rsquared))]
        }

        self.all_regressors = regressors_list

        identifier_train_val_metrics = self.get_file_path(self.training_metrics_dir, "cross_validation_metrics.csv")
        save_dict(self.cross_validation_metrics, identifier_train_val_metrics)

    def test(self):
        logger.info(f"Testing has started for {self.identifier}")

        all_models_predictions = []
        regressors = self.all_regressors if self.cross_validate and self.all_regressors else [self.single_regressor]

        start_time_test = time.time()

        for fold, regressor in enumerate(regressors):
            logger.info(f"Making predictions for model {fold}")
            predictions = regressor.predict(self.x_test)
            all_models_predictions.append(predictions)

        self.test_time = (time.time() - start_time_test) / 60

        use_cross_validation_results = self.cross_validate and len(all_models_predictions) > 1

        self.test_metrics = create_test_metrics(all_models_predictions, self.y_test, use_cross_validation_results)
        self.test_predictions_and_target_df = create_fold_predictions_and_target_df(
            all_models_predictions, self.y_test, use_cross_validation_results, self.test_size)

        logger.info(f"Testing is Done! {self.identifier}")
        return all_models_predictions

    def shap_analyses(self):
        if self.model_for_shap_analyses is None or self.test_for_shap_analyses is None:
            logger.warning("Skipping SHAP analyses because SHAP model was not trained.")
            return

        logger.info("Starting Shap Analyses.")

        smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values())
                  for smile in self.test_for_shap_analyses["smile"]]

        normalized_descriptors = self.scaler.transform(smiles)

        def model_predict(smiles_data):
            return self.model_for_shap_analyses.predict(smiles_data)

        feature_names = [
            "mol_weight", "num_atoms", "num_bonds", "num_rotatable_bonds",
            "num_h_donors", "num_h_acceptors", "logp", "mr", "tpsa",
            "num_rings", "num_aromatic_rings", "hall_kier_alpha",
            "fraction_csp3", "num_nitrogens", "num_oxygens", "num_sulphurs"
        ]

        masker = shap.maskers.Independent(data=normalized_descriptors)
        explainer = shap.explainers.Permutation(model_predict, masker)
        shap_values = explainer.shap_values(normalized_descriptors)

        shap_analyses_csv_dir = self.get_file_path(self.shap_analyses_dir, "shap_analyses.csv")
        shap_analyses_summary_plot = self.get_file_path(self.shap_analyses_dir, "shap_summary_plot.png")
        shap_analyses_feature_importance = self.get_file_path(self.shap_analyses_dir, "shap_feature_importance.png")

        normalized_descriptors_df = pd.DataFrame(normalized_descriptors, columns=feature_names)
        shap_df = pd.DataFrame(shap_values, columns=feature_names)

        avg_shap = shap_df.abs().mean().sort_values(ascending=False)
        avg_shap.to_csv(shap_analyses_csv_dir)

        plt.figure(figsize=(10, 6))
        avg_shap.sort_values().plot(kind="barh")
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("Feature")
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(shap_analyses_feature_importance)
        plt.close()

        shap_long_df = shap_df.melt(var_name="feature", value_name="shap_value")
        feature_order = avg_shap.index.tolist()

        shap_long_df["feature"] = pd.Categorical(
            shap_long_df["feature"],
            categories=feature_order,
            ordered=True
        )

        plt.figure(figsize=(10, 7))
        sns.stripplot(data=shap_long_df, x="shap_value", y="feature", size=4, alpha=0.6)
        plt.xlabel("SHAP value")
        plt.ylabel("Feature")
        plt.title("SHAP Summary")
        plt.tight_layout()
        plt.savefig(shap_analyses_summary_plot)
        plt.close()

        normalized_descriptors_csv = self.get_file_path(self.shap_analyses_dir, "normalized_descriptors.csv")
        shap_values_csv = self.get_file_path(self.shap_analyses_dir, "shap_values.csv")

        normalized_descriptors_df.to_csv(normalized_descriptors_csv, index=False)
        shap_df.to_csv(shap_values_csv, index=False)

        logger.info("SHAP Analyses finished.")

    def evaluate_structural_diversity(self):
        logger.info("Starting Structural Diversity Analyses.")

        if self.train_for_shap_analyses is None or self.test_for_shap_analyses is None:
            logger.warning("Skipping structural diversity because SHAP data is not available.")
            return

        tsne_visualization_dir = self.get_file_path(self.shap_analyses_dir, "tsne_visualization.png")
        tsne_dir = self.get_file_path(self.shap_analyses_dir, "tsne_data.csv")

        train_fps = self.get_mac_fingerprints(self.train_for_shap_analyses)
        test_fps = self.get_mac_fingerprints(self.test_for_shap_analyses)

        all_fps = np.vstack([train_fps, test_fps])
        tsne = TSNE(n_components=2, random_state=self.random_state).fit_transform(all_fps)

        pd.DataFrame(tsne, columns=["Dimension_1", "Dimension_2"]).to_csv(tsne_dir, index=False)

        plt.figure(figsize=(10, 7))
        plt.scatter(tsne[:len(train_fps), 0], tsne[:len(train_fps), 1], color="blue", label="Training Data", alpha=0.5)
        plt.scatter(tsne[len(train_fps):, 0], tsne[len(train_fps):, 1], color="red", label="Test Data", alpha=0.5)
        plt.legend(loc="upper right")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.title("t-SNE Visualization of Training vs Test Data")
        plt.savefig(tsne_visualization_dir, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def get_mac_fingerprints(data):
        fps = []

        for smile in data["smile"]:
            mol = Chem.MolFromSmiles(smile)

            if mol is None:
                logger.warning(f"Invalid SMILES skipped: {smile}")
                continue

            maccs_key = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(maccs_key, arr)
            fps.append(arr)

        return np.array(fps)

    def plot_docking_vs_mol_weight(self):
        logger.info(f"Started creating plot for mol weights of {self.identifier}")

        size_cor_plot_dir = self.get_file_path(self.shap_analyses_dir, "mol_weight.png")
        df = pd.read_csv(self.data_csv).dropna()

        df["mol_weight"] = [
            Descriptors.MolWt(Chem.MolFromSmiles(smile)) for smile in df["smile"]
        ]

        plt.figure(figsize=(10, 6))
        sns.kdeplot(x=df["mol_weight"], y=df["docking_score"], cmap="inferno", fill=True)
        plt.title("Docking Score vs Molecular Weight")
        plt.xlabel("Molecular Weight")
        plt.ylabel("Docking Score")
        plt.grid(True)
        plt.savefig(size_cor_plot_dir)
        plt.close()

    def save_results(self):
        identifier_test_metrics = self.get_file_path(self.testing_metrics_dir, "test_metrics.csv")
        identifier_test_pred_target_df = self.get_file_path(self.test_predictions_dir, "test_predictions.csv")
        identifier_project_info = self.get_file_path(self.project_info_dir, "project_info.csv")

        save_dict(self.test_metrics, identifier_test_metrics)
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)

        project_info_dict = {
            "training_size": [self.train_size],
            "testing_size": [self.test_size],
            f"{self.number_of_folds}_fold_validation_time": [self.cross_validation_time],
            "training_time": [self.train_time],
            "testing_time": [self.test_time]
        }

        save_dict(project_info_dict, identifier_project_info)
        logger.info(f"Saving done for {self.identifier}")

    @staticmethod
    def inference(input_path, output_path, model_path):
        logger.info("Inference has started.")

        input_path = Path(input_path)
        output_path = Path(output_path)
        model_path = Path(model_path)

        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(input_path)
        smiles = df["smile"].tolist()

        tmp_target_name = "tmp"
        tmp_path = DATASET_DIR / f"{tmp_target_name}.csv"

        tmp_df = df.copy()
        tmp_df["docking_score"] = 0
        tmp_df.to_csv(tmp_path, index=False)

        with open(model_path, "rb") as file:
            pickle_model, descriptor_dict = pickle.load(file)

        descriptor = descriptor_dict["descriptor"]

        featurizers = {
            "onehot": [3500, one_hot_encode],
            "morgan_onehot_mac": [4691, morgan_fingerprints_mac_and_one_hot],
            "mac": [167, mac_keys_fingerprints]
        }

        descriptor_dims = {
            "onehot": 3500 + 1,
            "morgan_onehot_mac": 4691 + 1,
            "mac": 167 + 1
        }

        create_features([tmp_target_name], {descriptor: {
            "feature_dim": featurizers[descriptor][0],
            "function": featurizers[descriptor][1]
        }})

        data_set_path = DATASET_DIR / f"{tmp_target_name}_{descriptor}.dat"
        data = np.memmap(data_set_path, dtype=np.float32)

        target_length = data.shape[0] // descriptor_dims[descriptor]
        data = data.reshape((target_length, descriptor_dims[descriptor]))

        x = data[:, :-1]
        predictions = pickle_model.predict(x)

        results_dict = {
            "smile": smiles,
            "docking_score": predictions
        }

        save_dict(results_dict, output_path / "results.csv")

        if tmp_path.exists():
            tmp_path.unlink()

        if data_set_path.exists():
            data_set_path.unlink()

        logger.info("Inference is finished.")
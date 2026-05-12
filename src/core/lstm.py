import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, Descriptors
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config.paths import PROJECT_ROOT
from config.settings import DEFAULT_CONFIG
from core.data_generator import DataGenerator, ShapAnalysesDataGenerator, InferenceDataGenerator
from core.model import AttentionNetwork
from features.smiles_featurizers import compute_descriptors
from llm.google_gemma import GemmaGoogle
from train.trainer import train_model
from utils.swift_dock_logger import swift_dock_logger
from utils.utils import get_data_splits_clustering, get_data_splits, get_training_and_test_data, test_model, \
    calculate_metrics, create_test_metrics, create_fold_predictions_and_target_df, save_dict, inference, get_target_seq, \
    prepare_llm_instruction

logger = swift_dock_logger()
warnings.filterwarnings("ignore")


class SwiftDock:
    def __init__(self, training_and_testing_data, training_metrics_dir, testing_metrics_dir, test_predictions_dir,
                 project_info_dir, target_path, train_size, test_size, val_size, identifier, number_of_folds,
                 descriptor, feature_dim, serialized_models_path, cross_validate, shap_analyses_dir,
                 tsne_analyses_dir, data_csv, batch_size, number_of_workers, sequence,
                 split_base_on_clustering, config=None):

        self.target_path = target_path
        self.training_and_testing_data = Path(training_and_testing_data)
        self.training_metrics_dir = Path(training_metrics_dir)
        self.testing_metrics_dir = Path(testing_metrics_dir)
        self.test_predictions_dir = Path(test_predictions_dir)
        self.project_info_dir = Path(project_info_dir)
        self.serialized_models_path = Path(serialized_models_path)
        self.shap_analyses_dir = Path(shap_analyses_dir)
        self.tsne_metrics_dir = Path(tsne_analyses_dir)
        self.data_csv = Path(data_csv)

        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.descriptor = descriptor
        self.feature_dim = feature_dim
        self.cross_validate = cross_validate
        self.batch_size = batch_size
        self.number_of_workers = number_of_workers
        self.sequence = sequence
        self.split_base_on_clustering = split_base_on_clustering

        self.config = config or DEFAULT_CONFIG
        self.learning_rate = self.config.get("learning_rate", 0.001)
        self.number_of_epochs = self.config.get("number_of_epochs", 1)
        self.shap_sample_size = self.config.get("shap_sample_size", 100)
        self.shap_test_size = self.config.get("shap_test_size", 0.8)
        self.shap_number_of_epochs = self.config.get("shap_number_of_epochs", 1)
        self.random_state = self.config.get("random_state", 42)
        self.torch_num_threads = self.config.get("torch_num_threads", 6)

        self.train_data = None
        self.test_data = None
        self.cross_validation_metrics = None
        self.all_networks = []
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.single_mode_time = None
        self.single_model = None
        self.model_for_shap_analyses = None
        self.train_for_shap_analyses = None
        self.test_for_shap_analyses = None
        self.scaler = StandardScaler()

    @staticmethod
    def to_dataframe(data):
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame(data, columns=["smile", "docking_score"])

        df = df.reset_index(drop=True)

        if "smile" not in df.columns or "docking_score" not in df.columns:
            if len(df.columns) >= 2:
                df = df.rename(columns={df.columns[0]: "smile", df.columns[1]: "docking_score"})
            else:
                raise ValueError("Data must contain smile and docking_score columns.")

        df["smile"] = df["smile"].astype(str)
        df["docking_score"] = pd.to_numeric(df["docking_score"])

        return df[["smile", "docking_score"]].reset_index(drop=True)

    def get_file_path(self, directory, suffix):
        return Path(directory) / f"{self.identifier}_{suffix}"

    def create_dataloader(self, data, shuffle):
        data = self.to_dataframe(data)
        dataset = DataGenerator(data, descriptor=self.descriptor)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.number_of_workers)

    def create_model_and_optimizer(self):
        if not self.sequence:
            raise ValueError(f"Sequence is empty for identifier: {self.identifier}")

        model = AttentionNetwork(self.feature_dim, len(self.sequence))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return model, optimizer

    def split_data(self):
        train_name = self.get_file_path(self.training_and_testing_data, "train_data.csv")
        val_name = self.get_file_path(self.training_and_testing_data, "val_data.csv")
        test_name = self.get_file_path(self.training_and_testing_data, "test_data.csv")

        if self.cross_validate:
            if self.split_base_on_clustering:
                self.train_data, self.test_data, self.val_data = get_data_splits_clustering(
                    self.target_path, self.train_size, self.test_size, self.val_size)
            else:
                self.train_data, self.test_data, self.val_data = get_data_splits(
                    self.target_path, self.train_size, self.test_size, self.val_size)

            self.val_data = self.to_dataframe(self.val_data)
            self.val_data.to_csv(val_name, index=False)

        else:
            self.train_data, self.test_data = get_training_and_test_data(
                self.target_path, self.train_size, self.test_size)

        self.train_data = self.to_dataframe(self.train_data)
        self.test_data = self.to_dataframe(self.test_data)

        self.train_data.to_csv(train_name, index=False)
        self.test_data.to_csv(test_name, index=False)

    def train(self):
        logger.info("Starting training...")

        identifier_model_path = self.get_file_path(self.serialized_models_path, "model.pt")
        train_data_identifier = self.get_file_path(self.tsne_metrics_dir, "train_data.csv")

        self.train_data.to_csv(train_data_identifier, index=False)

        train_dataloader = self.create_dataloader(self.train_data, shuffle=True)
        criterion = nn.MSELoss()
        model, optimizer = self.create_model_and_optimizer()

        start = time.time()
        model, _ = train_model(train_dataloader, model, criterion, optimizer, self.number_of_epochs)
        self.single_mode_time = (time.time() - start) / 60

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "descriptor": self.descriptor,
            "num_of_features": self.feature_dim,
            "sequence_length": len(self.sequence)
        }, identifier_model_path)

        self.single_model = model

        if not self.cross_validate:
            self.all_networks = [self.single_model]

        self.train_shap_model()

    def train_shap_model(self):
        data_df = pd.read_csv(self.data_csv).dropna().reset_index(drop=True)
        sample_size = min(self.shap_sample_size, len(data_df))

        if sample_size < 2:
            logger.warning("Skipping SHAP model training because the sample size is too small.")
            return

        data_df = data_df.sample(sample_size, random_state=self.random_state)
        self.train_for_shap_analyses, self.test_for_shap_analyses = train_test_split(
            data_df, test_size=self.shap_test_size, random_state=self.random_state)

        train_smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values())
                        for smile in self.train_for_shap_analyses["smile"]]

        train_docking_scores = self.train_for_shap_analyses["docking_score"].tolist()
        normalized_descriptors = self.scaler.fit_transform(train_smiles)

        self.model_for_shap_analyses = AttentionNetwork(16, len(self.sequence))
        optimizer = torch.optim.Adam(self.model_for_shap_analyses.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        shap_data_gen = ShapAnalysesDataGenerator(normalized_descriptors, train_docking_scores)
        shap_dataloader = DataLoader(shap_data_gen, batch_size=self.batch_size, shuffle=True,
                                     num_workers=self.number_of_workers)

        self.model_for_shap_analyses, _ = train_model(
            shap_dataloader, self.model_for_shap_analyses, criterion, optimizer, self.shap_number_of_epochs)

    def diagnose(self):
        if self.number_of_folds <= 1:
            logger.info("Skipping diagnosis because number_of_folds <= 1")
            return

        logger.info("Starting diagnosis...")

        all_train_metrics = []
        all_networks = []
        fold_mse, fold_mae, fold_rsquared = 0, 0, 0

        val_data = self.to_dataframe(self.val_data)
        df_split = [self.to_dataframe(split) for split in np.array_split(val_data, self.number_of_folds)]

        start_time_train_val = time.time()

        for fold in range(self.number_of_folds):
            logger.info(f"Starting fold {fold + 1}/{self.number_of_folds}")

            model, optimizer = self.create_model_and_optimizer()
            criterion = nn.MSELoss()

            train_data = df_split[fold]
            temp_data = [data for i, data in enumerate(df_split) if i != fold]

            if not temp_data:
                logger.warning(f"Skipping fold {fold + 1} because there is no validation data.")
                continue

            fold_val_data = pd.concat(temp_data, ignore_index=True)

            train_dataloader = self.create_dataloader(train_data, shuffle=True)
            fold_test_dataloader = self.create_dataloader(fold_val_data, shuffle=False)

            model, metrics_dict = train_model(train_dataloader, model, criterion, optimizer, self.number_of_epochs)

            all_networks.append(model)
            all_train_metrics.append(metrics_dict)

            fold_predictions = test_model(fold_test_dataloader, model)
            fold_target = fold_val_data["docking_score"].tolist()

            mse, mae, rsquared = calculate_metrics(fold_predictions, fold_target)

            fold_mse += mse
            fold_mae += mae
            fold_rsquared += rsquared

        self.cross_validation_time = (time.time() - start_time_train_val) / 60

        if not all_networks:
            logger.warning("No fold models were trained. Falling back to the single trained model.")
            self.all_networks = [self.single_model]
            return

        self.cross_validation_metrics = {
            "average_fold_mse": fold_mse / len(all_networks),
            "average_fold_mae": fold_mae / len(all_networks),
            "average_fold_rsquared": fold_rsquared / len(all_networks)
        }

        final_dict = {}

        for i, metrics in enumerate(all_train_metrics):
            final_dict[f"fold {i} mse"] = metrics["training_mse"]

        metrics_df = pd.DataFrame.from_dict(final_dict)
        metrics_df["fold mean"] = metrics_df.mean(axis=1)

        self.cross_validation_metrics["average_epoch_mse"] = metrics_df["fold mean"].tolist()
        self.all_networks = all_networks

    def test(self):
        logger.info("Starting testing...")

        all_models_predictions = []
        test_dataloader = self.create_dataloader(self.test_data, shuffle=False)
        models = self.all_networks if self.cross_validate and self.all_networks else [self.single_model]

        start_time_test = time.time()

        for fold, model in enumerate(models):
            logger.info(f"Making fold {fold} predictions")
            test_predictions = test_model(test_dataloader, model)
            all_models_predictions.append(test_predictions)

        self.test_time = (time.time() - start_time_test) / 60

        smiles_target = self.test_data["docking_score"].tolist()
        smiles_data = self.test_data["smile"].tolist()

        use_cross_validation_results = self.cross_validate and len(all_models_predictions) > 1

        self.test_metrics = create_test_metrics(all_models_predictions, smiles_target, use_cross_validation_results)
        self.test_predictions_and_target_df = create_fold_predictions_and_target_df(
            all_models_predictions, smiles_target, use_cross_validation_results, len(smiles_target))

        self.test_predictions_and_target_df["smile"] = smiles_data

        identifier_test_pred_target_df = self.get_file_path(self.tsne_metrics_dir, "test_predictions.csv")
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)

    def shap_analyses(self):
        if self.model_for_shap_analyses is None or self.test_for_shap_analyses is None:
            logger.warning("Skipping SHAP analyses because the SHAP model was not trained.")
            return

        logger.info("Starting Shap Analyses...")

        smiles = [list(compute_descriptors(Chem.MolFromSmiles(smile)).values())
                  for smile in self.test_for_shap_analyses["smile"]]

        normalized_descriptors = self.scaler.transform(smiles)

        def model_predict(smiles_data):
            smiles_tensor = torch.tensor(smiles_data, dtype=torch.float32).unsqueeze(0)
            self.model_for_shap_analyses.eval()

            with torch.no_grad():
                outputs = self.model_for_shap_analyses(smiles_tensor)

            return outputs.cpu().numpy()

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
        logger.info("Starting Structural Diversity Analyses...")

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
        if self.cross_validate and self.cross_validation_metrics:
            identifier_train_val_metrics = self.get_file_path(self.training_metrics_dir, "cross_validation_metrics.csv")
            save_dict(self.cross_validation_metrics, identifier_train_val_metrics)

        identifier_test_metrics = self.get_file_path(self.testing_metrics_dir, "test_metrics.csv")
        identifier_test_pred_target_df = self.get_file_path(self.test_predictions_dir, "test_predictions.csv")
        identifier_project_info = self.get_file_path(self.project_info_dir, "project_info.csv")

        save_dict(self.test_metrics, identifier_test_metrics)
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)

        project_info_dict = {
            "training_size": [self.train_size],
            "testing_size": [self.test_size],
            "training_time": [self.single_mode_time],
            f"{self.number_of_folds}_fold_validation_time": [self.cross_validation_time],
            "testing_time": [self.test_time]
        }

        save_dict(project_info_dict, identifier_project_info)
        logger.info("Training and Testing information has been saved.")

    @staticmethod
    def inference(input_path, output_path, model_path, batch_size=32, num_workers=10, cluster_analyses=False):
        logger.info("Inference has started...")

        input_path = Path(input_path)
        output_path = Path(output_path)
        model_path = Path(model_path)

        output_path.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(input_path)
        smiles = data["smile"].tolist()

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        descriptor = checkpoint["descriptor"]
        num_of_features = checkpoint["num_of_features"]
        sequence_length = checkpoint.get("sequence_length", 1)

        model = AttentionNetwork(num_of_features, sequence_length)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.eval()

        smiles_data = InferenceDataGenerator(data, descriptor=descriptor)
        inference_dataloader = DataLoader(smiles_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        torch.set_num_threads(DEFAULT_CONFIG.get("torch_num_threads", 6))
        predictions = inference(inference_dataloader, model)

        results_dict = {
            "smile": smiles,
            "predicted_docking_score": predictions
        }

        if "docking_score" in data.columns:
            results_dict["docking_score"] = data["docking_score"].tolist()

        if "cluster" in data.columns:
            results_dict["cluster"] = data["cluster"].tolist()

        file_name_without_extension = input_path.stem
        new_output_file_path = output_path / f"{file_name_without_extension}_predictions.csv"

        save_dict(results_dict, new_output_file_path)

        if cluster_analyses and "cluster" in results_dict and "docking_score" in results_dict:
            SwiftDock.save_cluster_results(results_dict, output_path, file_name_without_extension)

        logger.info("Inference is Done")

    @staticmethod
    def save_cluster_results(results_dict, output_path, file_name_without_extension):
        output_path = Path(output_path)
        cluster_errors = {}

        for predicted, actual, cluster in zip(results_dict["predicted_docking_score"],
                                              results_dict["docking_score"], results_dict["cluster"]):
            error = abs(predicted - actual)

            if cluster not in cluster_errors:
                cluster_errors[cluster] = {"total_error": 0, "count": 0}

            cluster_errors[cluster]["total_error"] += error
            cluster_errors[cluster]["count"] += 1

        cluster_average_mae = {
            str(cluster): round(errors["total_error"] / errors["count"], 4)
            for cluster, errors in cluster_errors.items()
        }

        cluster_output_file_path = output_path / f"{file_name_without_extension}_cluster_results.csv"
        save_dict({key: [value] for key, value in cluster_average_mae.items()}, cluster_output_file_path)

    @staticmethod
    def inference_with_llm(model_path, smile, target):
        data = pd.DataFrame({"smile": [smile, smile]})
        mol = Chem.MolFromSmiles(smile)
        smile_properties = compute_descriptors(mol)

        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        descriptor = checkpoint["descriptor"]
        num_of_features = checkpoint["num_of_features"]

        model = AttentionNetwork(num_of_features, len(get_target_seq(target)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.eval()

        smiles_data = InferenceDataGenerator(data, descriptor=descriptor)
        inference_dataloader = DataLoader(smiles_data, batch_size=2, shuffle=False, num_workers=0)

        predicted_docking_score = inference(inference_dataloader, model)
        instruction = prepare_llm_instruction(smile, smile_properties, predicted_docking_score[0])

        weights_config_path = PROJECT_ROOT / "weights_config.yml"
        llm = GemmaGoogle(weights_config_path)

        print("Predicting result...")
        return llm.predict(instruction)
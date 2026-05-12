import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.seq_dict import sequence_dict


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


def get_training_and_test_data(data, train_size, test_size, random_state=42):
    data = to_dataframe(data)

    if train_size + test_size > len(data):
        raise ValueError(f"train_size + test_size cannot exceed data size. Got {train_size + test_size}, data={len(data)}")

    sampled_data = data.sample(train_size + test_size, random_state=random_state).reset_index(drop=True)
    train_data = sampled_data.iloc[:train_size].reset_index(drop=True)
    test_data = sampled_data.iloc[train_size:train_size + test_size].reset_index(drop=True)

    return train_data, test_data


def get_data_splits(data, train_size, test_size, val_size, random_state=42):
    data = to_dataframe(data)
    total_size = train_size + test_size + val_size

    if total_size > len(data):
        raise ValueError(f"train_size + test_size + val_size cannot exceed data size. Got {total_size}, data={len(data)}")

    sampled_data = data.sample(total_size, random_state=random_state).reset_index(drop=True)

    train_data = sampled_data.iloc[:train_size].reset_index(drop=True)
    val_data = sampled_data.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_data = sampled_data.iloc[train_size + val_size:train_size + val_size + test_size].reset_index(drop=True)

    return train_data, test_data, val_data


def get_data_splits_clustering(data, train_size, test_size, val_size, random_state=42):
    data = to_dataframe(data)

    if "cluster" not in data.columns:
        return get_data_splits(data, train_size, test_size, val_size, random_state=random_state)

    total_size = train_size + test_size + val_size

    if total_size > len(data):
        raise ValueError(f"train_size + test_size + val_size cannot exceed data size. Got {total_size}, data={len(data)}")

    sampled_data = data.sample(total_size, random_state=random_state).reset_index(drop=True)

    train_data = sampled_data.iloc[:train_size].reset_index(drop=True)
    val_data = sampled_data.iloc[train_size:train_size + val_size].reset_index(drop=True)
    test_data = sampled_data.iloc[train_size + val_size:train_size + val_size + test_size].reset_index(drop=True)

    return train_data, test_data, val_data


def test_model(test_dataloader, model):
    predictions = []

    model.eval()

    with torch.no_grad():
        for data in test_dataloader:
            features, _ = data
            features = features.squeeze()
            features = features.unsqueeze(0)

            outputs = model(features).double()
            outputs = outputs.squeeze(0).detach().cpu().numpy().flatten().tolist()

            predictions.extend(outputs)

    return predictions


def inference(inference_dataloader, model):
    predictions = []

    model.eval()

    with torch.no_grad():
        for data in inference_dataloader:
            features = data
            features = features.squeeze()
            features = features.unsqueeze(0)

            outputs = model(features).double()
            outputs = outputs.squeeze(0).detach().cpu().numpy().flatten().tolist()

            predictions.extend(outputs)

    return predictions


def calculate_metrics(predictions, target):
    predictions = np.array(predictions, dtype=float)
    target = np.array(target, dtype=float)

    mse = mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    rsquared = r2_score(target, predictions)

    return mse, mae, rsquared


def create_test_metrics(all_models_predictions, smiles_target, cross_validate):
    metrics_dict = {}


    if not all_models_predictions:
        raise ValueError("No predictions were provided.")

    if cross_validate:
        fold_mse = []
        fold_mae = []
        fold_rsquared = []

        for fold, predictions in enumerate(all_models_predictions):
            mse, mae, rsquared = calculate_metrics(predictions, smiles_target)

            metrics_dict[f"fold_{fold}_mse"] = [mse]
            metrics_dict[f"fold_{fold}_mae"] = [mae]
            metrics_dict[f"fold_{fold}_rsquared"] = [rsquared]

            fold_mse.append(mse)
            fold_mae.append(mae)
            fold_rsquared.append(rsquared)

        metrics_dict["average_mse"] = [float(np.mean(fold_mse))]
        metrics_dict["average_mae"] = [float(np.mean(fold_mae))]
        metrics_dict["average_rsquared"] = [float(np.mean(fold_rsquared))]

    else:
        predictions = all_models_predictions[0]
        mse, mae, rsquared = calculate_metrics(predictions, smiles_target)

        metrics_dict["mse"] = [mse]
        metrics_dict["mae"] = [mae]
        metrics_dict["rsquared"] = [rsquared]

    return metrics_dict


def create_fold_predictions_and_target_df(all_models_predictions, smiles_target, cross_validate, test_size=None):
    results_dict = {
        "target": smiles_target
    }

    if cross_validate:
        for fold, predictions in enumerate(all_models_predictions):
            results_dict[f"fold_{fold}_prediction"] = predictions

        predictions_array = np.array(all_models_predictions, dtype=float)
        results_dict["average_prediction"] = predictions_array.mean(axis=0).tolist()

    else:
        results_dict["prediction"] = all_models_predictions[0]

    return pd.DataFrame(results_dict)


def create_test_metrics_old(all_models_predictions, smiles_target, cross_validate):
    return create_test_metrics(all_models_predictions, smiles_target, cross_validate)


def save_dict(data_dict, output_path):
    output_path = str(output_path)

    if isinstance(data_dict, pd.DataFrame):
        data_dict.to_csv(output_path, index=False)
        return

    normalized_dict = {}

    for key, value in data_dict.items():
        if isinstance(value, list):
            normalized_dict[key] = value
        else:
            normalized_dict[key] = [value]

    pd.DataFrame.from_dict(normalized_dict).to_csv(output_path, index=False)


def get_target_seq(target):
    return sequence_dict.get(target, "")


def prepare_llm_instruction(smile, smile_properties, predicted_docking_score):
    return f"""
        SMILES: {smile}
        
        Molecular properties:
        {json.dumps(smile_properties, indent=2)}
        
        Predicted docking score:
        {predicted_docking_score}
        
        Explain the possible significance of this molecule in the context of molecular docking.
        """
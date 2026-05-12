from pathlib import Path

readme = """# Swift Dock 🚀

Swift Dock is a machine learning framework for predicting molecular docking scores of ligands against target proteins. The main goal is to reduce the need for expensive large-scale docking calculations by training regression models on a small subset of explicitly docked molecules, then using the trained models to predict docking scores for the remaining molecules in a chemical library.

This project supports two main modeling workflows:

- LSTM-based neural network with attention, implemented in PyTorch.

- Classical machine learning regressors, including XGBoost, Decision Tree Regression, and Stochastic Gradient Descent Regression.

The project was originally developed as part of the thesis:

Accelerating Molecular Docking Using Machine Learning Methods

---

## Environment Setup 🛠️

Swift Dock now uses Python 3.12 for both Apple Silicon and Intel Macs.

### 1. Create a virtual environment

From the project root:

    python3.12 -m venv venv3.12

### 2. Activate the environment

    source venv3.12/bin/activate

### 3. Install requirements

    pip install --upgrade pip setuptools wheel

    pip install -r requirements.txt

### 4. Mark `src` as Sources Root in PyCharm

In PyCharm:

    Right-click src → Mark Directory as → Sources Root

This is important so imports like the following work correctly:

    from config.paths import DATASET_DIR

    from core.lstm import SwiftDock

    from features.smiles_featurizers import mac_keys_fingerprints

---

## Dataset Format

Add your target CSV files inside the `datasets/` folder.

Each dataset should follow this format:

    smile,docking_score

    CCO,-5.6

    CCN,-6.1

    CCC,-4.9

Required columns:

    smile

    docking_score

Example:

    datasets/sample_input.csv

---

## Protein Sequences

Target protein sequences are stored in:

    src/config/seq_dict.py

If you train an LSTM model on a new target, add the target name and sequence to `sequence_dict`.

Example:

    sequence_dict = {

        "sample_input": "MEALIPHH",

        "Drp1_GTPase": "MEALIPHH"

    }

The target name must match the dataset name.

For example:

    datasets/sample_input.csv

should have:

    "sample_input": "..."

inside `seq_dict.py`.

---

# Training Using LSTM 🧠

The LSTM workflow trains a PyTorch attention-based neural network directly from the CSV file.

## Basic command

From the project root:

    python src/train/main_lstm.py --input sample_input --descriptors mac --training_sizes 50

## Command format

    python src/train/main_lstm.py --input <TARGET_NAME> --descriptors <DESCRIPTOR> --training_sizes <TRAINING_SIZE>

Available descriptors:

    mac

    onehot

    morgan_onehot_mac

---

## LSTM training with cross-validation

Use `--cross_validate` with the number of folds.

Example:

    python src/train/main_lstm.py --input sample_input --descriptors mac --training_sizes 50 --cross_validate 5

Note:

    --cross_validate 1

is treated like normal training without cross-validation.

---

## More LSTM examples

### Train with multiple descriptors

    python src/train/main_lstm.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50

### Train with multiple descriptors and training sizes

    python src/train/main_lstm.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100

### Train with multiple targets

    python src/train/main_lstm.py --input sample_input sample_input_2 --descriptors mac morgan_onehot_mac --training_sizes 50 100

---

## LSTM Output

LSTM results are saved inside:

    results_seq/

Main output folders:

    results_seq/project_info

    results_seq/serialized_models

    results_seq/test_predictions

    results_seq/testing_metrics

    results_seq/validation_metrics

    results_seq/training_testing_data

    results_seq/shap_analyses

    results_seq/tsne_analyses

File naming format:

    lstm_<target>_<descriptor>_<training_size>_<output_type>

Example:

    lstm_sample_input_mac_50_model.pt

---

# Training Classical ML Models 🌳

Classical ML models use precomputed feature files stored as `.dat` files.

The workflow has two steps:

1. Create fingerprint feature files.

2. Train ML regressors.

---

## Step 1: Create Feature Files

Before training classical ML models, create the `.dat` files.

    python src/features/create_fingerprint_data.py --input sample_input --descriptors mac

Command format:

    python src/features/create_fingerprint_data.py --input <TARGET_NAME> --descriptors <DESCRIPTOR>

Available descriptors:

    mac

    onehot

    morgan_onehot_mac

Example with multiple descriptors:

    python src/features/create_fingerprint_data.py --input sample_input --descriptors mac morgan_onehot_mac

This creates files like:

    datasets/sample_input_mac.dat

    datasets/sample_input_morgan_onehot_mac.dat

---

## Step 2: Train ML Models

    python src/train/main_ml.py --input sample_input --descriptors mac --training_sizes 50 --regressors sgdreg

Command format:

    python src/train/main_ml.py --input <TARGET_NAME> --descriptors <DESCRIPTOR> --training_sizes <TRAINING_SIZE> --regressors <REGRESSOR>

Available regressors:

    sgdreg

    xgboost

    decision_tree

Available descriptors:

    mac

    onehot

    morgan_onehot_mac

---

## ML training with cross-validation

    python src/train/main_ml.py --input sample_input --descriptors mac --training_sizes 50 --regressors xgboost --cross_validate true

---

## More ML examples

### Train with multiple descriptors

    python src/train/main_ml.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 --regressors sgdreg

### Train with multiple training sizes

    python src/train/main_ml.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100 --regressors sgdreg

### Train with multiple regressors

    python src/train/main_ml.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100 --regressors sgdreg xgboost decision_tree

---

## ML Output

Classical ML results are saved inside:

    results/

Main output folders:

    results/project_info

    results/serialized_models

    results/test_predictions

    results/testing_metrics

    results/validation_metrics

    results/shap_analyses

File naming format:

    <regressor>_<target>_<descriptor>_<training_size>_<output_type>

Example:

    xgboost_sample_input_mac_50_model.pkl

---

# Making Predictions

## LSTM inference

Use the LSTM inference script with a trained `.pt` model.

Example:

    python src/inference/lstm_inference.py --input_file molecules_for_prediction.csv --output_dir prediction_results --model_name lstm_sample_input_mac_50_model.pt

Input CSV should contain:

    smile

    CCO

    CCN

    CCC

If docking scores are available, they may also be included:

    smile,docking_score

    CCO,-5.6

    CCN,-6.1

---

## Classical ML inference

Use the ML inference script with a trained `.pkl` model.

Example:

    python src/inference/other_models_inference.py --input_file molecules_for_prediction.csv --output_dir prediction_results --model_name xgboost_sample_input_mac_50_model.pkl

---

# Configuration

Project paths are managed in:

    src/config/paths.py

Training settings are managed in:

    src/config/settings.py

Protein sequences are managed in:

    src/config/seq_dict.py

This avoids hard-coded paths like:

    ../../datasets

    ../../results

and makes the code easier to run from PyCharm or the terminal.

---

# Notes

- Use Python 3.12.

- Keep datasets inside the `datasets/` folder.

- For LSTM training, make sure the target exists in `seq_dict.py`.

- For classical ML models, create `.dat` feature files before training.

- Use `--cross_validate 5` for real cross-validation.

- `--cross_validate 1` behaves like normal training.

---

# Project Structure

<details>

<summary>Click to expand project structure</summary>

    swifty/

      datasets/

        sample_input.csv

        molecules_for_prediction.csv

      results/

        validation_metrics/

        testing_metrics/

        test_predictions/

        project_info/

        serialized_models/

        shap_analyses/

        tanimoto_results/

      results_seq/

        validation_metrics/

        testing_metrics/

        test_predictions/

        project_info/

        serialized_models/

        shap_analyses/

        tsne_analyses/

        training_testing_data/

      src/

        config/

          paths.py

          settings.py

          seq_dict.py

        core/

          lstm.py

          ml_models.py

          model.py

          data_generator.py

        features/

          create_fingerprint_data.py

          smiles_featurizers.py

          calculate_tanimoto.py

        train/

          main_lstm.py

          main_ml.py

          trainer.py

        inference/

          lstm_inference.py

          other_models_inference.py

        utils/

          utils.py

          swift_dock_logger.py

</details>

"""

path = Path("/mnt/data/README_compact.md")

path.write_text(readme, encoding="utf-8")

print(f"Created: {path}")
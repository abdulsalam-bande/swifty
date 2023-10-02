# Swift Dock 🚀

In this study, we explored various machine learning (ML) models to forecast docking scores of ligands for specific target proteins, aiming to reduce the need for extensive docking calculations. Our primary goal? Find a regression model that can determine the docking scores of ligands from a chemical library in relation to a target protein. We achieve this with data from explicit docking of a select few molecules.

Among the ML models:
- 🧠 An **LSTM-based Neural Network** (common in Natural Language Processing tasks like speech recognition). Combined with an attention mechanism, it effectively extracts ligand data. We used Pytorch for this.
- 🌳 Models like **XGBoost**, **Decision Tree Regression**, and **Stochastic Gradient Descent** from libraries like XGBoost and scikit-learn.

## Setting up the Environment 🛠️

1. Ensure Python 3.7 is installed 🐍
2. Create a virtual environment and execute `pip install -r requirements.txt` 📦
3. Navigate to 'swifty' and run `sudo chmod -R 777 logs` 📑


## Setting up the Environment - Apple Silicon 🍎

1. Ensure Python 3.8 is installed 🐍
2. Create a virtual environment and execute `pip -r apple-silcon-requirements.txt` 📦
3. Navigate to 'swifty' and run `sudo chmod -R 777 logs` 📑

## Training Using LSTM 🧠

### Build & Validate 🛠️

1. Add your target to the 'dataset' folder. Follow the format in `sample_input.csv`.
2. Example: Lets say you want to train the lstm model for sample_input for mac descriptor and a training set size of 50 without cross validation. First,  Navigate to `src/models` and run the below command. Note: All possible descriptors are mac, onehot, and morgan_onehot_mac:
#### Command
```bash
python main_lstm.py --input sample_input --descriptors mac --training_sizes 50 --cross_validation False 
```
#### Command Format
```bash
python main_lstm.py --input <YOUR_INPUT_FILE> --descriptors <DESCRIPTOR> --training_sizes <TRAINING_SIZE> --cross_validation <CROSS_VALIDATION> 
```

This will produce a result directory with 5 categories. Each file follows the format: lstm_target_descriptor_training_size.
- **project_info**: Details like training size and durations.
- **serialized_models**: Trained model post-training.
- **test_predictions**: Each docking score and corresponding model prediction.
- **testing_metrics**: Metrics such as R-squared, mean absolute error from testing.
- **validation_metrics**: Metrics from 5-fold cross-validation (only if `--cross_validation True`).

### More examples 
1. Training Using Multiple Descriptors
```bash
python main_lstm.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 --cross_validation False 
```

2. Training Using Multiple Descriptors and Multiple Training set sizes
```bash
python main_lstm.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100 --cross_validation False 
```

3. Training Using Multiple Descriptors, Multiple Training set sizes and Multiple Targets
```bash
python main_lstm.py --input sample_input sample_input_2 --descriptors mac morgan_onehot_mac --training_sizes 50 100 --cross_validation False 
```

## Making Predictions with LSTM 🎯
Run
```bash
python lstm_inference.py --input_file <YOUR_INPUT_FILE> --output_dir <YOUR_OUTPUT_DIRECTORY> --model_name <YOUR_MODEL_NAME>
```
Ensure than <YOUR_INPUT_FILE>  follows the format of molecules_for_prediction.csv in the 'dataset' folder.
Example
```bash
python lstm_inference.py --input_file molecules_for_prediction.csv --output_dir prediction_results --model_name lstm_target_mac_50_model.pt
```

## Training Using other models (from scikit-learn) 🌳
1. Add your target to the 'dataset' folder. It should match the format of sample_input.csv
2. Run this command to prepare the dataset
#### Example
```bash
python create_fingerprint_data.py --input sample_input --descriptors mac
```
#### Command Format
```bash
python create_fingerprint_data.py --input <YOUR_INPUT_FILE> --descriptors <DESCRIPTOR>
```
3. Run this to train
```bash
python main_ml.py --input sample_input --descriptors mac --training_sizes 50 --regressor sgreg
```
Command Format
```bash
python main_ml.py --input <YOUR_INPUT_FILE> --descriptors <DESCRIPTOR> --training_sizes  <TRAINING_SIZE> --regressor  <REGRESSOR>
```
Note: All possible descriptors are mac, morgan_onehot_mac and onehot. All possible regressors are  sgreg, xgboost and decision_tree

### More examples 
1. Training Using Multiple Descriptors
```bash
python main_ml.py --input sample_input --descriptors mac  morgan_onehot_mac --training_sizes 50 --regressor sgreg
```

2. Training Using Multiple Descriptors and Multiple Training set sizes
```bash
python main_ml.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100 --regressor sgreg
```

2. Training Using Multiple Descriptors, Multiple Training set sizes and Multiple  Models
```bash
python main_ml.py --input sample_input --descriptors mac morgan_onehot_mac --training_sizes 50 100 --regressor sgreg xgboost
```

This will give you a result directory with similar categories and file formats as mentioned in the LSTM section.

## Making Predictions with other Models 🎯
1. Your input CSV should match the format of molecules_for_prediction.csv in the 'dataset' folder.
2. Run
```bash
python other_models_inference.py --input_file <YOUR_INPUT_FILE> --output_dir <YOUR_OUTPUT_DIRECTORY> --model_name <YOUR_MODEL_NAME>
```
Ensure than <YOUR_INPUT_FILE>  follows the format of molecules_for_prediction.csv in the 'dataset' folder.



import numpy as np
import argparse
import os
from sklearn.linear_model import SGDRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from ml_models import OtherModels

training_metrics_dir = '../../results/validation_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
serialized_models_path = '../../results/serialized_models/'
shap_analyses_dir = '../../results/shap_analyses/'
dataset_dir = "../../datasets/"

os.makedirs(training_metrics_dir, exist_ok=True)
os.makedirs(testing_metrics_dir, exist_ok=True)
os.makedirs(test_predictions_dir, exist_ok=True)
os.makedirs(project_info_dir, exist_ok=True)
os.makedirs(serialized_models_path, exist_ok=True)
os.makedirs(shap_analyses_dir, exist_ok=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train_ml(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,shap_analyses_dir,
             all_data, train_size, test_size, val_size, identifier, number_of_folds, regressor, serialized_models_path,
             descriptor, data_csv):
    model = OtherModels(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,shap_analyses_dir,
                        all_data=all_data, train_size=train_size, test_size=test_size, val_size=val_size,
                        identifier=identifier, number_of_folds=number_of_folds, regressor=regressor,
                        serialized_models_path=serialized_models_path, descriptor=descriptor, data_csv=data_csv)

    model.split_data(cross_validate=args.cross_validate)
    model.train()

    if args.cross_validate:
        model.diagnose()
    model.test()
    model.shap_analyses()
    model.save_results()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="train code for fast docking",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="specify the target protein to ", nargs='+')
    parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
    parser.add_argument("--training_sizes", type=int, help="Training and cross validation size", nargs='+')
    parser.add_argument("--regressors", type=str, help="If to use 5 cross validation", nargs='+')
    parser.add_argument("--cross_validate", type=str2bool, help="If to use 5 cross validation")
    args = parser.parse_args()

    ##### Models  Arugments
    regressors_dict_ml_models = {
        'decision_tree': DecisionTreeRegressor,
        'xgboost': XGBRegressor,
        'sgdreg': SGDRegressor
    }

    dimensions_ml_models = {'onehot': 3500 + 1, 'morgan_onehot_mac': 4691 + 1,
                            'mac': 167 + 1}

    training_sizes_ml = args.training_sizes
    targets = args.input
    descriptors_dictionary_command_line = {}
    regressor_command_line = {}

    for desc in args.descriptors:
        if desc in dimensions_ml_models:
            descriptors_dictionary_command_line[desc] = dimensions_ml_models[desc]

    for rg in args.regressors:
        if rg in regressors_dict_ml_models:
            regressor_command_line[rg] = regressors_dict_ml_models[rg]

    number_of_folds = 5

    for target in targets:
        for regressor_id, regressor in regressor_command_line.items():
            for data_file, data_dim in descriptors_dictionary_command_line.items():
                for size in training_sizes_ml:
                    data_set_path = f"{dataset_dir}{target}_{data_file}.dat"
                    data_csv = f"{dataset_dir}{target}.csv"
                    identifier = f"{regressor_id}_{target}_{data_file}_{str(size)}"
                    data = np.memmap(data_set_path, dtype=np.float32)
                    target_length = data.shape[0] // data_dim
                    data = data.reshape((target_length, data_dim))
                    train_size = size
                    val_size = size * number_of_folds if args.cross_validate else 0
                    testing_size_ml = len(data) - (train_size + val_size)

                    train_ml(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             shap_analyses_dir=shap_analyses_dir,
                             all_data=data, train_size=size, test_size=testing_size_ml, val_size=val_size,
                             identifier=identifier,
                             number_of_folds=number_of_folds, regressor=regressor,
                             serialized_models_path=serialized_models_path, descriptor=data_file, data_csv=data_csv)

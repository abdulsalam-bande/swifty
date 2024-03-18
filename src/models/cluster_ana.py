import os
import warnings

from src.models.lstm import SwiftDock

all_targets = ['Drp1_GTPase', 'RyR2', 'Drp1_MiD49', 'ace', 'spike', 'nsp', 'nsp_sam', '5ht1b', 'fimh', 'adeR', 'adeS']
models = ['lstm']
descriptors = ['morgan_onehot_mac']
results_dir = '../../results/training_testing_data/'
serialized_models_dir = '../../results/serialized_models/'
output_path = '../../results/cluster_ana_results'
os.makedirs(output_path, exist_ok=True)

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    for target in all_targets:
        for model in models:
            for descriptor in descriptors:
                input_path = f"{results_dir}lstm_{target}_morgan_onehot_mac_7000_test_data.csv"
                model_path = f"{serialized_models_dir}lstm_{target}_morgan_onehot_mac_7000_model.pt"
                if os.path.exists(input_path) and os.path.exists(model_path):
                    SwiftDock.inference(input_path, output_path, model_path,cluster_analyses=True)

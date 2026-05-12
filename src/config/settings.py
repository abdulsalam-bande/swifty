import os


DEFAULT_CONFIG = {
    "batch_size": 2,
    "number_of_workers": 2,
    "learning_rate": 0.001,
    "number_of_epochs": 5,
    "number_of_folds": 5,
    "random_state": 42,
    "shap_sample_size": 100,
    "shap_test_size": 0.8,
    "shap_number_of_epochs": 1,
    "torch_num_threads": 6
}


FEATURE_CONFIG = {
    "batch_size": 128,
    "num_workers": min(8, os.cpu_count() or 1)
}


ML_CONFIG = {
    "number_of_folds": 5,
    "random_state": 42,
    "shap_sample_size": 50,
    "shap_test_size": 0.8
}
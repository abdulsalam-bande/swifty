from pathlib import Path


def find_project_root():
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "datasets").exists() and (parent / "src").exists():
            return parent

    raise FileNotFoundError("Could not find project root containing datasets/ and src/ folders.")


PROJECT_ROOT = find_project_root()

DATASET_DIR = PROJECT_ROOT / "datasets"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_SEQ_DIR = PROJECT_ROOT / "results_seq"

ML_PATHS = {
    "training_metrics_dir": RESULTS_DIR / "validation_metrics",
    "testing_metrics_dir": RESULTS_DIR / "testing_metrics",
    "test_predictions_dir": RESULTS_DIR / "test_predictions",
    "project_info_dir": RESULTS_DIR / "project_info",
    "serialized_models_path": RESULTS_DIR / "serialized_models",
    "shap_analyses_dir": RESULTS_DIR / "shap_analyses",
    "tanimoto_results_dir": RESULTS_DIR / "tanimoto_results"
}

LSTM_PATHS = {
    "training_metrics_dir": RESULTS_SEQ_DIR / "validation_metrics",
    "testing_metrics_dir": RESULTS_SEQ_DIR / "testing_metrics",
    "test_predictions_dir": RESULTS_SEQ_DIR / "test_predictions",
    "project_info_dir": RESULTS_SEQ_DIR / "project_info",
    "serialized_models_path": RESULTS_SEQ_DIR / "serialized_models",
    "shap_analyses_dir": RESULTS_SEQ_DIR / "shap_analyses",
    "tsne_analyses_dir": RESULTS_SEQ_DIR / "tsne_analyses",
    "training_and_testing_data": RESULTS_SEQ_DIR / "training_testing_data"
}


def create_directories(paths):
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
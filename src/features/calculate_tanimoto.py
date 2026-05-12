import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import DATASET_DIR, ML_PATHS
from config.settings import ML_CONFIG
from utils import TanimotoDataGenerator, save_dict, save_dict_with_one_index

mpl.rcParams["figure.dpi"] = 100
warnings.filterwarnings("ignore")

RESULT_DIR = ML_PATHS["tanimoto_results_dir"]
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_tanimoto(target_name, sample_ratio=0.35, batch_size=32, num_workers=1,
                       random_state=ML_CONFIG["random_state"]):
    target_dir = RESULT_DIR / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    data_path = DATASET_DIR / f"{target_name}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    original = pd.read_csv(data_path).dropna().reset_index(drop=True)

    sample_size = int(len(original) * sample_ratio)

    if sample_size == 0:
        raise ValueError(f"Sample size is 0 for {target_name}. Increase sample_ratio or use a larger dataset.")

    data = original.sample(sample_size, random_state=random_state).reset_index(drop=True)

    data_docking_scores = data["docking_score"].to_list()
    smiles_data = data["smile"].to_dict()

    data_information = {
        "size of original data": len(original),
        "size of sampled data": len(data),
        "sample ratio": sample_ratio
    }

    rdkit_info = {}
    counter = 0

    for _, smile in smiles_data.items():
        mol = Chem.MolFromSmiles(smile)

        if mol is None:
            continue

        rdkit_info[counter] = [smile, Chem.RDKFingerprint(mol)]
        counter += 1

    smiles_data_train = TanimotoDataGenerator(rdkit_info)
    train_dataloader = DataLoader(smiles_data_train, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    start_time = time.time()

    all_distances = []
    all_maxes = []
    all_mins = []

    for sample_batched in tqdm(train_dataloader):
        all_distances.extend(sample_batched["avg"].tolist())
        all_maxes.extend(sample_batched["max"].tolist())
        all_mins.extend(sample_batched["min"].tolist())

    elapsed_minutes = (time.time() - start_time) / 60
    data_information["time_minutes"] = elapsed_minutes

    result_dict = {
        "sampled_data_docking_scores": data_docking_scores,
        "avg_distances": all_distances,
        "max_distances": all_maxes,
        "min_distances": all_mins
    }

    result_dir_plots = target_dir / "tanimoto_calculation_information.png"
    result_dir_distances = target_dir / "all_distances.csv"
    result_dir_extra_information = target_dir / "tanimoto_calculation_information.csv"

    save_dict(result_dict, result_dir_distances)
    save_dict_with_one_index(data_information, result_dir_extra_information)

    fig, axes = plt.subplots(1, 5, figsize=(40, 8))
    ax1, ax2, ax3, ax4, ax5 = axes

    ax1.hist(original["docking_score"])
    ax1.set_title("Original Data", fontsize=30)
    ax1.set_xlabel("Docking Score", fontsize=25)
    ax1.set_ylabel("Frequency", fontsize=25)
    ax1.tick_params(axis="x", labelsize=20)
    ax1.tick_params(axis="y", labelsize=20)

    ax2.hist(data["docking_score"])
    ax2.set_title("Sampled Data", fontsize=30)
    ax2.set_xlabel("Docking Score", fontsize=25)
    ax2.set_ylabel("Frequency", fontsize=25)
    ax2.tick_params(axis="x", labelsize=20)
    ax2.tick_params(axis="y", labelsize=20)

    ax3.hist(all_distances)
    ax3.set_title("Average Distance", fontsize=30)
    ax3.set_xlabel("Average Tanimoto Distance", fontsize=25)
    ax3.set_ylabel("Frequency", fontsize=25)
    ax3.tick_params(axis="x", labelsize=20)
    ax3.tick_params(axis="y", labelsize=20)

    ax4.hist(all_maxes)
    ax4.set_title("Maximum Distance", fontsize=30)
    ax4.set_xlabel("Maximum Tanimoto Distance", fontsize=25)
    ax4.set_ylabel("Frequency", fontsize=25)
    ax4.tick_params(axis="x", labelsize=20)
    ax4.tick_params(axis="y", labelsize=20)

    ax5.hist(all_mins)
    ax5.set_title("Minimum Distance", fontsize=30)
    ax5.set_xlabel("Minimum Tanimoto Distance", fontsize=25)
    ax5.set_ylabel("Frequency", fontsize=25)
    ax5.tick_params(axis="x", labelsize=20)
    ax5.tick_params(axis="y", labelsize=20)

    fig.tight_layout()
    fig.savefig(result_dir_plots, facecolor="w")
    plt.close(fig)


if __name__ == "__main__":
    targets = [
        "Drp1_GTPase",
        "Drp1_MiD49",
        "RyR2",
        "nsp",
        "nsp_sam",
        "spike",
        "ace"
    ]

    for target in targets:
        calculate_tanimoto(target)
import warnings

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans

from config.paths import DATASET_DIR

warnings.filterwarnings("ignore")

N_CLUSTERS = 20
INPUT_SUFFIX = "sample_5k.csv"
OUTPUT_FILE_NAME = "sample_5k_cluster.csv"


def smiles_to_fingerprint(smile, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smile)

    if mol is None:
        return None

    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def add_clusters_to_file(input_path, output_path, n_clusters=N_CLUSTERS):
    df = pd.read_csv(input_path)

    if "smile" not in df.columns:
        raise ValueError(f"{input_path} must contain a 'smile' column.")

    fingerprints = df["smile"].apply(smiles_to_fingerprint).dropna()

    if len(fingerprints) < n_clusters:
        raise ValueError(f"Number of valid fingerprints is smaller than n_clusters. "
                         f"valid_fingerprints={len(fingerprints)}, n_clusters={n_clusters}")

    fp_array = np.array([list(fp) for fp in fingerprints])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(fp_array)

    df.loc[fingerprints.index, "cluster"] = labels
    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)


def cluster_sample_files():
    for input_path in DATASET_DIR.iterdir():
        if input_path.name.endswith(INPUT_SUFFIX):
            output_path = DATASET_DIR / OUTPUT_FILE_NAME
            add_clusters_to_file(input_path, output_path)


if __name__ == "__main__":
    cluster_sample_files()
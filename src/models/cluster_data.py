import os
import pandas as pd
from sklearn.cluster import KMeans
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


origin_dir = '../../datasets'
target_dir = '../../datasets'
n_clusters = 20 # Change this to the desired number of clusters

# Define a function to convert SMILES to fingerprints
def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return None

# Iterate over each file in origin_dir
for filename in os.listdir(origin_dir):
    if filename.endswith('sample_5k.csv'):  # Check for CSV files
        data_path = os.path.join(origin_dir, filename)
        df = pd.read_csv(data_path)
    
        fingerprints = df['smile'].apply(smiles_to_fingerprint)
        fingerprints = fingerprints.dropna()
        fp_array = np.array([list(fp) for fp in fingerprints])
    
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(fp_array)
        df.loc[fingerprints.index, 'cluster'] = kmeans.labels_
        df = df.reset_index(drop=True)

        # Save the dataframe to the target_dir with the same file name
        new_file_name = "sample_5k_cluster.csv"
        save_path = os.path.join(target_dir, new_file_name)
        df.to_csv(f'{save_path}', index=False)

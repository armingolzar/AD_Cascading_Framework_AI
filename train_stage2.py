import os 
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, Dataloader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from model.stage2_biomarkers import Stage2_Encoder, evaluate2

class ClinicalDataPayload(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)
    
    def __get_item__(self, idx):
        return {"features" : self.X[idx], "label" : self.y[idx]}
    
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running Stage 2 execution engine on device: {device}\n")

    data_path = "data/stage2_synthetic_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"[-] missing training dataset at: {data_path}")
    
    df = pd.read_csv(data_path)

    stage1_features = ['age', 'sex', 'education_years', 'mmse', 'moca', 'faq', 'cdr_sb']
    stage2_features = ['csf_amyloid_beta', 'csf_ptau', 'plasma_ptau217', 'plasma_nfl', 'apoe4_alleles']
    fused_features = stage1_features + stage2_features

    X = df[fused_features].values
    y = df["diagnostic_class"].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    os.makedirs("model", exist_ok=True)
    with open("model/stage2_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    
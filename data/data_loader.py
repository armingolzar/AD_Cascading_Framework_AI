import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class Stage1_Dataset(Dataset):
    def __init__(self, dataframe, scaler_meta=None):
        self.df = dataframe.reset_index(drop=True)
        self.feature_cols = ["age", "sex", "education_years", "mmse", "moca", "faq", "cdr_sb"]

        self.X_raw = self.df[self.feature_cols].values.astype(np.float32)
        self.y_raw = self.df["diagnostic_class"].values.astype(np.int64)

        if scaler_meta is None:
            self.means = self.X_raw.mean(axis=0)
            self.stds = self.X_raw.std(axis=0)
            self.stds[self.stds == 0] = 1.0
        else:
            self.means = scaler_meta["means"]
            self.stds = scaler_meta["stds"]

        self.X_scaled = (self.X_raw - self.means) / self.stds

    def get_scaler_meta(self):
        return {"means" : self.means, "stds" : self.stds}
    
    
    def save_scaler_meta(self, filepath="model/scaler_meta.npy"):
        """Saves the calculated training means and stds to disk."""
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)
            
        np.save(filepath, self.get_scaler_meta())
        print(f"[+] Scaler parameters saved to disk: {filepath}")
        

    @staticmethod
    def load_scaler_meta(filepath="model/scaler_meta.npy"):
        """Utility function to load parameters back into memory from disk."""
        return np.load(filepath, allow_pickle=True).item()
    

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {"features" : torch.tensor(self.X_scaled[idx], dtype=torch.float32),
                "label" : torch.tensor(self.y_raw[idx], dtype=torch.long)}
                
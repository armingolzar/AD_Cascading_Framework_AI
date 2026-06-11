import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data.data_loader import Stage1_Dataset

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    df = pd.read_csv("./data/stage1_synthetic_data.csv")
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['diagnostic_class'])

    train_dataset = Stage1_Dataset(train_df)
    test_dataset = Stage1_Dataset(test_df, scaler_meta=train_dataset.get_scaler_meta())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    

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
from model.stage1_screening import Stage1_Encoder

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    df = pd.read_csv("./data/stage1_synthetic_data.csv")
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['diagnostic_class'])

    train_dataset = Stage1_Dataset(train_df)
    test_dataset = Stage1_Dataset(test_df, scaler_meta=train_dataset.get_scaler_meta())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Stage1_Encoder(input_dim=7, embedding_dim=16, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    best_val_loss = float("inf")
    epochs = 40

    

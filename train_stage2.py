import os 
import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    
    def __getitem__(self, idx):
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

    train_set = ClinicalDataPayload(X_train_scaled, y_train)
    val_set = ClinicalDataPayload(X_val_scaled, y_val)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = Stage2_Encoder(input_dim=len(fused_features), hidden_dim=64, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    metrics_history = {"train_loss" : [], "val_loss" : []}
    epochs = 40
    optimal_loss = float("inf")

    print("Beginning Stage 2 Gradient Updates...")
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * features.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        epoch_val_loss, val_labels, val_preds = evaluate2(model, val_loader, criterion, device)

        metrics_history["train_loss"].append(epoch_train_loss)
        metrics_history["val_loss"].append(epoch_val_loss)

        save_indicator = ""
        if epoch_val_loss < optimal_loss:
            optimal_loss = epoch_val_loss
            torch.save(model.state_dict(), "model/stage2_model.pth")
            save_indicator = "--> Optimal checkpoint saved"

        if ((epoch + 1) % 5 == 0) or (epoch == 0) or ("saved" in save_indicator):
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} {save_indicator}")

    print("\nTraining complete. Running metrics audit with optimal weights...")
    model.load_state_dict(torch.load("model/stage2_model.pth"))
    _, evaluation_labels, evaluation_predictions = evaluate2(model, val_loader, criterion, device)

    # Plotting execution metrics
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), metrics_history["train_loss"], label="Train-Loss", color="royalblue")
    plt.plot(range(1, epochs+1), metrics_history["val_loss"], label="Val-Loss", color="crimson", linestyle="--")
    plt.title('Stage 2 Optimization Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model/stage2_training_plots.png", dpi=150)
    print("[+] Saved execution dynamics graphics to models/stage2_training_plots.png")

    print("\n" + "="*50)
    print("STAGE 2 AUDIT REPORT")
    print("="*50)
    print(classification_report(evaluation_labels, evaluation_predictions, target_names=["CN", "MCI", "AD"], digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(evaluation_labels, evaluation_predictions))

if __name__ == "__main__":
    main()
      
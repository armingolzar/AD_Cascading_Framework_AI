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
from model.stage1_screening import Stage1_Encoder, evaluate
import matplotlib.pyplot as plt

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing pipeline on device: {device}")

    df = pd.read_csv("./data/stage1_synthetic_data.csv")
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['diagnostic_class'])

    train_dataset = Stage1_Dataset(train_df)
    train_dataset.save_scaler_meta()
    test_dataset = Stage1_Dataset(test_df, scaler_meta=train_dataset.get_scaler_meta())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = Stage1_Encoder(input_dim=7, embedding_dim=16, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    metrics_history = {"train_loss" : [], "val_loss" : []}
    best_val_loss = float("inf")
    epochs = 40

    print("\nBeginning Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss, val_labels, val_preds = evaluate(model, test_loader, criterion, device)

        val_acc = np.mean(val_preds == val_labels)

        metrics_history["train_loss"].append(epoch_train_loss)
        metrics_history["val_loss"].append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "model/stage1_model.pth")
            save_msg = "--> Checkpoint Saved"
        else:
            save_msg = ""

        if ((epoch + 1) % 5 == 0) or (epoch == 0) or ("Saved" in save_msg):
            print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | {epoch_val_loss:.4f} | {val_acc*100:.1f}% {save_msg}")
        
    print("\nTraining complete. Running final evaluation metrics using optimal checkpoint...")
    model.load_state_dict(torch.load("model/stage1_model.pth"))
    _, final_labels, final_preds = evaluate(model, test_loader, criterion, device)


    # Plotting execution metrics
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), metrics_history["train_loss"], label="Train-Loss", color="royalblue")
    plt.plot(range(1, epochs+1), metrics_history["val_loss"], label="Val-Loss", color="crimson", linestyle="--")
    plt.title('Stage 1 Optimization Convergence')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("model/stage1_training_plots.png", dpi=150)
    print("[+] Saved execution dynamics graphics to models/stage1_training_plots.png")


    print("\n" + "="*50)
    print("STAGE 1 CLINICAL EVALUATION METRICS REPORT")
    print("="*50)
    print(classification_report(final_labels, final_preds, target_names=["CN (Normal)", "MCI (Mild)", "AD (Alzheimer\'s)"]))

    print("Confusion Matrix Layout:")
    print(confusion_matrix(final_labels, final_preds))

if __name__ == "__main__":
    main()



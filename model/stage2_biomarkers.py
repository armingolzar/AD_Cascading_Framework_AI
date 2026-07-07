import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Stage2_Encoder(nn.module):

    def __init__(self, input_dim=12, hidden_dim=64, num_classes=3):
        super(Stage2_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNrom1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.drop_prob = 0.2

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_prob, training=self.training)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_prob, training=self.training)

        logits = self.classifier(x)
        return logits
    
def evaluate2(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad:
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        total_loss = running_loss / len(dataloader.dataset)
        return total_loss, np.array(all_labels), np.array(all_preds)
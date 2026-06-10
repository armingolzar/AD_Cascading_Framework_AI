import torch 
import torch.nn as nn
import torch.nn.functional as F

class Stage1_Encoder(nn.Module):
    def __init__(self, input_dim=7, embedding_dim=16, num_classes=3):
        super(Stage1_Encoder, self).__init__()

        self.fc1 = nn.Linear(7, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.drop_prob = 0.2

    def forward(self, x, return_embedding=False):

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = F.dropout(x, p=self.drop_prob, training=self.training)

        embedding = self.fc2(x)
        embedding = self.bn2(embedding)
        embedding = F.relu(embedding)

        if return_embedding:
            return embedding

        logits = self.classifier(embedding)
        return embedding


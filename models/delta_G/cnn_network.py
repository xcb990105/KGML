import torch
import torch.nn as nn
from torchvision import models

class CNNResnet50(nn.Module):
    def __init__(self, hidden_channels, out_channels, metal_feature_dim=8, metal_embed_dim=128):
        super(CNNResnet50, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Node embedding layers
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Linear(in_features=512, out_features=hidden_channels)

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, metal_embed_dim)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels + metal_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, metal_features):

        cnn_out = self.cnn(data)

        metal_embed = self.metal_fc(metal_features)

        combined = torch.cat([cnn_out, metal_embed], dim=1)

        energy = self.regressor(combined).squeeze(-1)

        return energy

class CNNVgg16(nn.Module):
    def __init__(self, hidden_channels, out_channels, metal_feature_dim=8, metal_embed_dim=128):
        super(CNNVgg16, self).__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        # Node embedding layers
        self.cnn = models.vgg16(pretrained=True)
        self.cnn.classifier = torch.nn.Identity()
        self.cnn_fc = torch.nn.Linear(25088, hidden_channels)

        self.metal_fc = nn.Sequential(
            nn.Linear(metal_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, metal_embed_dim)
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels + metal_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # Fully connected layer for output
        self.fc_out = nn.Linear(hidden_channels, out_channels)

    def forward(self, data, metal_features):

        cnn_out = self.cnn(data)

        cnn_out = self.cnn_fc(cnn_out)

        metal_embed = self.metal_fc(metal_features)

        combined = torch.cat([cnn_out, metal_embed], dim=1)

        energy = self.regressor(combined).squeeze(-1)

        return energy

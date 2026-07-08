import torch.nn as nn
from torchvision import models


class DocumentClassifier(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Fine-tune ALL layers
        for param in self.model.parameters():
            param.requires_grad = True

        # Replace classifier
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)
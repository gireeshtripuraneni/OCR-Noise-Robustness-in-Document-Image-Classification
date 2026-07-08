import torch

from src.model import DocumentClassifier

model = DocumentClassifier(num_classes=16)

print(model)

dummy = torch.randn(1, 3, 224, 224)

output = model(dummy)

print("\nOutput shape:", output.shape)
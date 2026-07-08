import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from src.dataset_loader import RVLCDIPDataset
from src.transforms import train_transform, test_transform
from src.model import DocumentClassifier

# ==============================
# Configuration
# ==============================

DATASET_PATH = "data"

BATCH_SIZE = 32
TRAIN_RATIO = 0.8
EPOCHS = 25              # full text
LEARNING_RATE = 1e-5

# Use a small subset for quick testing
USE_SUBSET = False
SUBSET_SIZE = 4000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#EPOCHS = 10
print(f"Using Device: {device}")

# ==============================
# Load Dataset
# ==============================

train_dataset_full = RVLCDIPDataset(
    root_dir=DATASET_PATH,
    transform=train_transform
)

val_dataset_full = RVLCDIPDataset(
    root_dir=DATASET_PATH,
    transform=test_transform
)

print(f"Total Images : {len(train_dataset_full)}")
print(f"Classes      : {len(train_dataset_full.classes)}")
num_classes = len(train_dataset_full.classes)
# ==============================
# Optional Subset (for quick testing)
# ==============================

if USE_SUBSET:

    indices = torch.randperm(len(train_dataset_full))[:SUBSET_SIZE]

    train_dataset_full = Subset(train_dataset_full, indices)
    val_dataset_full = Subset(val_dataset_full, indices)

    print(f"\nUsing subset of {SUBSET_SIZE} images")

# ==============================
# Train / Validation Split
# ==============================

train_size = int(TRAIN_RATIO * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size

generator = torch.Generator().manual_seed(42)

train_dataset, _ = random_split(
    train_dataset_full,
    [train_size, val_size],
    generator=generator
)

_, val_dataset = random_split(
    val_dataset_full,
    [train_size, val_size],
    generator=generator
)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")

# ==============================
# DataLoaders
# ==============================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nDataLoaders Created Successfully!")

# ==============================
# Model
# ==============================

model = DocumentClassifier(
    num_classes=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# ==============================
# Training
# ==============================

best_accuracy = 0.0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

os.makedirs("models", exist_ok=True)

for epoch in range(EPOCHS):

    # --------------------------
    # Train
    # --------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels, _ in tqdm(
    train_loader,
    desc=f"Training Epoch {epoch+1}/{EPOCHS}",
    leave=False
):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total

    # --------------------------
    # Validation
    # --------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels, _ in tqdm(
    val_loader,
    desc="Validation",
    leave=False
):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print("-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss : {train_loss/len(train_loader):.4f}")
    print(f"Train Acc  : {train_accuracy:.2f}%")
    print(f"Val Loss   : {val_loss/len(val_loader):.4f}")
    print(f"Val Acc    : {val_accuracy:.2f}%")

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        torch.save({
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "best_accuracy": val_accuracy
}, "models/document_classifier.pth")

        print(f"Best model saved! Validation Accuracy = {val_accuracy:.2f}%")
        
    scheduler.step(val_accuracy)
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

print("\nTraining Finished!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
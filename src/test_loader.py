from src.dataset_loader import RVLCDIPDataset

dataset = RVLCDIPDataset("data")

print("Total Images:", len(dataset))
print("Number of Classes:", len(dataset.classes))
print("\nClasses:")

for cls in dataset.classes:
    print("-", cls)
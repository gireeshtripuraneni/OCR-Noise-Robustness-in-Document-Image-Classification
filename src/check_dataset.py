import os

DATASET_PATH = "data"

classes = sorted([
    c for c in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, c))
])

print("=" * 50)
print(f"Number of classes: {len(classes)}")
print("=" * 50)

total = 0

for cls in classes:
    folder = os.path.join(DATASET_PATH, cls)

    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".tif", ".tiff"))
    ])

    total += count
    print(f"{cls:<25} {count}")

print("=" * 50)
print(f"Total Images: {total}")
print("=" * 50)
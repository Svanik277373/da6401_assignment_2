import os
from collections import Counter
from data.pets_dataset import OxfordIIITPetDataset

def check_class_split(data_root):
    print(f"Reading dataset index from {data_root}...")
    
    # Initialize datasets (this parses the list.txt but doesn't load images)
    train_ds = OxfordIIITPetDataset(root=data_root, split="train")
    val_ds = OxfordIIITPetDataset(root=data_root, split="val")

    # Extract labels directly from the PetSample dataclass objects
    train_labels = [sample.breed_label for sample in train_ds.samples]
    val_labels = [sample.breed_label for sample in val_ds.samples]

    # Count occurrences
    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print(f"\nTotal Training Samples: {len(train_labels)}")
    print(f"Total Validation Samples: {len(val_labels)}")

    print("\n--- Training Set Class Distribution ---")
    print("Class ID | Num Samples | Percentage")
    print("-" * 35)
    for cls_id in sorted(train_counts.keys()):
        count = train_counts[cls_id]
        pct = (count / len(train_labels)) * 100
        print(f"Class {cls_id:<2}  | {count:<11} | {pct:.2f}%")

    print("\n--- Validation Set Class Distribution ---")
    print("Class ID | Num Samples | Percentage")
    print("-" * 35)
    for cls_id in sorted(val_counts.keys()):
        count = val_counts[cls_id]
        pct = (count / len(val_labels)) * 100
        print(f"Class {cls_id:<2}  | {count:<11} | {pct:.2f}%")

    # Check for severe imbalance
    min_train = min(train_counts.values())
    max_train = max(train_counts.values())
    print("\n--- Summary ---")
    print(f"Train Min Samples/Class: {min_train}")
    print(f"Train Max Samples/Class: {max_train}")
    if max_train / min_train > 1.5:
        print("Warning: Notable class imbalance detected.")
    else:
        print("Classes are relatively well-balanced.")

if __name__ == "__main__":
    # Use your specific dataset path
    check_class_split(r"oxford-iiit-pet")
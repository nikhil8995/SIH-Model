#!/usr/bin/env python3
"""
Script to fix GRAFFITI label files - change class 1 to class 0
"""

import os
import glob


def fix_graffiti_labels(dataset_dir):
    """Fix GRAFFITI label files by changing class ID from 1 to 0"""

    # Paths to train and val label directories
    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    val_labels_dir = os.path.join(dataset_dir, "val", "labels")

    total_fixed = 0

    for split_name, labels_dir in [("train", train_labels_dir), ("val", val_labels_dir)]:
        if not os.path.exists(labels_dir):
            print(f"Skipping {split_name} - directory not found: {labels_dir}")
            continue

        # Find all GRAFFITI label files
        graffiti_labels = glob.glob(os.path.join(labels_dir, "GRAFFITI_*.txt"))
        print(
            f"\nProcessing {len(graffiti_labels)} GRAFFITI files in {split_name}...")

        for label_file in graffiti_labels:
            try:
                # Read current content
                with open(label_file, 'r') as f:
                    lines = f.readlines()

                # Process each line
                modified_lines = []
                changes_made = False

                for line in lines:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) >= 5:  # Valid YOLO format
                            class_id = parts[0]

                            # Change class 1 to class 0 (GRAFFITI should be class 0)
                            if class_id == '1':
                                parts[0] = '0'
                                changes_made = True

                            modified_lines.append(' '.join(parts) + '\n')
                        else:
                            # Keep malformed lines as-is
                            modified_lines.append(line + '\n')

                # Write back if changes were made
                if changes_made:
                    with open(label_file, 'w') as f:
                        f.writelines(modified_lines)
                    total_fixed += 1
                    print(f"  Fixed: {os.path.basename(label_file)}")
                else:
                    print(
                        f"  No changes needed: {os.path.basename(label_file)}")

            except Exception as e:
                print(f"  Error processing {label_file}: {e}")

    print(f"\nTotal files fixed: {total_fixed}")
    return total_fixed


def verify_fix(dataset_dir):
    """Verify that the fix worked by checking some GRAFFITI label files"""

    train_labels_dir = os.path.join(dataset_dir, "train", "labels")
    graffiti_labels = glob.glob(os.path.join(
        train_labels_dir, "GRAFFITI_*.txt"))[:3]

    print(f"\nVerification - checking first 3 GRAFFITI label files:")
    for label_file in graffiti_labels:
        try:
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                class_id = first_line.split()[0] if first_line else "empty"
                print(f"  {os.path.basename(label_file)}: class {class_id}")
        except Exception as e:
            print(f"  {os.path.basename(label_file)}: Error - {e}")


if __name__ == "__main__":
    # Dataset directory
    dataset_dir = "/home/nikhil/Documents/python/projs/SIH/archive/dataset/00 Test"

    print("Fixing GRAFFITI label files...")
    print(f"Dataset directory: {dataset_dir}")

    # Fix the labels
    fixed_count = fix_graffiti_labels(dataset_dir)

    # Verify the fix
    verify_fix(dataset_dir)

    if fixed_count > 0:
        print(f"\nSuccessfully fixed {fixed_count} GRAFFITI label files!")
        print("Next steps:")
        print("1. cd /home/nikhil/Documents/dev/python/projs/SIH/ml")
        print("2. rm -rf artifacts/")
        print("3. PYTHONPATH=.. python -m ml.train_probe")
    else:
        print("\nNo files were modified. Check if GRAFFITI files actually use class 1.")

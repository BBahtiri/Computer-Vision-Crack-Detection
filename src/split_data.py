"""
Data splitting utility for organizing datasets.
This script splits images and labels into train/validation/test sets.
"""

import os
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional, Dict
import argparse
import random # For random seed setting

class DatasetSplitter:
    """Handle dataset splitting into train/validation/test sets."""
    
    def __init__(self, image_folder: str, label_folder: str, 
                 output_dir: str = ".", random_seed: int = 42):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.output_dir = output_dir
        self.random_seed = random_seed
        
        # MODIFIED: Set random seed for reproducibility for all random operations in this class
        random.seed(self.random_seed)
        np.random.seed(self.random_seed) # If numpy random operations are used

        self._validate_directories()
    
    def _validate_directories(self):
        if not os.path.isdir(self.image_folder): # MODIFIED: Check if it's a directory
            raise ValueError(f"Image folder not found or not a directory: {self.image_folder}")
        if not os.path.isdir(self.label_folder): # MODIFIED: Check if it's a directory
            raise ValueError(f"Label folder not found or not a directory: {self.label_folder}")
    
    def _get_matching_files(self) -> Tuple[List[str], List[str]]:
        """
        Get lists of image and label files, ensuring they match.
        MODIFIED: More robust file matching based on basenames.
        """
        image_files_all = sorted([f for f in os.listdir(self.image_folder) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        label_files_all = sorted([f for f in os.listdir(self.label_folder) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Create dictionaries for quick lookup by base filename (without extension)
        # This assumes image and label have the same base name (e.g., img1.png, img1.png)
        # If there's a pattern like img1_image.png and img1_mask.png, this needs adjustment.
        # For now, assuming identical basenames in different folders.
        
        # Example: img_001_frame_0.png and mask_001_frame_last.png
        # The original code implies a fixed transformation like "frame_0.png" -> "frame_last.png"
        # If that's the case, we should enforce that relationship.

        matched_images = []
        matched_labels = []

        # MODIFIED: Let's assume the mask filename can be derived from image filename
        # using the specific transformation rule from other scripts.
        # This makes matching explicit based on the transformation.
        for img_file in image_files_all:
            # This is the transformation rule seen in other scripts.
            # It needs to be consistent for your dataset.
            expected_mask_file = img_file.replace("frame_0.png", "frame_last.png") 
            
            if expected_mask_file in label_files_all:
                matched_images.append(img_file)
                matched_labels.append(expected_mask_file)
            else:
                print(f"Warning: No matching label found for image '{img_file}' (expected '{expected_mask_file}' in label folder). Skipping this image.")

        if not matched_images:
            raise ValueError("No matching image-label pairs found. Check file naming and paths.")

        if len(matched_images) != len(matched_labels):
            # This should not happen if the above logic is correct
            raise RuntimeError("Internal error: Mismatched count of images and labels after matching.")
        
        print(f"Found {len(matched_images)} matched image-label pairs.")
        return matched_images, matched_labels
    
    def split_dataset(self, val_percent: float = 0.2, test_percent: float = 0.1,
                     stratify_by_labels: Optional[List] = None) -> Dict[str, Dict[str, List[str]]]:
        # MODIFIED: Renamed stratify to stratify_by_labels for clarity
        if not (0 < val_percent < 1 and 0 < test_percent < 1 and (val_percent + test_percent) < 1):
            raise ValueError("val_percent and test_percent must be between 0 and 1, and their sum < 1.")
        
        images, labels = self._get_matching_files() # These are now guaranteed to correspond
        
        # Ensure indices are consistent for splitting
        indices = np.arange(len(images))

        # For stratification, stratify_by_labels should be an array/list of the same length as images/labels
        # that contains the labels to stratify on (e.g., class indicators for each sample).
        # If stratify_by_labels is provided, it must correspond to the 'images' list.
        stratify_array = None
        if stratify_by_labels is not None:
            if len(stratify_by_labels) != len(images):
                raise ValueError("Length of stratify_by_labels must match the number of image samples.")
            stratify_array = np.array(stratify_by_labels)

        # First split: separate test set
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=test_percent,
            random_state=self.random_seed,
            stratify=stratify_array[indices] if stratify_array is not None else None
        )
        
        # Prepare remaining data and stratification labels for the second split
        remaining_images = [images[i] for i in train_val_indices]
        remaining_labels = [labels[i] for i in train_val_indices]
        remaining_indices_for_split = np.arange(len(remaining_images)) # New indices for this subset

        stratify_for_train_val = None
        if stratify_array is not None:
            stratify_for_train_val = stratify_array[train_val_indices]

        # Second split: separate validation from training
        # Adjust val_percent for the remaining data
        val_size_adjusted = val_percent / (1.0 - test_percent)
        if val_size_adjusted >= 1.0: # Safety check
            val_size_adjusted = 0.5 # Default if percentages are too high

        train_indices_local, val_indices_local = train_test_split(
            remaining_indices_for_split, # Use local indices
            test_size=val_size_adjusted,
            random_state=self.random_seed, # Use the same seed for consistency in sequential splits
            stratify=stratify_for_train_val if stratify_for_train_val is not None else None
        )

        # Map local indices back to original image/label lists if needed, or just use the split lists
        train_images = [remaining_images[i] for i in train_indices_local]
        train_labels = [remaining_labels[i] for i in train_indices_local]
        val_images = [remaining_images[i] for i in val_indices_local]
        val_labels = [remaining_labels[i] for i in val_indices_local]
        
        test_images = [images[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        split_info = {
            'train': {'images': train_images, 'labels': train_labels},
            'val': {'images': val_images, 'labels': val_labels},
            'test': {'images': test_images, 'labels': test_labels}
        }
        
        print("\nDataset split complete:")
        print(f"  Training:   {len(train_images)} samples")
        print(f"  Validation: {len(val_images)} samples")
        print(f"  Testing:    {len(test_images)} samples")
        return split_info
    
    def save_split_datasets(self, split_info: Dict[str, Dict[str, List[str]]], copy_files: bool = True):
        print("\nSaving split datasets...")
        for split_name, data in split_info.items():
            image_dir = os.path.join(self.output_dir, f'{split_name}_images')
            label_dir = os.path.join(self.output_dir, f'{split_name}_labels')
            os.makedirs(image_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            print(f"\nProcessing {split_name} set...")
            file_operation = shutil.copy2 if copy_files else shutil.move
            
            for img_file in data['images']:
                src = os.path.join(self.image_folder, img_file)
                dst = os.path.join(image_dir, img_file)
                if os.path.exists(src): file_operation(src, dst)
                else: print(f"Warning: Source image file not found during save: {src}")
            
            for lbl_file in data['labels']:
                src = os.path.join(self.label_folder, lbl_file)
                dst = os.path.join(label_dir, lbl_file) # MODIFIED: Use original label filename for destination
                if os.path.exists(src): file_operation(src, dst)
                else: print(f"Warning: Source label file not found during save: {src}")
            
            print(f"  Saved {len(data['images'])} images to {image_dir}")
            print(f"  Saved {len(data['labels'])} labels to {label_dir}")
    
    def create_split_summary(self, split_info: Dict[str, Dict[str, List[str]]], filename: str = "split_summary.txt"):
        # (Implementation identical to previous corrected version)
        summary_path = os.path.join(self.output_dir, filename)
        with open(summary_path, 'w') as f:
            f.write("Dataset Split Summary\n"); f.write("=" * 50 + "\n\n")
            f.write(f"Source image folder: {self.image_folder}\n")
            f.write(f"Source label folder: {self.label_folder}\n")
            f.write(f"Output directory: {self.output_dir}\n")
            f.write(f"Random seed: {self.random_seed}\n\n")
            f.write("Split Statistics:\n"); f.write("-" * 30 + "\n")
            total = sum(len(data_set['images']) for data_set in split_info.values())
            if total == 0: total = 1 # Avoid division by zero if no files
            for split_name, data_set in split_info.items():
                count = len(data_set['images'])
                percentage = (count / total) * 100
                f.write(f"{split_name.capitalize():<12} {count:>5d} ({percentage:5.1f}%)\n")
            f.write("-" * 30 + "\n"); f.write(f"{'Total':<12} {total:>5d} (100.0%)\n")
        print(f"\nSplit summary saved to: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Split image dataset into train/validation/test sets")
    parser.add_argument("image_folder", help="Path to folder containing images")
    parser.add_argument("label_folder", help="Path to folder containing labels")
    parser.add_argument("--output-dir", default=".", help="Output directory for split datasets (default: current directory)")
    parser.add_argument("--val-percent", type=float, default=0.2, help="Validation set percentage (0-1, default: 0.2)")
    parser.add_argument("--test-percent", type=float, default=0.1, help="Test set percentage (0-1, default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    # MODIFIED: Added option for stratification labels file (e.g., a CSV with one class label per image)
    parser.add_argument("--stratify-labels-file", type=str, default=None, help="Path to a file containing labels for stratification (one label per line, matching image order)")
    args = parser.parse_args()

    stratify_data = None
    if args.stratify_labels_file:
        if not os.path.exists(args.stratify_labels_file):
            print(f"Warning: Stratification labels file not found: {args.stratify_labels_file}. Proceeding without stratification.")
        else:
            with open(args.stratify_labels_file, 'r') as f:
                stratify_data = [line.strip() for line in f.readlines()]
            print(f"Using stratification based on labels from: {args.stratify_labels_file}")
    
    try:
        splitter = DatasetSplitter(args.image_folder, args.label_folder, args.output_dir, args.seed)
        split_info = splitter.split_dataset(
            val_percent=args.val_percent,
            test_percent=args.test_percent,
            stratify_by_labels=stratify_data # Pass the loaded stratification labels
        )
        splitter.save_split_datasets(split_info, copy_files=not args.move)
        splitter.create_split_summary(split_info)
        print("\nDataset splitting complete!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage function (remains for testing)
def example_usage():
    # This example assumes specific local paths and might not run directly without adjustment.
    # It also assumes the specific filename transformation ("frame_0.png" -> "frame_last.png") is correct.
    # For stratification, you would need to create a file with labels corresponding to each image in "Raw photos/1_PHI_V".
    # e.g., if image_files = ['img1_frame_0.png', 'img2_frame_0.png'], strat_labels_file could contain:
    # classA
    # classB 
    # (matching the order of images found by _get_matching_files)
    print("Running example usage...")
    try:
        splitter = DatasetSplitter(
            image_folder="Raw photos/1_PHI_V", # Replace with your actual path
            label_folder="Raw photos/100_phi_threshold", # Replace with your actual path
            output_dir="./data_dir_PHI_split_example", # Example output
            random_seed=42
        )
        # For stratification example (assuming you have such a file):
        # strat_labels = [...] # Load your stratification labels here, same order as images
        split_info = splitter.split_dataset(val_percent=0.2, test_percent=0.1) # stratify_by_labels=strat_labels
        splitter.save_split_datasets(split_info, copy_files=True)
        splitter.create_split_summary(split_info)
        print("Example usage completed.")
    except ValueError as ve:
        print(f"Example usage failed: {ve}. Please ensure paths are correct and files exist.")
    except Exception as e:
        print(f"An unexpected error occurred during example usage: {e}")

if __name__ == "__main__":
    main()
    # To run the example directly (ensure paths are valid):
    # example_usage()
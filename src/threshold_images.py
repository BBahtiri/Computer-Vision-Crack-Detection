"""
Example usage script for Crack Detection U-Net.
This script demonstrates how to use the trained models for crack detection.
NOTE: This script was originally incomplete. Thresholding logic is not implemented here.
"""

import os
import yaml # Ensure PyYAML is installed: pip install PyYAML
import numpy as np
from keras.models import load_model # From tensorflow.keras.models for TF 2.x
import segmentation_models as sm # Ensure segmentation_models is installed

# MODIFIED: Assuming prediction_new_model.py is in the same directory or accessible in PYTHONPATH
from prediction_new_model import ModelInference, visualize_prediction, Config as PredictionConfig

# Load configuration for this script (threshold_images.py)
# This config might override or supplement parts of PredictionConfig
# For simplicity, we'll primarily use PredictionConfig and assume it's set up correctly.
# If 'config.yaml' is specific to this script's needs, load it.
# Example:
# try:
#     with open('config_threshold.yaml', 'r') as f: # A config specific to this script
#         threshold_config = yaml.safe_load(f)
# except FileNotFoundError:
#     print("Warning: config_threshold.yaml not found. Using defaults or PredictionConfig.")
#     threshold_config = {}


def example_single_prediction():
    """Example: Predict on a single image"""
    print("\n=== Single Image Prediction Example ===")
    
    pred_cfg = PredictionConfig() # Use the Config from prediction_new_model.py

    # MODIFIED: Check if model path and image path exist
    if not os.path.exists(pred_cfg.MODEL_PATH):
        print(f"Model not found at: {pred_cfg.MODEL_PATH}. Skipping single prediction example.")
        return
        
    # Example image path (ensure this image exists in your test set)
    # This should be a full path or relative to where the script is run.
    # Using a placeholder relative to the TEST_IMAGES_DIR from PredictionConfig
    example_image_name = "sample_frame_0.png" # REPLACE with an actual image name from your test set
    image_path = os.path.join(pred_cfg.TEST_IMAGES_DIR, example_image_name) 

    if not os.path.exists(image_path):
        print(f"Example image not found at: {image_path}. Ensure TEST_IMAGES_DIR in prediction_new_model.py's Config is correct and the image exists.")
        print(f"Current TEST_IMAGES_DIR is: {pred_cfg.TEST_IMAGES_DIR}")
        return

    try:
        inference_engine = ModelInference(pred_cfg.MODEL_PATH)
        print(f"Predicting on: {image_path}")
        visualize_prediction(image_path, inference_engine, pred_cfg, save_dir=pred_cfg.OUTPUT_DIR)
        print(f"Prediction visualization (if successful) might be saved in: {pred_cfg.OUTPUT_DIR}")
    except Exception as e:
        print(f"Error during single image prediction: {e}")


def example_batch_prediction_and_pixel_counts():
    """Example: Predict on multiple images and count pixels per class."""
    print("\n=== Batch Prediction & Pixel Count Example ===")
    
    pred_cfg = PredictionConfig()

    if not os.path.exists(pred_cfg.MODEL_PATH):
        print(f"Model not found at: {pred_cfg.MODEL_PATH}. Skipping batch prediction example.")
        return
    if not os.path.isdir(pred_cfg.TEST_IMAGES_DIR):
        print(f"Test images directory not found: {pred_cfg.TEST_IMAGES_DIR}. Skipping batch prediction.")
        return

    try:
        inference_engine = ModelInference(pred_cfg.MODEL_PATH)
        
        test_image_files = [
            f for f in os.listdir(pred_cfg.TEST_IMAGES_DIR) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        if not test_image_files:
            print(f"No images found in {pred_cfg.TEST_IMAGES_DIR}")
            return

        # Process a few images from the test directory
        num_images_to_process = min(3, len(test_image_files)) # Process up to 3 images
        selected_images = random.sample(test_image_files, num_images_to_process)
            
        print(f"Processing {num_images_to_process} images for pixel counts...")
        
        for img_name in selected_images:
            # MODIFIED: Use os.path.join for robust path construction
            img_path = os.path.join(pred_cfg.TEST_IMAGES_DIR, img_name)
            
            try:
                _, _, prediction_probs = inference_engine.predict_from_path(
                    img_path, pred_cfg.IMAGE_HEIGHT, pred_cfg.IMAGE_WIDTH, pred_cfg.BACKBONE
                )
                
                # Convert probability maps to class predictions
                # For multi-class, prediction_probs shape: (H, W, NumClasses)
                predicted_classes_map = np.argmax(prediction_probs, axis=-1) # Shape: (H, W)
                
                # Count pixels per predicted class
                unique_classes, counts = np.unique(predicted_classes_map, return_counts=True)
                
                print(f"\nPixel counts for: {img_name}")
                total_pixels = predicted_classes_map.size
                for class_id_val, pixel_count in zip(unique_classes, counts):
                    # Map integer class_id_val back to string label if possible
                    # This assumes class_id_val (0, 1, 2) maps to Config.CLASSES indices
                    # or directly to Config.CLASS_VALUES_IN_MASK if those are 0-indexed.
                    # For this example, we'll just use the integer value.
                    # If CLASS_VALUES_IN_MASK are [1,2,3], argmax will give 0,1,2. Need careful mapping.
                    # Let's assume for now class_id_val from argmax (0, 1, ...) maps to CLASS_VALUES_IN_MASK.
                    class_label_str = f"ClassValue {class_id_val}" # Placeholder
                    try: # Attempt to map index from argmax to actual class value then to string label
                        # If Config.CLASSES = ['1', '2', '3'], then index 0 from argmax is class '1'
                        class_label_str = pred_cfg.CLASSES[class_id_val]
                    except IndexError:
                        pass # Keep placeholder if index is out of bounds for CLASSES list

                    percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
                    print(f"  Predicted {class_label_str}: {pixel_count} pixels ({percentage:.1f}%)")

            except Exception as e:
                print(f"Error processing {img_name} for pixel counts: {e}")
    except Exception as e:
        print(f"Error initializing model for batch prediction: {e}")


def example_apply_thresholding(prediction_probabilities, threshold=0.5):
    """
    Applies a threshold to probability maps to get binary masks.
    This is a basic example and might need adjustment based on single/multi-class output.
    """
    if prediction_probabilities.ndim == 3 and prediction_probabilities.shape[-1] > 1: # Multi-class softmax/sigmoid output
        # For multi-class, usually take argmax first, then threshold if necessary,
        # or threshold each probability channel if classes are not mutually exclusive.
        # If mutually exclusive, argmax is standard.
        print("Applying argmax for multi-class probabilities.")
        binary_mask = np.argmax(prediction_probabilities, axis=-1)
        # To make it a "binary" mask in the sense of [0, 1] per class, you might one-hot encode argmax result
        # or threshold individual probability channels.
        # For simplicity, if you want a mask for a specific class channel after argmax:
        # specific_class_of_interest = 1 # e.g., class '2' if CLASSES=['1','2','3'] maps to indices 0,1,2
        # binary_mask_for_specific_class = (binary_mask == specific_class_of_interest).astype(np.uint8)
        # return binary_mask_for_specific_class

        # If the goal is to threshold each probability map independently (for multi-label):
        return (prediction_probabilities > threshold).astype(np.uint8)

    elif prediction_probabilities.ndim == 2 or prediction_probabilities.shape[-1] == 1: # Binary sigmoid output
        binary_mask = (prediction_probabilities > threshold).astype(np.uint8)
        return binary_mask
    else:
        raise ValueError(f"Unsupported prediction_probabilities shape: {prediction_probabilities.shape}")

# NOTE: The original script was truncated. The `example_model_comparison` was incomplete.
# I'm adding a placeholder for it.
def example_model_comparison():
    """Placeholder for comparing different model predictions."""
    print("\n=== Model Comparison Example (Placeholder) ===")
    print("This function needs to be implemented.")
    print("You would typically load two or more models and compare their predictions on the same set of images.")
    # Example steps:
    # 1. Initialize ModelInference for model_A (e.g., from 'best_model_phi.h5')
    # 2. Initialize ModelInference for model_B (e.g., from 'best_unet_hyper_tuned.h5')
    # 3. Load a test image.
    # 4. Get prediction_A from model_A.
    # 5. Get prediction_B from model_B.
    # 6. Visualize original, prediction_A, prediction_B side-by-side.

if __name__ == "__main__":
    print("Running Prediction Examples Script")
    print("Make sure `prediction_new_model.py`'s Config class has correct paths.")
    
    example_single_prediction()
    example_batch_prediction_and_pixel_counts()
    example_model_comparison() # This is a placeholder

    # Example of using the thresholding function (you'd get `pred_probs` from a model)
    # pred_cfg_for_thresh = PredictionConfig()
    # if os.path.exists(pred_cfg_for_thresh.MODEL_PATH) and os.path.isdir(pred_cfg_for_thresh.TEST_IMAGES_DIR):
    #     try:
    #         print("\n--- Thresholding Example ---")
    #         inference_for_thresh = ModelInference(pred_cfg_for_thresh.MODEL_PATH)
    #         test_image_files_thresh = [f for f in os.listdir(pred_cfg_for_thresh.TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #         if test_image_files_thresh:
    #             img_path_thresh = os.path.join(pred_cfg_for_thresh.TEST_IMAGES_DIR, test_image_files_thresh[0])
    #             _, _, pred_probs_thresh = inference_for_thresh.predict_from_path(
    #                 img_path_thresh, pred_cfg_for_thresh.IMAGE_HEIGHT, pred_cfg_for_thresh.IMAGE_WIDTH, pred_cfg_for_thresh.BACKBONE
    #             )
    #             
    #             thresholded_mask_example = example_apply_thresholding(pred_probs_thresh, threshold=0.5)
    #             print(f"Shape of thresholded mask for {os.path.basename(img_path_thresh)}: {thresholded_mask_example.shape}")
    #             # Visualize this thresholded_mask_example if needed
    #             # visualize(original_image=cv2.imread(img_path_thresh), thresholded_mask=thresholded_mask_example)
    #         else:
    #             print("No test images found for thresholding example.")
    #     except Exception as e:
    #         print(f"Error in thresholding example: {e}")
    # else:
    #     print("Skipping thresholding example due to missing model or test image directory.")

    print("\nScript finished.")
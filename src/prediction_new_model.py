"""
Model inference and visualization script.
This script loads a trained model and performs predictions on test images.
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import segmentation_models as sm # Ensure this is available and compatible
import albumentations as A

# Configuration
class Config:
    """Central configuration for inference."""
    # MODIFIED: Ensure MODEL_PATH points to the model trained by hyper.py or phi.py (corrected versions)
    MODEL_PATH = 'best_unet_hyper_tuned_corrected.h5' # Or 'best_model_phi_corrected.h5'
    # MODIFIED: This backbone should match the backbone of the loaded MODEL_PATH
    # It's used for getting the correct preprocessing function.
    BACKBONE = 'resnet34' # Example, adjust if your best model used a different one
    
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    # MODIFIED: These are the string labels for your classes.
    # The actual integer values used in masks should align with SegmentationDataset in training.
    CLASSES = ['1', '2', '3'] 
    CLASS_VALUES_IN_MASK = [1, 2, 3] # Actual integer values corresponding to CLASSES in the mask files
    
    # MODIFIED: Use os.path.join for data directory
    TEST_IMAGES_DIR = os.path.join("data_dir", "test_images") # Example, ensure this exists
    TEST_LABELS_DIR = os.path.join("data_dir", "test_labels") # Needed for get_label_mask

    SAVE_PREDICTIONS = True
    OUTPUT_DIR = "predictions_output" # MODIFIED: More descriptive name

# Utility functions
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize(**images):
    n = len(images); plt.figure(figsize=(16, 5)) # Adjusted for more images if needed
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1); plt.xticks([]); plt.yticks([])
        plt.title(' '.join(name.split('_')).title()); plt.imshow(image)
    plt.tight_layout(); plt.show()

def denormalize(x):
    x_max = np.percentile(x, 98); x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min + 1e-6); return x.clip(0, 1)

# Preprocessing functions
def preprocess_input_image(image_path, target_height, target_width, backbone_name='resnet34'):
    """Preprocesses a single image for model inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (target_width, target_height))
    
    preprocess_fn = sm.get_preprocessing(backbone_name)
    image = preprocess_fn(image)
    return image

def get_label_mask(image_path, target_height, target_width, class_values_in_mask, test_labels_dir):
    """
    Loads and preprocesses the ground truth mask for comparison.
    MODIFIED: Made more robust mask path generation.
    """
    img_filename = os.path.basename(image_path)
    # Specific filename transformation (assumed critical for the dataset)
    mask_filename = img_filename.replace("frame_0.png", "frame_last.png")
    
    label_path = os.path.join(test_labels_dir, mask_filename)
    
    mask = cv2.imread(label_path, 0) # Read as grayscale
    if mask is None:
        print(f"Warning: Could not find label mask at: {label_path} (derived from {image_path})")
        return None
    
    mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    masks_channels = [(mask == v) for v in class_values_in_mask]
    mask_stacked = np.stack(masks_channels, axis=-1).astype('float32')
    return mask_stacked

# Inference class
class ModelInference:
    """Handle model loading and inference."""
    def __init__(self, model_path):
        print(f"Loading model from: {model_path}")
        # MODIFIED: Suggestion #4 - Add custom_objects for robust loading
        custom_objects = {
            'DiceLoss': sm.losses.DiceLoss(),
            'CategoricalFocalLoss': sm.losses.CategoricalFocalLoss(),
            'BinaryFocalLoss': sm.losses.BinaryFocalLoss(),
            'iou_score': sm.metrics.IOUScore(threshold=0.5), # Ensure threshold matches training
            'f1-score': sm.metrics.FScore(threshold=0.5, name='f1-score') # Ensure name matches
            # Add any other custom components if used during model saving/training
        }
        try:
            # Try loading with compile=False first as it's faster for inference only
            self.model = load_model(model_path, custom_objects=custom_objects, compile=False)
        except Exception as e:
            print(f"Failed to load model with compile=False: {e}. Trying with compile=True...")
            # If compile=False fails (e.g. model contains training-specific ops not handled by compile=False),
            # try loading with compile=True. This is slower but more robust for complex models.
            # The optimizer state will be loaded but not used for inference.
            self.model = load_model(model_path, custom_objects=custom_objects, compile=True)
        print("Model loaded successfully!")

    def predict(self, image_array):
        """Performs inference on a single preprocessed image array."""
        image_batch = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_batch, verbose=0)
        return prediction.squeeze() # Remove batch dimension

    def predict_from_path(self, image_path, target_height, target_width, backbone_name):
        """Performs inference on an image from file path."""
        preprocessed_image = preprocess_input_image(image_path, target_height, target_width, backbone_name)
        
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_image_resized = cv2.resize(original_image, (target_width, target_height))
        
        prediction = self.predict(preprocessed_image)
        return original_image_resized, preprocessed_image, prediction

def visualize_prediction(image_path, model_inference_instance, config_instance, save_dir=None):
    """Visualizes model prediction on a single image."""
    original_image, _, prediction = model_inference_instance.predict_from_path(
        image_path, config_instance.IMAGE_HEIGHT, config_instance.IMAGE_WIDTH, config_instance.BACKBONE
    )
    
    ground_truth = get_label_mask(
        image_path, config_instance.IMAGE_HEIGHT, config_instance.IMAGE_WIDTH,
        config_instance.CLASS_VALUES_IN_MASK, config_instance.TEST_LABELS_DIR
    )
    
    vis_dict = {'original_image': original_image}
    num_classes_to_show = prediction.shape[-1]

    for i in range(num_classes_to_show):
        class_label = config_instance.CLASSES[i] if i < len(config_instance.CLASSES) else f'class_{i+1}'
        if ground_truth is not None and ground_truth.shape[-1] == num_classes_to_show:
            vis_dict[f'gt_{class_label}'] = ground_truth[..., i]
        # MODIFIED: Apply argmax for multi-class visualization if needed, or show probability maps
        # For this visualization, we show the probability map for each class.
        vis_dict[f'pred_prob_{class_label}'] = prediction[..., i] 

    visualize(**vis_dict)
    
    if save_dir:
        ensure_dir(save_dir)
        save_path = os.path.join(save_dir, f"prediction_{os.path.basename(image_path)}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")

def evaluate_on_test_set(model_inference_instance, config_instance, num_samples=5):
    """Evaluates model on random test samples and visualizes."""
    test_image_files = [
        os.path.join(config_instance.TEST_IMAGES_DIR, f) 
        for f in os.listdir(config_instance.TEST_IMAGES_DIR) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not test_image_files:
        print(f"No test images found in {config_instance.TEST_IMAGES_DIR}")
        return
    
    print(f"\nFound {len(test_image_files)} test images.")
    num_samples_to_show = min(num_samples, len(test_image_files))
    print(f"Visualizing {num_samples_to_show} random samples...")
    
    random_indices = random.sample(range(len(test_image_files)), num_samples_to_show)
    
    for idx, i in enumerate(random_indices):
        image_path = test_image_files[i]
        print(f"\nProcessing image {idx + 1}/{num_samples_to_show}: {os.path.basename(image_path)}")
        visualize_prediction(
            image_path, model_inference_instance, config_instance, 
            save_dir=config_instance.OUTPUT_DIR if config_instance.SAVE_PREDICTIONS else None
        )

def main():
    """Main inference pipeline."""
    print("Model Inference and Visualization"); print("=" * 50)
    
    cfg = Config() # Load configuration
    
    # Ensure necessary directories exist
    if not os.path.exists(cfg.TEST_IMAGES_DIR):
        print(f"Error: Test images directory not found: {cfg.TEST_IMAGES_DIR}")
        return
    if not os.path.exists(cfg.TEST_LABELS_DIR):
        print(f"Warning: Test labels directory not found: {cfg.TEST_LABELS_DIR}. Ground truth masks will not be shown.")
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"Error: Model file not found: {cfg.MODEL_PATH}")
        return

    model_inf = ModelInference(cfg.MODEL_PATH)
    
    # Option: Evaluate on random test samples
    evaluate_on_test_set(model_inf, cfg, num_samples=3) # Visualize 3 samples
    
    print("\nInference completed!")

if __name__ == "__main__":
    main()
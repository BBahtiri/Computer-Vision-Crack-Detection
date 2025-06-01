"""
Standard U-Net training script using segmentation_models.
"""
import os
import cv2
import numpy as np
import keras
import albumentations as A
import segmentation_models as sm
import matplotlib.pyplot as plt

# MODIFIED: Using a Config class for better organization
class Config:
    """Configuration for the training pipeline."""
    DATA_DIR = "data_dir"  # Ensure this directory exists and is populated
    IMAGE_HEIGHT, IMAGE_WIDTH = 512, 512
    CLASSES = ['1', '2', '3'] # String labels for your classes
    NUM_CLASSES = len(CLASSES)
    
    # Model settings
    BACKBONE = 'resnet34' # MODIFIED: Example, was resnet152 which is heavy
    # MODIFIED: Suggestion #5 - Use pre-trained weights
    ENCODER_WEIGHTS = 'imagenet' 
    FINAL_ACTIVATION = 'softmax' if NUM_CLASSES > 1 else 'sigmoid'

    # Training settings
    BATCH_SIZE = 8 # Adjust based on GPU memory for selected backbone
    LEARNING_RATE = 0.001 # A common starting LR for Adam
    EPOCHS = 100 # Rely on EarlyStopping

    # Directory paths
    X_TRAIN_DIR = os.path.join(DATA_DIR, 'train_images')
    Y_TRAIN_DIR = os.path.join(DATA_DIR, 'train_labels')
    X_VALID_DIR = os.path.join(DATA_DIR, 'val_images')
    Y_VALID_DIR = os.path.join(DATA_DIR, 'val_labels')
    X_TEST_DIR = os.path.join(DATA_DIR, 'test_images')
    Y_TEST_DIR = os.path.join(DATA_DIR, 'test_labels')
    
    MODEL_SAVE_PATH = './best_model_seg_corrected.h5'
    LOG_SAVE_PATH = "training_log_seg_corrected.csv"
    PLOT_SAVE_PATH = 'training_validation_plot_seg_corrected.png'

# Helper functions (Same as other scripts, ideal for a utils.py)
def visualize(**images):
    n = len(images); plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1); plt.xticks([]); plt.yticks([]);
        plt.title(' '.join(name.split('_')).title()); plt.imshow(image)
    plt.tight_layout(); plt.show()

def denormalize(x):
    x_max = np.percentile(x, 98); x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min + 1e-6); return x.clip(0, 1)

# MODIFIED: Using the corrected Dataset and DataLoader classes
class SegmentationDataset: # Identical to corrected hyper.py / phi.py
    POSSIBLE_MASK_VALUES = [0, 1, 2, 3]
    def __init__(self, images_dir, masks_dir, classes_to_extract=None, augmentation=None, preprocessing_fn=None):
        self.ids = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_root_dir = masks_dir
        classes_to_extract = classes_to_extract if classes_to_extract is not None else Config.CLASSES
        self.class_values_to_segment = [int(cls_str) for cls_str in classes_to_extract]
        self.augmentation = augmentation
        self.preprocessing_fn = preprocessing_fn
    def __getitem__(self, i):
        image_path = self.images_fps[i]
        image = cv2.imread(image_path);
        if image is None: raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = self._get_mask_path(image_path)
        mask_raw = cv2.imread(mask_path, 0);
        # BUGFIX: Suggestion #7 - Raise error if mask is not found
        if mask_raw is None: raise FileNotFoundError(f"Mask not found: {mask_path} (derived from {image_path})")
        image_resized = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        mask_resized = cv2.resize(mask_raw, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
        masks_channels = [(mask_resized == v) for v in self.class_values_to_segment]
        mask_final = np.stack(masks_channels, axis=-1).astype('float32')
        if self.augmentation:
            sample = self.augmentation(image=image_resized, mask=mask_final)
            image_resized, mask_final = sample['image'], sample['mask']
        if self.preprocessing_fn: image_processed = self.preprocessing_fn(image_resized)
        else: image_processed = image_resized.astype('float32') / 255.0
        return image_processed, mask_final
    def _get_mask_path(self, image_path): # Identical robust version
        img_filename = os.path.basename(image_path)
        mask_filename = img_filename.replace("frame_0.png", "frame_last.png")
        img_parent_dir_name = os.path.basename(os.path.dirname(image_path))
        if img_parent_dir_name == os.path.basename(Config.X_TRAIN_DIR): mask_dir = Config.Y_TRAIN_DIR
        elif img_parent_dir_name == os.path.basename(Config.X_VALID_DIR): mask_dir = Config.Y_VALID_DIR
        elif img_parent_dir_name == os.path.basename(Config.X_TEST_DIR): mask_dir = Config.Y_TEST_DIR
        else: mask_dir = os.path.dirname(image_path).replace("_images", "_labels")
        if not os.path.exists(mask_dir): mask_dir = self.masks_root_dir
        return os.path.join(mask_dir, mask_filename)
    def __len__(self): return len(self.images_fps)

class Dataloder(keras.utils.Sequence): # Identical corrected version
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset; self.batch_size = batch_size; self.shuffle = shuffle
        self.indexes = np.arange(len(dataset)); self.on_epoch_end()
    def __getitem__(self, i):
        start = i * self.batch_size; stop = min((i + 1) * self.batch_size, len(self.indexes))
        indexes_in_batch = self.indexes[start:stop]
        batch_data = [self.dataset[idx_in_dataset] for idx_in_dataset in indexes_in_batch]
        return np.array([item[0] for item in batch_data]), np.array([item[1] for item in batch_data])
    def __len__(self): return int(np.ceil(len(self.indexes) / self.batch_size))
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.indexes)

# Augmentation & Preprocessing (Same as corrected hyper.py)
def get_training_augmentation():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([A.CLAHE(p=1),A.RandomBrightnessContrast(p=1),A.RandomGamma(p=1)],p=0.9),
        A.OneOf([A.Sharpen(p=1),A.Blur(blur_limit=3,p=1),A.MotionBlur(blur_limit=3,p=1)],p=0.9),
        A.HueSaturationValue(p=0.3),
    ])
def get_validation_augmentation(): return A.Compose([])

def get_preprocessing_fn(backbone_name): # Renamed for clarity
    return sm.get_preprocessing(backbone_name)

def main():
    cfg = Config()
    print(f"Training U-Net with Backbone: {cfg.BACKBONE}")

    # Create datasets
    preprocess_input = get_preprocessing_fn(cfg.BACKBONE)
    train_dataset = SegmentationDataset(
        cfg.X_TRAIN_DIR, cfg.Y_TRAIN_DIR, classes_to_extract=cfg.CLASSES,
        augmentation=get_training_augmentation(), preprocessing_fn=preprocess_input
    )
    valid_dataset = SegmentationDataset(
        cfg.X_VALID_DIR, cfg.Y_VALID_DIR, classes_to_extract=cfg.CLASSES,
        augmentation=get_validation_augmentation(), preprocessing_fn=preprocess_input
    )
    train_dataloader = Dataloder(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False) # Use same batch size for val

    # Visualize a sample
    if len(train_dataset) > 0:
        print("\nVisualizing a training sample...")
        image, mask = train_dataset[0]
        vis_masks = {f'mask_class_{cfg.CLASSES[j]}': mask[..., j].squeeze() for j in range(mask.shape[-1])}
        visualize(image=denormalize(image), **vis_masks)

    # Create model
    model = sm.Unet(cfg.BACKBONE, classes=cfg.NUM_CLASSES, activation=cfg.FINAL_ACTIVATION, encoder_weights=cfg.ENCODER_WEIGHTS)
    
    # Define optimizer and loss
    optimizer = keras.optimizers.Adam(learning_rate=cfg.LEARNING_RATE)
    # MODIFIED: Define class weights if needed for DiceLoss, e.g., if classes are imbalanced
    # class_weights_for_loss = np.array([1.0, 1.0, 1.0]) # Adjust based on class frequencies
    dice_loss = sm.losses.DiceLoss() # class_weights=class_weights_for_loss
    focal_loss = sm.losses.CategoricalFocalLoss() if cfg.NUM_CLASSES > 1 else sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + (1.0 * focal_loss) # Equal weighting

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5, name='f1-score')]
    model.compile(optimizer, total_loss, metrics)

    # Load weights if model file exists (for resuming training)
    if os.path.exists(cfg.MODEL_SAVE_PATH):
        print(f"Loading pre-existing model weights from {cfg.MODEL_SAVE_PATH}")
        # MODIFIED: Load with custom_objects to be safe
        custom_objects_for_load = {
            'DiceLoss': sm.losses.DiceLoss(), 
            'CategoricalFocalLoss': sm.losses.CategoricalFocalLoss(),
            'BinaryFocalLoss': sm.losses.BinaryFocalLoss(),
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5, name='f1-score')
        }
        # Load weights into the defined model structure
        model.load_weights(cfg.MODEL_SAVE_PATH)
        # Or load the full model:
        # model = keras.models.load_model(cfg.MODEL_SAVE_PATH, custom_objects=custom_objects_for_load)
        # model.compile(optimizer, total_loss, metrics) # Re-compile if optimizer state needs reset or LR changes

    callbacks = [
        keras.callbacks.ModelCheckpoint(cfg.MODEL_SAVE_PATH, save_weights_only=False, save_best_only=True, monitor='val_loss', mode='min', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1), # MODIFIED: Increased patience
        keras.callbacks.CSVLogger(cfg.LOG_SAVE_PATH, append=True),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True) # MODIFIED: Increased patience
    ]

    # Train model - MODIFIED: Using model.fit instead of fit_generator
    print("\nStarting model training...")
    history = model.fit(
        train_dataloader,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
        validation_data=valid_dataloader,
        verbose=1
    )

    # Plot training history (using corrected plot function)
    plt.figure(figsize=(15, 5))
    metrics_to_plot = [('iou_score', 'IoU Score'), ('f1-score', 'F1-Score'), ('loss', 'Loss')]
    for i, (metric_key, title_str) in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), i + 1)
        if metric_key in history.history: plt.plot(history.history[metric_key], label=f'Train {title_str}')
        val_metric_key = f'val_{metric_key}'
        if val_metric_key in history.history: plt.plot(history.history[val_metric_key], label=f'Validation {title_str}')
        plt.title(title_str); plt.ylabel(title_str); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(cfg.PLOT_SAVE_PATH); plt.show()

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = SegmentationDataset(
        cfg.X_TEST_DIR, cfg.Y_TEST_DIR, classes_to_extract=cfg.CLASSES,
        augmentation=get_validation_augmentation(), preprocessing_fn=preprocess_input
    )
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for evaluation
    
    # MODIFIED: Load best model from checkpoint for final evaluation
    print(f"Loading best model from {cfg.MODEL_SAVE_PATH} for final evaluation.")
    custom_objects_for_load = { # Duplicated here for clarity, can be global
        'DiceLoss': sm.losses.DiceLoss(), 
        'CategoricalFocalLoss': sm.losses.CategoricalFocalLoss(),
        'BinaryFocalLoss': sm.losses.BinaryFocalLoss(),
        'iou_score': sm.metrics.IOUScore(threshold=0.5),
        'f1-score': sm.metrics.FScore(threshold=0.5, name='f1-score')
    }
    model.load_weights(cfg.MODEL_SAVE_PATH) # Weights are loaded into current model structure
    # Or: model = keras.models.load_model(cfg.MODEL_SAVE_PATH, custom_objects=custom_objects_for_load)

    # MODIFIED: Using model.evaluate instead of evaluate_generator
    scores = model.evaluate(test_dataloader)
    print("Test Set Evaluation Results:")
    print(f"Loss: {scores[0]:.5f}")
    for metric_obj, value in zip(metrics, scores[1:]): # model.metrics might be better if names are set
        print(f"Mean {metric_obj.name if hasattr(metric_obj, 'name') else type(metric_obj).__name__}: {value:.5f}")

    # Visualize predictions on a few test samples
    print("\nVisualizing predictions on test samples...")
    n_vis = min(3, len(test_dataset))
    if n_vis > 0:
        ids = np.random.choice(np.arange(len(test_dataset)), size=n_vis, replace=False)
        for i in ids:
            image, gt_mask = test_dataset[i]
            pr_mask = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            vis_data = {'image': denormalize(image.squeeze())}
            for k_idx in range(gt_mask.shape[-1]):
                 vis_data[f'gt_mask_class_{cfg.CLASSES[k_idx]}'] = gt_mask[..., k_idx].squeeze()
                 vis_data[f'pr_mask_class_{cfg.CLASSES[k_idx]}'] = pr_mask[..., k_idx].squeeze()
            visualize(**vis_data)
    print("\nTraining and evaluation finished.")

if __name__ == '__main__':
    # Ensure all configured directories exist before running
    # os.makedirs(Config.DATA_DIR, exist_ok=True)
    # os.makedirs(Config.X_TRAIN_DIR, exist_ok=True) ... etc.
    main()
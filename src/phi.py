"""
Custom U-Net implementation with dropout regularization.
This script trains a modified U-Net with ResNet152 backbone and custom decoder.
"""

import os
import cv2
import numpy as np
import keras
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Dropout
from keras.models import Model
import albumentations as A
import segmentation_models as sm
import matplotlib.pyplot as plt


# Configuration
class Config:
    """Central configuration for the training pipeline."""
    # Data settings
    DATA_DIR = "data_dir_PHI"
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    CLASSES = ['1', '2', '3']
    NUM_CLASSES = len(CLASSES)
    
    # Model settings
    BACKBONE = 'resnet152'
    DROPOUT_RATE = 0.5
    ACTIVATION = 'softmax'  # 'sigmoid' for binary segmentation
    
    # Training settings
    BATCH_SIZE = 8
    LEARNING_RATE = 0.01
    EPOCHS = 1000
    
    # Directory paths
    TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')
    TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels')
    VAL_IMAGES = os.path.join(DATA_DIR, 'val_images')
    VAL_LABELS = os.path.join(DATA_DIR, 'val_labels')
    TEST_IMAGES = os.path.join(DATA_DIR, 'test_images')
    TEST_LABELS = os.path.join(DATA_DIR, 'test_labels')


# Utility functions
def visualize(**images):
    """Plot images in one row for comparison."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    
    plt.tight_layout()
    plt.show()


def denormalize(x):
    """Scale image to range 0-1 for visualization."""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    return x.clip(0, 1)


# Dataset class
class SegmentationDataset:
    """Dataset class for loading images and masks with augmentations."""
    
    CLASSES = ['0', '1', '2', '3']
    
    def __init__(self, images_dir, masks_dir, classes=None, 
                 augmentation=None, preprocessing=None):
        """
        Initialize dataset.
        
        Args:
            images_dir: Path to images folder
            masks_dir: Path to masks folder
            classes: List of class names to extract
            augmentation: Albumentations augmentation pipeline
            preprocessing: Preprocessing function for the model
        """
        self.ids = sorted(os.listdir(images_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # Convert class names to indices
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        """Get image and mask pair."""
        # Read image
        image_path = self.images_fps[i]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask
        mask_path = self._get_mask_path(image_path)
        mask = cv2.imread(mask_path, 0)
        
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Resize to target dimensions
        image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        mask = cv2.resize(mask, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
        
        # Create multi-channel mask for different classes
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # Apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image, mask
    
    def _get_mask_path(self, image_path):
        """Convert image path to corresponding mask path."""
        return (image_path.replace("frame_0.png", "frame_last.png")
                         .replace("train_images", "train_labels")
                         .replace("val_images", "val_labels")
                         .replace("test_images", "test_labels"))
    
    def __len__(self):
        """Return total number of samples."""
        return len(self.ids)


class DataLoader(keras.utils.Sequence):
    """Data loader for batch generation."""
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        """Initialize data loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()
    
    def __getitem__(self, i):
        """Generate one batch of data."""
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        
        batch_data = []
        for j in range(start, stop):
            batch_data.append(self.dataset[self.indexes[j]])
        
        # Transpose list of tuples
        batch = [np.stack(samples, axis=0) for samples in zip(*batch_data)]
        return batch
    
    def __len__(self):
        """Return number of batches per epoch."""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch if required."""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# Augmentation pipelines
def get_training_augmentation():
    """Create training augmentation pipeline."""
    train_transform = [
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH, always_apply=True),
        # Add more augmentations here as needed
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomGamma(p=0.2),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Create validation augmentation pipeline (minimal)."""
    val_transform = [
        A.Resize(height=Config.IMAGE_HEIGHT, width=Config.IMAGE_WIDTH, always_apply=True)
    ]
    return A.Compose(val_transform)


def get_preprocessing(preprocessing_fn):
    """Create preprocessing pipeline."""
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


# Model architecture
def create_decoder_block(input_tensor, skip_features, num_filters, dropout_rate=0.5):
    """
    Create a decoder block with skip connections and dropout.
    
    Args:
        input_tensor: Input tensor from previous layer
        skip_features: Skip connection features from encoder
        num_filters: Number of filters for convolution layers
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Output tensor
    """
    # Upsample
    x = UpSampling2D((2, 2))(input_tensor)
    
    # Concatenate with skip features
    x = concatenate([x, skip_features])
    
    # Two convolution blocks
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
    
    # Dropout for regularization
    x = Dropout(dropout_rate)(x)
    
    return x


def build_custom_unet(input_shape, num_classes, backbone_name='resnet152', dropout_rate=0.5):
    """
    Build custom U-Net with dropout in decoder.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        backbone_name: Name of the backbone model
        dropout_rate: Dropout rate for decoder blocks
    
    Returns:
        Keras model
    """
    # Load pre-trained backbone
    base_model = sm.Unet(
        backbone_name,
        classes=num_classes,
        activation='softmax',
        encoder_weights='imagenet',
        encoder_freeze=True
    )
    
    # Get encoder outputs for skip connections
    # Note: These layer indices are specific to ResNet152
    skip_connections = [
        base_model.layers[92].output,   # Skip connection 1
        base_model.layers[154].output,  # Skip connection 2
        base_model.layers[366].output,  # Skip connection 3
        base_model.layers[478].output,  # Skip connection 4
    ]
    
    # Get the last encoder output
    encoder_output = base_model.get_layer("decoder_stage4b_relu").output
    
    # Build decoder with dropout
    decoder0 = create_decoder_block(encoder_output, skip_connections[3], 256, dropout_rate)
    decoder1 = create_decoder_block(decoder0, skip_connections[2], 128, dropout_rate)
    decoder2 = create_decoder_block(decoder1, skip_connections[1], 64, dropout_rate)
    decoder3 = create_decoder_block(decoder2, skip_connections[0], 32, dropout_rate)
    
    # Final upsampling and output
    decoder4 = UpSampling2D((2, 2))(decoder3)
    output = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(decoder4)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=output)
    
    return model


def create_callbacks():
    """Create training callbacks."""
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            './best_model_phi.h5',
            save_weights_only=False,
            save_best_only=True,
            mode='min',
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=10,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.CSVLogger("training_log_phi.csv", append=True),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            verbose=1,
            restore_best_weights=True
        )
    ]
    return callbacks


def compile_model(model):
    """Compile model with optimizer and loss."""
    # Optimizer
    optimizer = keras.optimizers.Adam(Config.LEARNING_RATE)
    
    # Loss function (combination of Dice and Focal loss)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = (sm.losses.BinaryFocalLoss() if Config.NUM_CLASSES == 1 
                  else sm.losses.CategoricalFocalLoss())
    total_loss = dice_loss + focal_loss
    
    # Metrics
    metrics = [
        sm.metrics.IOUScore(threshold=0.5),
        sm.metrics.FScore(threshold=0.5)
    ]
    
    model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)
    return model


def plot_training_history(history):
    """Plot training metrics."""
    plt.figure(figsize=(15, 5))
    
    # Plot IoU scores
    plt.subplot(1, 2, 1)
    plt.plot(history.history['iou_score'], label='Train')
    plt.plot(history.history['val_iou_score'], label='Validation')
    plt.title('Model IoU Score')
    plt.ylabel('IoU Score')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_validation_plot_phi.png')
    plt.show()


def evaluate_model(model, test_dataloader, test_dataset):
    """Evaluate model on test set and visualize predictions."""
    # Calculate metrics
    scores = model.evaluate(test_dataloader, verbose=1)
    metrics_names = ['loss', 'iou_score', 'f1-score']
    
    print("\nTest Set Evaluation:")
    print("-" * 30)
    for name, score in zip(metrics_names, scores):
        print(f"{name}: {score:.5f}")
    
    # Visualize random predictions
    n_samples = 5
    ids = np.random.choice(len(test_dataset), size=n_samples)
    
    print("\nVisualizing predictions...")
    for i in ids:
        image, gt_mask = test_dataset[i]
        image_input = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image_input, verbose=0)
        
        visualize(
            image=denormalize(image.squeeze()),
            ground_truth=gt_mask.squeeze(),
            prediction=pr_mask.squeeze()
        )


def main():
    """Main training pipeline."""
    print("Starting Custom U-Net Training with Dropout")
    print("=" * 50)
    
    # Set up preprocessing
    preprocess_input = sm.get_preprocessing(Config.BACKBONE)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMAGES,
        Config.TRAIN_LABELS,
        classes=Config.CLASSES,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )
    
    valid_dataset = SegmentationDataset(
        Config.VAL_IMAGES,
        Config.VAL_LABELS,
        classes=Config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )
    
    test_dataset = SegmentationDataset(
        Config.TEST_IMAGES,
        Config.TEST_LABELS,
        classes=Config.CLASSES,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocess_input)
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Build model
    print("\nBuilding custom U-Net model...")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Dropout rate: {Config.DROPOUT_RATE}")
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    input_shape = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3)
    model = build_custom_unet(
        input_shape=input_shape,
        num_classes=Config.NUM_CLASSES,
        backbone_name=Config.BACKBONE,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    # Compile model
    model = compile_model(model)
    print("\nModel compiled successfully!")
    
    # Train model
    print("\nStarting training...")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    
    history = model.fit(
        train_dataloader,
        steps_per_epoch=len(train_dataloader),
        epochs=Config.EPOCHS,
        validation_data=valid_dataloader,
        validation_steps=len(valid_dataloader),
        callbacks=create_callbacks(),
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluate_model(model, test_dataloader, test_dataset)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
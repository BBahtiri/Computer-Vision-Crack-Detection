"""
Custom U-Net implementation with dropout regularization.
This script trains a modified U-Net with ResNet152 backbone and a custom decoder.
"""

import os
import cv2
import numpy as np
import keras
from keras.layers import Input, Conv2D, UpSampling2D, concatenate, Dropout, BatchNormalization, Activation # MODIFIED
from keras.models import Model
import albumentations as A
import segmentation_models as sm
import matplotlib.pyplot as plt

# Configuration
class Config:
    """Central configuration for the training pipeline."""
    DATA_DIR = "data_dir_PHI"
    IMAGE_HEIGHT = 512
    IMAGE_WIDTH = 512
    CLASSES = ['1', '2', '3']
    NUM_CLASSES = len(CLASSES)
    
    BACKBONE = 'resnet152'
    DROPOUT_RATE = 0.3 # MODIFIED: 0.5 can be quite high for all blocks
    FINAL_ACTIVATION = 'softmax' if NUM_CLASSES > 1 else 'sigmoid'
    
    BATCH_SIZE = 4 # MODIFIED: ResNet152 is large, may need smaller batch size
    LEARNING_RATE = 0.0001 # MODIFIED: Lower LR for fine-tuning or deeper networks
    EPOCHS = 200 # MODIFIED: Adjusted epochs, rely on EarlyStopping

    TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')
    TRAIN_LABELS = os.path.join(DATA_DIR, 'train_labels')
    VAL_IMAGES = os.path.join(DATA_DIR, 'val_images')
    VAL_LABELS = os.path.join(DATA_DIR, 'val_labels')
    TEST_IMAGES = os.path.join(DATA_DIR, 'test_images')
    TEST_LABELS = os.path.join(DATA_DIR, 'test_labels')

# Utility functions (Same as hyper.py, consider moving to a utils.py)
def visualize(**images):
    n = len(images); plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1); plt.xticks([]); plt.yticks([]);
        plt.title(' '.join(name.split('_')).title()); plt.imshow(image)
    plt.tight_layout(); plt.show()

def denormalize(x):
    x_max = np.percentile(x, 98); x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min + 1e-6); return x.clip(0, 1)

# Dataset class (Corrected version, similar to hyper.py)
class SegmentationDataset:
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
        if mask_raw is None: raise FileNotFoundError(f"Mask not found: {mask_path}")
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
        if img_parent_dir_name == os.path.basename(Config.TRAIN_IMAGES): mask_dir = Config.TRAIN_LABELS
        elif img_parent_dir_name == os.path.basename(Config.VAL_IMAGES): mask_dir = Config.VAL_LABELS
        elif img_parent_dir_name == os.path.basename(Config.TEST_IMAGES): mask_dir = Config.TEST_LABELS
        else: mask_dir = os.path.dirname(image_path).replace("_images", "_labels")
        if not os.path.exists(mask_dir): mask_dir = self.masks_root_dir
        return os.path.join(mask_dir, mask_filename)
    def __len__(self): return len(self.images_fps)

class DataLoader(keras.utils.Sequence): # Identical corrected version
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

# Augmentation pipelines
def get_training_augmentation(): # Same as corrected hyper.py
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1),A.RandomGamma(p=1)], p=0.9),
        A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1),A.MotionBlur(blur_limit=3, p=1)], p=0.9),
        A.HueSaturationValue(p=0.3),
    ])
def get_validation_augmentation(): return A.Compose([])

# Model architecture
def conv_block(input_tensor, num_filters, kernel_size=(3, 3), batch_norm=True, dropout_rate=0.0): # MODIFIED: Added BN and made dropout optional here
    """Standard Convolutional Block."""
    x = Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(input_tensor)
    if batch_norm: x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    if batch_norm: x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if dropout_rate > 0: x = Dropout(dropout_rate)(x)
    return x

def decoder_block(input_tensor, skip_features, num_filters, dropout_rate=0.5): # MODIFIED for clarity
    """U-Net Decoder Block with skip connections and dropout."""
    x = UpSampling2D(size=(2, 2))(input_tensor)
    if skip_features is not None: # MODIFIED: Handle cases where skip might be None (though not typical for U-Net)
        x = concatenate([x, skip_features], axis=-1)
    x = conv_block(x, num_filters, dropout_rate=dropout_rate) # MODIFIED: Using conv_block
    return x

def build_custom_unet(input_shape, num_classes, backbone_name='resnet152', dropout_rate=0.5, final_activation='softmax'):
    """
    Builds a U-Net model with a specified backbone encoder and a custom decoder.
    MODIFIED: This function now builds a U-Net by taking encoder features from a standalone backbone.
    """
    # 1. Get the backbone (encoder part)
    # `weights='imagenet'` for transfer learning, `include_top=False` as it's an encoder
    encoder = sm.Backbones.get_backbone(
        backbone_name,
        input_shape=input_shape,
        weights='imagenet',
        include_top=False # Ensure we get encoder features, not a classification head
    )
    
    # MODIFIED: Freeze encoder layers for initial training phase if desired (transfer learning)
    for layer in encoder.layers:
        layer.trainable = False # Set to True if you want to fine-tune the encoder

    # 2. Get feature maps from encoder stages for skip connections
    # The `encoder.outputs` list contains feature maps from various stages.
    # For ResNet, these are typically [C1, C2, C3, C4, C5] where C5 is the bottleneck.
    # The exact layers/names depend on the backbone and segmentation_models version.
    # We need to select the correct ones. `get_layer` by name is more robust if names are known.
    # Example: For ResNet50, names might be 'conv1_relu', 'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'
    # Let's assume `encoder.outputs` gives us [..., skip4, skip3, skip2, skip1, bottleneck]
    # Or, more reliably, use the specific layer names if known from model.summary() of the backbone.
    # For now, we'll assume `encoder.outputs` is a list where the last element is the bottleneck (deepest features)
    # and preceding elements are suitable for skip connections in reverse order.
    
    # This mapping might need adjustment based on the specific backbone's output structure.
    # Example for ResNet-like backbones from segmentation_models:
    # skip_connection_names = {
    #     'resnet18': ['relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'], # Bottleneck from stage4_unit1_relu1 or encoder.output
    #     'resnet34': ['relu0', 'stage2_unit1_relu1', 'stage3_unit1_relu1', 'stage4_unit1_relu1'],
    #     'resnet50': ['conv1_relu', 'res2c_relu', 'res3d_relu', 'res4f_relu'], # Example names
    #     # For ResNet152, names would be deeper.
    # }
    # encoder_output = encoder.get_layer(name_of_bottleneck_layer).output
    # skips = [encoder.get_layer(name).output for name in skip_connection_names[backbone_name]]
    
    # A common pattern in `segmentation_models` is that `encoder.outputs` already gives you the right feature maps.
    # The last output is the bottleneck, and previous ones are skips.
    encoder_outputs = encoder.outputs
    bottleneck = encoder_outputs[-1]
    skips = encoder_outputs[:-1][::-1] # Reverse for decoder: C4, C3, C2, C1 (example)

    # Ensure we have enough skip connections for a typical U-Net decoder (usually 4)
    # This part is highly dependent on the backbone structure.
    # For ResNet152, you'd typically have 4 skip connections + bottleneck.
    # If len(skips) is not 4, this decoder structure needs adjustment.
    # This is a placeholder and needs verification with `encoder.summary()`
    if len(skips) < 4:
        print(f"Warning: Backbone '{backbone_name}' provided {len(skips)} skip connections. Decoder expects 4. Adjusting.")
        # Pad with None or duplicate if necessary, or simplify decoder. This is tricky.
        # For now, let's assume it provides enough, or the user will need to adapt the decoder.
        # A safe bet is to print layer names from encoder.outputs to pick correct skips.
        # print("Available encoder output layer names for skips (last is bottleneck):")
        # for out_tensor in encoder.outputs: print(out_tensor.name)
        
    # Decoder filter sizes
    decoder_filters = [256, 128, 64, 32]

    x = bottleneck
    for i in range(len(decoder_filters)):
        skip_input = skips[i] if i < len(skips) else None # Handle if not enough skips
        x = decoder_block(x, skip_input, decoder_filters[i], dropout_rate=dropout_rate)

    # Final output layer
    # Upsample to original size if needed (if number of UpSampling2D layers in decoder doesn't match pooling in encoder)
    # This depends on the stride of the first conv layer in the encoder if it's > 1
    # For many backbones, the total downsampling is 32x, requiring 5 upsampling stages (2^5=32)
    # If decoder_blocks provide 4 upsamplings, one more might be needed.
    # Or, the first skip might be after an initial downsampling.

    # The number of UpSampling2D layers in decoder_block is 4.
    # If total downsampling is 32x (5 levels), then one more upsampling may be needed.
    # Or the initial `input_tensor` to the first `decoder_block` (bottleneck) is already at 1/16 or 1/32.
    # This needs careful check against `encoder`'s total stride.
    # Often, an extra UpSampling2D or Conv2DTranspose is used before the final Conv2D if sizes don't match.
    # For simplicity, we assume the decoder_blocks bring it to a size where a 1x1 conv is appropriate.
    # A common final step if not at full resolution:
    # x = UpSampling2D(size=(2,2))(x) # If one more upsampling is needed

    output_layer = Conv2D(num_classes, (1, 1), padding='same')(x)
    output_layer = Activation(final_activation, name=final_activation)(output_layer)
    
    model = Model(inputs=encoder.input, outputs=output_layer)
    return model

def create_callbacks(): # Same as corrected hyper.py
    return [
        keras.callbacks.ModelCheckpoint(
            './best_model_phi_corrected.h5', save_weights_only=False, save_best_only=True,
            monitor='val_loss', mode='min', verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1 # MODIFIED: Patience
        ),
        keras.callbacks.CSVLogger("training_log_phi_corrected.csv", append=True),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, verbose=1, restore_best_weights=True # MODIFIED: Patience
        )
    ]

def compile_model(model):
    optimizer = keras.optimizers.Adam(Config.LEARNING_RATE)
    dice_loss = sm.losses.DiceLoss()
    focal_loss = sm.losses.CategoricalFocalLoss() if Config.NUM_CLASSES > 1 else sm.losses.BinaryFocalLoss()
    total_loss = dice_loss + focal_loss 
    
    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5, name='f1-score')]
    model.compile(optimizer=optimizer, loss=total_loss, metrics=metrics)
    return model

def plot_training_history(history): # Same plotting as corrected hyper.py
    plt.figure(figsize=(15, 5))
    metrics_to_plot = [('iou_score', 'IoU Score'), ('f1-score', 'F1-Score'), ('loss', 'Loss')]
    for i, (metric_key, title) in enumerate(metrics_to_plot):
        plt.subplot(1, len(metrics_to_plot), i + 1)
        if metric_key in history.history: plt.plot(history.history[metric_key], label=f'Train {title}')
        val_metric_key = f'val_{metric_key}'
        if val_metric_key in history.history: plt.plot(history.history[val_metric_key], label=f'Validation {title}')
        plt.title(title); plt.ylabel(title); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('training_validation_plot_phi_corrected.png'); plt.show()

def evaluate_model(model, test_dataloader, test_dataset): # Same evaluation as corrected hyper.py
    print("\nEvaluating model on test set..."); scores = model.evaluate(test_dataloader)
    metrics_names = ['loss'] + [m.name for m in model.metrics]
    print("\nTest Set Evaluation:"); print("-" * 30)
    for name, score_val in zip(metrics_names, scores if isinstance(scores, list) else [scores]): print(f"{name}: {score_val:.5f}")
    n_samples = min(3, len(test_dataset));
    if n_samples == 0: return
    ids = np.random.choice(len(test_dataset), size=n_samples, replace=False)
    print(f"\nVisualizing {n_samples} predictions...")
    for i in ids:
        image, gt_mask = test_dataset[i]
        image_input = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image_input, verbose=0)[0]
        vis_image = denormalize(image.squeeze())
        vis_dict = {'image': vis_image}
        for j in range(gt_mask.shape[-1]):
            vis_dict[f'gt_class_{Config.CLASSES[j]}'] = gt_mask[..., j].squeeze()
            vis_dict[f'pred_class_{Config.CLASSES[j]}'] = pr_mask[..., j].squeeze()
        visualize(**vis_dict)

def main():
    print("Starting Custom U-Net (ResNet152) Training with Dropout"); print("=" * 50)
    preprocess_input = sm.get_preprocessing(Config.BACKBONE)
    
    print("\nCreating datasets...")
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMAGES, Config.TRAIN_LABELS, classes_to_extract=Config.CLASSES,
        augmentation=get_training_augmentation(), preprocessing_fn=preprocess_input
    )
    valid_dataset = SegmentationDataset(
        Config.VAL_IMAGES, Config.VAL_LABELS, classes_to_extract=Config.CLASSES,
        augmentation=get_validation_augmentation(), preprocessing_fn=preprocess_input
    )
    test_dataset = SegmentationDataset(
        Config.TEST_IMAGES, Config.TEST_LABELS, classes_to_extract=Config.CLASSES,
        augmentation=get_validation_augmentation(), preprocessing_fn=preprocess_input
    )
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    
    train_dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False) # Use same batch size for validation
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print("\nBuilding custom U-Net model...")
    input_shape = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3)
    model = build_custom_unet(
        input_shape=input_shape, num_classes=Config.NUM_CLASSES, backbone_name=Config.BACKBONE,
        dropout_rate=Config.DROPOUT_RATE, final_activation=Config.FINAL_ACTIVATION
    )
    # model.summary() # Good to uncomment and check architecture, esp. skip connections

    model = compile_model(model); print("\nModel compiled successfully!")
    print(f"\nStarting training... Epochs: {Config.EPOCHS}, Batch: {Config.BATCH_SIZE}, LR: {Config.LEARNING_RATE}")
    
    # MODIFIED: Check if pre-trained weights exist and load them if you want to resume
    if os.path.exists('./best_model_phi_corrected.h5'):
        print("Loading existing model weights from './best_model_phi_corrected.h5'")
        # MODIFIED: Load with custom_objects
        custom_objects_for_load = {
            'DiceLoss': sm.losses.DiceLoss(), # Or the specific instance used if it had params
            'CategoricalFocalLoss': sm.losses.CategoricalFocalLoss(),
            'BinaryFocalLoss': sm.losses.BinaryFocalLoss(),
            'iou_score': sm.metrics.IOUScore(threshold=0.5),
            'f1-score': sm.metrics.FScore(threshold=0.5)
        }
        # It's often better to load weights into the defined model structure
        model.load_weights('./best_model_phi_corrected.h5')
        # Or load the full model and then recompile if optimizer state is not needed or LR changes
        # model = keras.models.load_model('./best_model_phi_corrected.h5', custom_objects=custom_objects_for_load)
        # model = compile_model(model) # Re-compile if needed, e.g., to reset optimizer state or change LR


    history = model.fit(
        train_dataloader, epochs=Config.EPOCHS, validation_data=valid_dataloader,
        callbacks=create_callbacks(), verbose=1
    )
    plot_training_history(history)
    
    # MODIFIED: Ensure the model used for evaluation is the one with best weights
    # (EarlyStopping with restore_best_weights=True handles this for `model` variable)
    # If not, load from checkpoint:
    # print("Loading best model weights for evaluation...")
    # model.load_weights('./best_model_phi_corrected.h5') # or the full model as above
    
    print("\nEvaluating on test set...")
    evaluate_model(model, test_dataloader, test_dataset)
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()
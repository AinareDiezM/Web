# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 2025

@author: Ainare

U-Net for lung segmentation:
 - Trains with BCE + Dice + Focal loss
 - Includes metrics: Dice, IoU, Precision, Recall
 - Uses EarlyStopping with moderate patience (10) to avoid waiting indefinitely
 - Saves probabilities and binarized masks for later metrics computation
"""

#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
import random
from glob import glob

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# BASIC CONFIGURATION
# ------------------------------------------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 1   

# ------------------------------------------------------------------
# DATASET PATHS (SPLIT + AUG)
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM"

TRAIN_IMAGES_DIR      = os.path.join(BASE_DIR, "train - sin vacio", "Images")
TRAIN_MASKS_DIR       = os.path.join(BASE_DIR, "train - sin vacio", "Masks")
TRAIN_AUG_IMAGES_DIR  = os.path.join(BASE_DIR, "train_aug - sin vacio", "Images")
TRAIN_AUG_MASKS_DIR   = os.path.join(BASE_DIR, "train_aug - sin vacio", "Masks")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "val - sin vacio", "Images")
VAL_MASKS_DIR  = os.path.join(BASE_DIR, "val - sin vacio", "Masks")

TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test - sin vacio", "Images")
TEST_MASKS_DIR  = os.path.join(BASE_DIR, "test - sin vacio", "Masks")


# ------------------------------------------------------------------
# FUNCTION TO LOAD AND MATCH IMAGE/MASK FROM TWO FOLDERS
# ------------------------------------------------------------------
def load_dataset(images_dirs, masks_dirs, img_height, img_width):
    """
    images_dirs, masks_dirs: list of folders (to combine train + train_aug)
    Returns X, Y normalized and binarized.
    """
    # 1) Collect paths
    image_paths_raw = []
    mask_paths_raw  = []

    for d in images_dirs:
        if d is not None and os.path.exists(d):
            image_paths_raw += glob(os.path.join(d, "*.png"))

    for d in masks_dirs:
        if d is not None and os.path.exists(d):
            mask_paths_raw  += glob(os.path.join(d, "*.png"))

    print(f"Images found in {images_dirs}: {len(image_paths_raw)}")
    print(f"Masks found in {masks_dirs}: {len(mask_paths_raw)}")

    # 2) Dictionary of masks by base name
    mask_dict = {os.path.basename(p): p for p in mask_paths_raw}

    image_paths = []
    mask_paths  = []

    for img_path in image_paths_raw:
        img_name = os.path.basename(img_path)
        # p1_adc_slice_001.png -> p1_adc_mask_001.png
        # p1_adc_slice_001_aug1.png -> p1_adc_mask_001_aug1.png
        mask_name = img_name.replace("_slice_", "_mask_")

        if mask_name in mask_dict:
            image_paths.append(img_path)
            mask_paths.append(mask_dict[mask_name])
        else:
            print(f"Mask not found for image {img_name} (expected {mask_name})")

    n_images = len(image_paths)
    print(f"Final number of image-mask pairs: {n_images}")
    assert n_images > 0, "Could not match images and masks. Check the filenames."

    # 3) Allocate arrays
    X = np.zeros((n_images, img_height, img_width, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((n_images, img_height, img_width, 1),             dtype=np.float32)

    print("Loading and resizing images and masks...")

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):

        # ----- Image -----
        img = imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        elif img.ndim == 3 and img.shape[-1] > 1:
            img = img[..., 0:1]

        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        img = img / 255.0 

        X[i] = img

        # ----- Mask -----
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = resize(mask, (img_height, img_width), mode='constant', preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        Y[i] = mask

    print("Loading completed.")
    print("X shape:", X.shape, "Y shape:", Y.shape)
    return X, Y


# ------------------------------------------------------------------
# LOAD TRAIN (ORIGINAL + AUGMENTED) AND VALIDATION
# ------------------------------------------------------------------
X_train, Y_train = load_dataset(
    images_dirs=[TRAIN_IMAGES_DIR, TRAIN_AUG_IMAGES_DIR],
    masks_dirs=[TRAIN_MASKS_DIR,  TRAIN_AUG_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

X_val, Y_val = load_dataset(
    images_dirs=[VAL_IMAGES_DIR],
    masks_dirs=[VAL_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:",   X_val.shape,   "Y_val:",   Y_val.shape)


# ------------------------------------------------------------------
# CHECK HOW MANY SLICES HAVE TUMOR (for weights)
# ------------------------------------------------------------------
train_has_tumor = (Y_train.reshape(Y_train.shape[0], -1).sum(axis=1) > 0)
val_has_tumor   = (Y_val.reshape(Y_val.shape[0],   -1).sum(axis=1)   > 0)

print(f"Slices with tumor in TRAIN : {train_has_tumor.sum()} of {len(train_has_tumor)}")
print(f"Slices with tumor in VAL   : {val_has_tumor.sum()} of {len(val_has_tumor)}")

# Weights per slice: those with tumor have higher weight
sample_weights = np.where(train_has_tumor, 3.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------
# QUICK VISUALIZATION OF A TRAIN EXAMPLE
# ------------------------------------------------------------------
idx = random.randint(0, len(X_train)-1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Train image")
imshow(X_train[idx].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Train mask")
imshow(Y_train[idx].squeeze(), cmap="gray")
plt.axis("off")
plt.show()


# ------------------------------------------------------------------
# METRICS AND LOSS FUNCTIONS: BCE + Dice + Focal
# ------------------------------------------------------------------
def dice_coefficient_tf(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient_tf(y_true, y_pred)

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, smooth=1e-6):
    """
    Standard binary focal loss:
      - gamma controls how much you emphasize hard examples
      - alpha controls the class balance
    """
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

def bce_dice_focal_loss(y_true, y_pred):
    bce   = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    floss = binary_focal_loss(y_true, y_pred)
    # Weights (adjustable): 0.4 BCE, 0.3 Dice, 0.3 Focal
    return 0.4 * bce + 0.3 * dloss + 0.3 * floss

# Extra metrics
def iou_metric_tf(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)

def precision_metric_tf(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.round(y_pred)
    y_true_f   = tf.reshape(y_true, [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FP = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    return (TP + smooth) / (TP + FP + smooth)

def recall_metric_tf(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.round(y_pred)
    y_true_f   = tf.reshape(y_true, [-1])
    y_pred_f   = tf.reshape(y_pred_bin, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    return (TP + smooth) / (TP + FN + smooth)


# ------------------------------------------------------------------
# U-NET MODEL DEFINITION
# ------------------------------------------------------------------
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = inputs

# Contracting path
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

# Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_focal_loss,
    metrics=[
        'accuracy',
        dice_coefficient_tf,
        iou_metric_tf,
        precision_metric_tf,
        recall_metric_tf
    ]
)

model.summary()

# ------------------------------------------------------------------
# CUSTOM CALLBACK: SAVE PREDICTION EXAMPLES AS PNG
# ------------------------------------------------------------------
class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val,
                 output_dir="training_examples_bce_dice_focal",
                 num_images=3,
                 freq=5):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.output_dir = output_dir
        self.num_images = num_images
        self.freq = freq

        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Save in the first epoch and then every 'freq' epochs
        if (epoch + 1) % self.freq != 0 and epoch != 0:
            return

        print(f"\nSaving prediction examples at epoch {epoch+1}...")
        # Choose random validation indices
        idxs = np.random.choice(self.x_val.shape[0],
                                size=min(self.num_images, self.x_val.shape[0]),
                                replace=False)

        # Model predictions
        preds = self.model.predict(self.x_val[idxs], verbose=0)
        preds_bin = (preds > 0.5).astype(np.float32)

        for i, idx in enumerate(idxs):
            img = self.x_val[idx, ..., 0]
            gt  = self.y_val[idx, ..., 0]
            pr  = preds_bin[i, ..., 0]

            fig, axs = plt.subplots(1, 3, figsize=(9, 3))

            axs[0].imshow(img, cmap='gray')
            axs[0].set_title("Image")
            axs[0].axis("off")

            axs[1].imshow(gt, cmap='gray')
            axs[1].set_title("GT mask")
            axs[1].axis("off")

            axs[2].imshow(img, cmap='gray')
            axs[2].imshow(pr, alpha=0.4)
            axs[2].set_title("Prediction")
            axs[2].axis("off")

            plt.tight_layout()
            fname = os.path.join(
                self.output_dir,
                f"epoch{epoch+1:03d}_slice{idx}_ex{i}.png"
            )
            plt.savefig(fname, dpi=150)
            plt.close(fig)


# ------------------------------------------------------------------
# CALLBACKS 
# ------------------------------------------------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'unet_lung_256_gray_aug_bce_dice_focal.h5',
    verbose=1,
    save_best_only=True,
    monitor='val_loss'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=10,
    monitor='val_loss',
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# Callback to save examples as PNG
save_examples_cb = SavePredictionsCallback(
    X_val, Y_val,
    output_dir="training_examples_bce_dice_focal",
    num_images=3,
    freq=5    # save every 5 epochs
)

callbacks = [
    checkpointer,
    early_stop,
    reduce_lr,
    tf.keras.callbacks.TensorBoard(log_dir='logs_lung_aug_bce_dice_focal'),
    save_examples_cb
]


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
results = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=60,
    callbacks=callbacks
)


# ------------------------------------------------------------------
# PREDICTIONS AND SANITY CHECK
# ------------------------------------------------------------------
preds_val = model.predict(X_val, verbose=1)        # PROBABILITIES  
preds_val_t = (preds_val > 0.5).astype(np.uint8)   # BINARIZED THR-0.5

ix = random.randint(0, len(X_val)-1)
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Image")
imshow(X_val[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("GT mask")
imshow(Y_val[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("U-Net Prediction (thr=0.5)")
imshow(preds_val_t[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.show()

# ================================================================
# SAVE ARRAYS FOR METRICS
# ================================================================
np.save("Y_val_modelo_bce_dice_focal.npy", Y_val)
np.save("preds_val_modelo_bce_dice_focal_probs.npy", preds_val)
np.save("preds_val_modelo_bce_dice_focal_bin05.npy", preds_val_t)

print("Arrays saved for metrics")
print(" - Y_val_modelo_bce_dice_focal.npy")
print(" - preds_val_modelo_bce_dice_focal_probs.npy")
print(" - preds_val_modelo_bce_dice_focal_bin05.npy")

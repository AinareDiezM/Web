# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:17:38 2025

@author: Ainare
"""

# -*- coding: utf-8 -*-
"""
U-Net for lung tumour segmentation – NEW PROTOTYPE
 - Dataset without empty slices (train, train_aug, val)
 - Adjusted loss: 0.3 * BCE + 0.6 * Dice + 0.1 * Focal
 - Oversampling of slices with small tumours in TRAIN
 - Two-phase training:
     * Phase 1: on train + train_aug (with oversampling)
     * Phase 2 (fine-tuning): only slices with tumour (without oversampling)
 - Fixed threshold: 0.90
 - Post-processing: keep the largest connected component
 - Saves:
     * X_val.npy
     * Y_val_modelo9_bce_dice_focal.npy
     * preds_val_modelo9_bce_dice_focal_probs.npy
     * preds_val_modelo9_bce_dice_focal_bestthr.npy
     * preds_val_modelo9_bce_dice_focal_bestthr_pp.npy
 - Generates PNGs in folder:
     * resultados_val_nuevo_modelo_png
"""

#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
import random
from glob import glob

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.measure import label, regionprops
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
IMG_CHANNELS = 1   # GRAYSCALE

# ------------------------------------------------------------------
# DATASET PATHS
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
# FUNCTION TO LOAD AND MATCH IMAGE/MASK FROM FOLDERS
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
    print(f"Final number of image–mask pairs: {n_images}")
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

# Save X_val for future visualisations/metrics
np.save("X_val.npy", X_val)

# ------------------------------------------------------------------
# INFORMATION ABOUT SLICES WITH TUMOUR
# ------------------------------------------------------------------
train_has_tumor = (Y_train.reshape(Y_train.shape[0], -1).sum(axis=1) > 0)
val_has_tumor   = (Y_val.reshape(Y_val.shape[0],   -1).sum(axis=1)   > 0)

print(f"Slices with tumour in TRAIN : {train_has_tumor.sum()} of {len(train_has_tumor)}")
print(f"Slices with tumour in VAL   : {val_has_tumor.sum()} of {len(val_has_tumor)}")

# ------------------------------------------------------------------
# OVERSAMPLING OF SMALL-TUMOUR SLICES (TRAIN ONLY)
# ------------------------------------------------------------------
tumor_area_train = Y_train.reshape(Y_train.shape[0], -1).sum(axis=1)

# Define "small tumour": adjust this threshold if needed
pequeño_umbral = 500.0
mask_tumor_pequeño = (tumor_area_train > 0) & (tumor_area_train < pequeño_umbral)

idx_tumor_pequeño = np.where(mask_tumor_pequeño)[0]
idx_all           = np.arange(Y_train.shape[0])

print(f"Slices with small tumour in TRAIN: {len(idx_tumor_pequeño)}")

factor_rep = 2  # repeat each small-tumour slice 2 additional times
idx_extra  = np.repeat(idx_tumor_pequeño, factor_rep)

idx_final = np.concatenate([idx_all, idx_extra])
np.random.shuffle(idx_final)

X_train_bal = X_train[idx_final]
Y_train_bal = Y_train[idx_final]

print("X_train_bal:", X_train_bal.shape, "Y_train_bal:", Y_train_bal.shape)

# ------------------------------------------------------------------
# METRICS AND LOSS FUNCTIONS: BCE + Dice + Focal (ADJUSTED)
# ------------------------------------------------------------------
def dice_coefficient_tf(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient_tf(y_true, y_pred)

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1.0 - smooth)
    cross_entropy = - (y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
    weight = alpha * y_true * tf.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)

def bce_dice_focal_loss(y_true, y_pred):
    """
    Combined adjusted loss:
      0.3 * BCE + 0.6 * Dice + 0.1 * Focal
    """
    bce   = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dloss = dice_loss(y_true, y_pred)
    floss = binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25)
    return 0.3 * bce + 0.6 * dloss + 0.1 * floss

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
    metrics=['accuracy', dice_coefficient_tf, iou_metric_tf, precision_metric_tf, recall_metric_tf]
)

model.summary()

# ------------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'unet_lung_256_gray_aug_bce_dice_focal_nuevo.h5',
    verbose=1,
    save_best_only=True,
    monitor='val_loss'
)

early_stop_phase1 = tf.keras.callbacks.EarlyStopping(
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

tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir='logs_lung_aug_bce_dice_focal_nuevo')

# ------------------------------------------------------------------
# PHASE 1: NORMAL TRAINING (WITH OVERSAMPLING)
# ------------------------------------------------------------------
history1 = model.fit(
    X_train_bal, Y_train_bal,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=50,
    callbacks=[checkpointer, early_stop_phase1, reduce_lr, tensorboard_cb]
)

# ------------------------------------------------------------------
# PHASE 2: FINE-TUNING ONLY ON TUMOUR SLICES (WITHOUT OVERSAMPLING)
# ------------------------------------------------------------------
X_train_tumor = X_train[train_has_tumor]
Y_train_tumor = Y_train[train_has_tumor]

print("Fine-tuning on tumour slices:")
print("  X_train_tumor:", X_train_tumor.shape)
print("  Y_train_tumor:", Y_train_tumor.shape)

early_stop_phase2 = tf.keras.callbacks.EarlyStopping(
    patience=5,
    monitor='val_loss',
    restore_best_weights=True
)

history2 = model.fit(
    X_train_tumor, Y_train_tumor,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=20,
    callbacks=[checkpointer, early_stop_phase2, reduce_lr, tensorboard_cb]
)

# ------------------------------------------------------------------
# POST-PROCESSING (LARGEST CONNECTED COMPONENT)
# ------------------------------------------------------------------
def postprocess_mask(mask_bin):
    """
    mask_bin: array (H, W) with 0/1
    Returns a mask with only the largest connected component.
    """
    lab = label(mask_bin)
    if lab.max() == 0:
        return mask_bin

    regions = regionprops(lab)
    areas = np.array([r.area for r in regions])
    largest_label = regions[areas.argmax()].label

    mask_clean = (lab == largest_label).astype(np.uint8)
    return mask_clean

# ------------------------------------------------------------------
# VALIDATION PREDICTIONS + THRESHOLD + POST-PROCESSING
# ------------------------------------------------------------------
print("Predicting on the validation set...")
preds_val_probs = model.predict(X_val, verbose=1)

thr_opt = 0.90
print(f"Using threshold: {thr_opt}")

preds_val_bestthr = (preds_val_probs > thr_opt).astype(np.uint8)

preds_val_bestthr_pp = np.zeros_like(preds_val_bestthr)
for i in range(preds_val_bestthr.shape[0]):
    preds_val_bestthr_pp[i, ..., 0] = postprocess_mask(preds_val_bestthr[i, ..., 0])

# ------------------------------------------------------------------
# SAVE ARRAYS FOR METRICS – PROTOTYPE 9
# ------------------------------------------------------------------
np.save("Y_val_modelo9_bce_dice_focal.npy", Y_val)
np.save("preds_val_modelo9_bce_dice_focal_probs.npy", preds_val_probs)
np.save("preds_val_modelo9_bce_dice_focal_bestthr.npy", preds_val_bestthr)
np.save("preds_val_modelo9_bce_dice_focal_bestthr_pp.npy", preds_val_bestthr_pp)

print("Saved arrays for PROTOTYPE 9 metrics:")
print("  - Y_val_modelo9_bce_dice_focal.npy")
print("  - preds_val_modelo9_bce_dice_focal_probs.npy")
print("  - preds_val_modelo9_bce_dice_focal_bestthr.npy  (thr=0.90)")
print("  - preds_val_modelo9_bce_dice_focal_bestthr_pp.npy (thr=0.90 + post-processing)")

# ------------------------------------------------------------------
# GENERATE AUTOMATIC PNGs ON VALIDATION
# ------------------------------------------------------------------
OUT_DIR = "resultados_val_nuevo_modelo_png"
os.makedirs(OUT_DIR, exist_ok=True)

num_ejemplos = min(50, X_val.shape[0])

print(f"Saving {num_ejemplos} examples as PNG in '{OUT_DIR}'...")

for i in range(num_ejemplos):
    img  = X_val[i].squeeze()
    gt   = Y_val[i].squeeze()
    pred = preds_val_bestthr_pp[i].squeeze()

    img_u8  = (img * 255).astype("uint8")
    gt_u8   = (gt  * 255).astype("uint8")
    pred_u8 = (pred * 255).astype("uint8")

    imsave(os.path.join(OUT_DIR, f"val_{i:03d}_img.png"),  img_u8)
    imsave(os.path.join(OUT_DIR, f"val_{i:03d}_gt.png"),   gt_u8)
    imsave(os.path.join(OUT_DIR, f"val_{i:03d}_pred.png"), pred_u8)

print(f"Saved PNGs in '{OUT_DIR}'")

ix = random.randint(0, len(X_val)-1)
plt.figure(figsize=(12,4))

plt.subplot(1,4,1)
plt.title("Image")
plt.imshow(X_val[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,4,2)
plt.title("GT")
plt.imshow(Y_val[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title(f"Pred thr={thr_opt}")
plt.imshow(preds_val_bestthr[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Pred postproc")
plt.imshow(preds_val_bestthr_pp[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

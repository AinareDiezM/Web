# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 18:48:24 2025

@author: Ainare
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
IMG_CHANNELS = 1   # GRAYSCALE

# ------------------------------------------------------------------
# PATHS OF YOUR NEW DATASET (SPLIT + AUG)
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM"

TRAIN_IMAGES_DIR      = os.path.join(BASE_DIR, "train", "Images")
TRAIN_MASKS_DIR       = os.path.join(BASE_DIR, "train", "Masks")
TRAIN_AUG_IMAGES_DIR  = os.path.join(BASE_DIR, "train_aug", "Images")
TRAIN_AUG_MASKS_DIR   = os.path.join(BASE_DIR, "train_aug", "Masks")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "val", "Images")
VAL_MASKS_DIR  = os.path.join(BASE_DIR, "val", "Masks")

TEST_IMAGES_DIR = os.path.join(BASE_DIR, "test", "Images")
TEST_MASKS_DIR  = os.path.join(BASE_DIR, "test", "Masks")


# ------------------------------------------------------------------
# FUNCTION TO LOAD AND MATCH IMAGE/MASK FROM TWO FOLDERS
# ------------------------------------------------------------------
def load_dataset(images_dirs, masks_dirs, img_height, img_width):
    """
    images_dirs, masks_dirs: lists of folders 
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
        # Replacing 'slice' with 'mask' to build the expected mask name
        # This works for original and augmented images like:
        #   p1_adc_slice_001.png        -> p1_adc_mask_001.png
        #   p1_adc_slice_001_aug1.png   -> p1_adc_mask_001_aug1.png
        mask_name = img_name.replace("_slice_", "_mask_")

        if mask_name in mask_dict:
            image_paths.append(img_path)
            mask_paths.append(mask_dict[mask_name])
        else:
            print(f"⚠ Mask not found for image {img_name} (expected {mask_name})")

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
        img = img / 255.0  # normalization to [0,1]

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
# TRAIN = train + train_aug
X_train, Y_train = load_dataset(
    images_dirs=[TRAIN_IMAGES_DIR, TRAIN_AUG_IMAGES_DIR],
    masks_dirs=[TRAIN_MASKS_DIR,  TRAIN_AUG_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

# VALIDATION = only val folder
X_val, Y_val = load_dataset(
    images_dirs=[VAL_IMAGES_DIR],
    masks_dirs=[VAL_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:",   X_val.shape,   "Y_val:",   Y_val.shape)


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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# ------------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'unet_lung_256_gray_aug.h5',
    verbose=1,
    save_best_only=True
)

callbacks = [
    checkpointer,
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='logs_lung_aug')
]

# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
results = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=8,
    epochs=50,
    callbacks=callbacks
)

# ------------------------------------------------------------------
# PREDICTIONS AND SANITY CHECK
# ------------------------------------------------------------------
preds_val = model.predict(X_val, verbose=1)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

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
plt.title("U-Net prediction")
imshow(preds_val_t[ix].squeeze(), cmap="gray")
plt.axis("off")

plt.show()

# ================================================================
# SAVE ARRAYS FOR METRICS
# ================================================================
np.save("X_val_completetrain.npy", X_val)
np.save("Y_val_completetrain.npy", Y_val)
np.save("preds_val_completetrain.npy", preds_val_t)

print("Arrays saved for metrics:")
print(" - X_val_completetrain.npy")
print(" - Y_val_completetrain.npy")
print(" - preds_val_completetrain.npy")

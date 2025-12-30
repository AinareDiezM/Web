# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 20:33:23 2025

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
IMG_CHANNELS = 1

# ------------------------------------------------------------------
# DATASET PATHS
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM"

TRAIN_IMAGES_DIR      = os.path.join(BASE_DIR, "train", "Images")
TRAIN_MASKS_DIR       = os.path.join(BASE_DIR, "train", "Masks")
TRAIN_AUG_IMAGES_DIR  = os.path.join(BASE_DIR, "train_aug", "Images")
TRAIN_AUG_MASKS_DIR   = os.path.join(BASE_DIR, "train_aug", "Masks")

VAL_IMAGES_DIR = os.path.join(BASE_DIR, "val", "Images")
VAL_MASKS_DIR  = os.path.join(BASE_DIR, "val", "Masks")

# ------------------------------------------------------------------
# FUNCTION TO LOAD AND MATCH IMAGE/MASK
# ------------------------------------------------------------------
def load_dataset(images_dirs, masks_dirs, img_height, img_width):
    image_paths_raw = []
    mask_paths_raw  = []

    for d in images_dirs:
        image_paths_raw += glob(os.path.join(d, "*.png"))

    for d in masks_dirs:
        mask_paths_raw  += glob(os.path.join(d, "*.png"))

    mask_dict = {os.path.basename(p): p for p in mask_paths_raw}

    image_paths = []
    mask_paths  = []

    for img_path in image_paths_raw:
        img_name = os.path.basename(img_path)
        mask_name = img_name.replace("_slice_", "_mask_")

        if mask_name in mask_dict:
            image_paths.append(img_path)
            mask_paths.append(mask_dict[mask_name])

    n_images = len(image_paths)
    X = np.zeros((n_images, img_height, img_width, IMG_CHANNELS), dtype=np.float32)
    Y = np.zeros((n_images, img_height, img_width, 1), dtype=np.float32)

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):

        img = imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = resize(img, (img_height, img_width), preserve_range=True)
        img = img / 255.0
        X[i] = img

        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = resize(mask, (img_height, img_width), preserve_range=True)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        Y[i] = mask

    return X, Y

# ------------------------------------------------------------------
# LOAD DATASET (train + train_aug)
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

# ------------------------------------------------------------------
# DICE LOSS + BCE LOSS
# ------------------------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return 1 - (2 * inter + smooth) / (denom + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d   = dice_loss(y_true, y_pred)
    return bce + d

# ------------------------------------------------------------------
# U-NET
# ------------------------------------------------------------------
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = inputs

c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same')(c5)

u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    "unet_lung_256_gray_aug_bcedice.h5",
    monitor="val_loss",
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks = [
    checkpointer,
    reduce_lr,
    tf.keras.callbacks.EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir="logs_lung_aug_bcedice")
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
# SAVE PREDICTIONS FOR METRICS
# ------------------------------------------------------------------
preds_val = model.predict(X_val)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

np.save("Y_val_modelo2_bcedice.npy", Y_val)
np.save("preds_val_modelo2_bcedice.npy", preds_val_t)

print("Saved:")
print("- Y_val_modelo2_bcedice.npy")
print("- preds_val_modelo2_bcedice.npy")

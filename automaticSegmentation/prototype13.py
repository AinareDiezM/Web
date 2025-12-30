# -*- coding: utf-8 -*-
"""
Prototype 13 - ResU-Net + Focal Tversky
Direct reading from PNG:
 - Reads PNG images and masks from folders (train/val)
 - Builds X_train, Y_train, X_val, Y_val in memory
 - Saves the clean .npy files for Prototype 13:
     X_train_modelo13.npy, Y_train_modelo13.npy,
     X_val_modelo13.npy,   Y_val_modelo13.npy
 - Trains ResU-Net with Focal Tversky
 - Applies threshold 0.90 + largest connected component
 - Calculates metrics and saves them
 - Generates PNGs with Original + GT + Prediction (postprocessed)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                     concatenate, BatchNormalization, Dropout,
                                     Add)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from skimage.io import imread
from skimage.transform import resize
from skimage.measure import label, regionprops




# ------------------------------------------------------------------
# BASIC CONFIGURATION 
# ------------------------------------------------------------------
IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 1

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# PATHS TO PNG FOLDERS
TRAIN_IMG_DIR  = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM\train_aug - sin vacio\Images"
TRAIN_MASK_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM\train_aug - sin vacio\Masks"
VAL_IMG_DIR    = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM\val - sin vacio - f\Images"
VAL_MASK_DIR   = r"C:\Users\Ainare\Desktop\Automatic segmentation\BOTH DATASETS_FINAL_SPLIT_RANDOM\val - sin vacio - f\Masks"

# Folder where the specific Prototype 13 .npy files will be saved
NPY_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\DATASET_MODELO13_NPY"
os.makedirs(NPY_DIR, exist_ok=True)

# Folder where example PNG results will be saved
PNG_DIR = r"C:\Users\Ainare\Desktop\Automatic segmentation\RESULTADOS_MODELO13_PNG"
os.makedirs(PNG_DIR, exist_ok=True)

# ------------------------------------------------------------------
# LOADING FUNCTIONS FROM PNG → NP.ARRAY
# ------------------------------------------------------------------
def load_dataset_from_png(img_dir, mask_dir, img_height, img_width):
    """
    Reads PNGs from img_dir and mask_dir, pairs them by filename,
    resizes, normalizes, and returns X, Y with shape (N, H, W, 1).

    For your data:

    - VALIDATION:
        img : p1_scc_slice_003.png
        mask: p1_scc_mask_003.png

    - TRAIN:
        img : p1_adc_slice_001_aug1.png
        mask: p1_adc_mask_001_aug1.png

    The mask is obtained by replacing "_slice_" with "_mask_"
    in the filename.
    """
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]
    img_files = sorted(img_files)  

    X_list = []
    Y_list = []

    if len(img_files) == 0:
        raise RuntimeError(f"No PNG files found in {img_dir}")

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)

        if "_slice_" not in fname:
            print(f"The file {fname} does not contain '_slice_'. Skipping.")
            continue

        mask_fname = fname.replace("_slice_", "_mask_")
        mask_path  = os.path.join(mask_dir, mask_fname)

        if not os.path.exists(mask_path):
            print(f"Mask not found for {fname} -> {mask_fname}, skipping this slice.")
            continue

        # Read image and mask
        img = imread(img_path)
        msk = imread(mask_path)

        # Ensure grayscale (2D)
        if img.ndim == 3:
            img = img[..., 0] 
        if msk.ndim == 3:
            msk = msk[..., 0]

        # Resize
        img_res = resize(img, (img_height, img_width),
                         preserve_range=True, anti_aliasing=True)
        msk_res = resize(msk, (img_height, img_width),
                         preserve_range=True, anti_aliasing=False)

        # Convert types
        img_res = img_res.astype(np.float32)

        # Binarize mask (0/1)
        if np.max(msk_res) > 0:
            msk_bin = (msk_res >= 0.5 * np.max(msk_res)).astype(np.uint8)
        else:
            msk_bin = msk_res.astype(np.uint8)

        # Add channel axis
        img_res = img_res[..., np.newaxis]
        msk_bin = msk_bin[..., np.newaxis]

        X_list.append(img_res)
        Y_list.append(msk_bin)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    return X, Y


# ------------------------------------------------------------------
# LOAD DATA FROM PNG AND SAVE AS NPY
# ------------------------------------------------------------------
print("Loading TRAIN from PNG...")
X_train, Y_train = load_dataset_from_png(
    TRAIN_IMG_DIR, TRAIN_MASK_DIR, IMG_HEIGHT, IMG_WIDTH
)

print("Loading VAL from PNG...")
X_val, Y_val = load_dataset_from_png(
    VAL_IMG_DIR, VAL_MASK_DIR, IMG_HEIGHT, IMG_WIDTH
)

print("Original shapes (before normalization):")
print("  X_train:", X_train.shape, X_train.dtype)
print("  Y_train:", Y_train.shape, Y_train.dtype)
print("  X_val  :", X_val.shape,   X_val.dtype)
print("  Y_val  :", Y_val.shape,   Y_val.dtype)

# Normalize images (assuming PNG in [0,255] or similar)
X_train = X_train.astype(np.float32)
X_val   = X_val.astype(np.float32)

if X_train.max() > 1.0:
    X_train /= 255.0
if X_val.max() > 1.0:
    X_val /= 255.0

# Ensure masks are 0/1 and float32
Y_train = (Y_train > 0).astype(np.float32)
Y_val   = (Y_val   > 0).astype(np.float32)


print("Shapes after normalization:")
print("  X_train:", X_train.shape, X_train.dtype, "min/max:", X_train.min(), X_train.max())
print("  Y_train:", Y_train.shape, Y_train.dtype, "unique values:", np.unique(Y_train))
print("  X_val  :", X_val.shape,   X_val.dtype,   "min/max:", X_val.min(), X_val.max())
print("  Y_val  :", Y_val.shape,   Y_val.dtype,   "unique values:", np.unique(Y_val))

# Save .npy files for Prototype 13
np.save(os.path.join(NPY_DIR, "X_train_modelo13.npy"), X_train)
np.save(os.path.join(NPY_DIR, "Y_train_modelo13.npy"), Y_train)
np.save(os.path.join(NPY_DIR, "X_val_modelo13.npy"),   X_val)
np.save(os.path.join(NPY_DIR, "Y_val_modelo13.npy"),   Y_val)

print("Clean Prototype 13 .npy files saved in:", NPY_DIR)

# ------------------------------------------------------------------
# METRICS AND LOSSES
# ------------------------------------------------------------------
smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def tversky_index(y_true, y_pred, alpha=0.7, beta=0.3):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos   = K.sum(y_true_pos * y_pred_pos)
    false_neg  = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos  = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    ti = tversky_index(y_true, y_pred, alpha, beta)
    return K.pow((1.0 - ti), gamma)

def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ------------------------------------------------------------------
# RESU-NET ARCHITECTURE
# ------------------------------------------------------------------
def conv_block_res(x, filters, kernel_size=(3,3), padding='same', dropout=0.0):
    shortcut = Conv2D(filters, (1,1), padding=padding)(x)

    c = Conv2D(filters, kernel_size, padding=padding, activation='relu')(x)
    c = BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding)(c)
    c = BatchNormalization()(c)

    c = Add()([c, shortcut])
    c = tf.keras.layers.Activation('relu')(c)

    if dropout > 0.0:
        c = Dropout(dropout)(c)

    return c

def build_resunet(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)):
    inputs = Input(input_shape)

    # Encoder
    c1 = conv_block_res(inputs, 32)
    p1 = MaxPooling2D((2,2))(c1)

    c2 = conv_block_res(p1, 64)
    p2 = MaxPooling2D((2,2))(c2)

    c3 = conv_block_res(p2, 128)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = conv_block_res(p3, 256, dropout=0.3)
    p4 = MaxPooling2D((2,2))(c4)

    # Bottleneck
    c5 = conv_block_res(p4, 512, dropout=0.4)

    # Decoder
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block_res(u6, 256, dropout=0.3)

    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block_res(u7, 128)

    u8 = UpSampling2D((2,2))(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block_res(u8, 64)

    u9 = UpSampling2D((2,2))(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block_res(u9, 32)

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs], name="Modelo13_ResUNet")
    return model

model = build_resunet()
model.summary()

# ------------------------------------------------------------------
# MODEL COMPILATION
# ------------------------------------------------------------------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=focal_tversky_loss,
              metrics=[dice_coef, iou_coef, 'accuracy'])

# ------------------------------------------------------------------
# CALLBACKS
# ------------------------------------------------------------------
checkpoint_path = os.path.join(NPY_DIR, "modelo13_resunet_best.h5")

checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_dice_coef',
                             mode='max',
                             save_best_only=True,
                             verbose=1)

early_stop = EarlyStopping(monitor='val_dice_coef',
                           mode='max',
                           patience=20,
                           verbose=1,
                           restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_dice_coef',
                              mode='max',
                              factor=0.5,
                              patience=10,
                              min_lr=1e-6,
                              verbose=1)

# ------------------------------------------------------------------
# RESUME TRAINING (IF PREVIOUS CHECKPOINT EXISTS)
# ------------------------------------------------------------------
if os.path.exists(checkpoint_path):
    print("Checkpoint found. Loading previous weights from:")
    print("   ", checkpoint_path)
    try:
        model.load_weights(checkpoint_path)
        print("Weights loaded correctly. Training will resume from these weights.")
    except Exception as e:
        print("Could not load weights from checkpoint.")
        print("Error:", e)
else:
    print("No previous checkpoint found. Training will start from scratch.")


# ------------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------------
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    batch_size=2,
    epochs=100,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

print("Prototype 13 training completed.")

# ------------------------------------------------------------------
# POSTPROCESSING: LARGEST CONNECTED COMPONENT
# ------------------------------------------------------------------
def postprocess_mask(mask_bin):
    labeled = label(mask_bin)
    if labeled.max() == 0:
        return mask_bin
    max_region = max(regionprops(labeled), key=lambda r: r.area)
    mask_out = (labeled == max_region.label).astype(np.uint8)
    return mask_out

# ------------------------------------------------------------------
# PREDICTION ON VALIDATION SET + THRESHOLD + POSTPROCESSING
# ------------------------------------------------------------------
print("Predicting on validation set...")
preds_val_probs = model.predict(X_val, verbose=1)

thr_opt = 0.90
print(f"Using threshold: {thr_opt}")

preds_val_bestthr = (preds_val_probs > thr_opt).astype(np.uint8)

preds_val_bestthr_pp = np.zeros_like(preds_val_bestthr, dtype=np.uint8)
for i in range(preds_val_bestthr.shape[0]):
    preds_val_bestthr_pp[i, ..., 0] = postprocess_mask(preds_val_bestthr[i, ..., 0])

# ------------------------------------------------------------------
# METRIC CALCULATION PER SLICE (Dice, IoU)
# ------------------------------------------------------------------
def dice_np(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_np(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

n_slices = Y_val.shape[0]
dice_slices = np.zeros(n_slices)
iou_slices  = np.zeros(n_slices)

for i in range(n_slices):
    gt  = Y_val[i, ..., 0]
    prd = preds_val_bestthr_pp[i, ..., 0]
    dice_slices[i] = dice_np(gt, prd)
    iou_slices[i]  = iou_np(gt, prd)

print("======================================================================")
print("Evaluating: Prototype 13 - ResUNet + Focal Tversky (PNG→NPY custom pipeline)")
print(f"Number of evaluated slices: {n_slices}")
print("")
print("Dice coefficient:")
print(f"  Mean      : {100*np.mean(dice_slices):.2f} %")
print(f"  Std. dev. : {100*np.std(dice_slices):.2f} %")
print(f"  Minimum   : {100*np.min(dice_slices):.2f} %")
print(f"  Maximum   : {100*np.max(dice_slices):.2f} %")
print("")
print("IoU:")
print(f"  Mean      : {100*np.mean(iou_slices):.2f} %")
print(f"  Std. dev. : {100*np.std(iou_slices):.2f} %")
print(f"  Minimum   : {100*np.min(iou_slices):.2f} %")
print(f"  Maximum   : {100*np.max(iou_slices):.2f} %")
print("======================================================================")

# ------------------------------------------------------------------
# SAVE METRICS AND PREDICTIONS WITH NAME "modelo13"
# ------------------------------------------------------------------
np.save(os.path.join(NPY_DIR, "Y_val_modelo13.npy"), Y_val)
np.save(os.path.join(NPY_DIR, "preds_val_modelo13_bestthr_pp.npy"), preds_val_bestthr_pp)
np.save(os.path.join(NPY_DIR, "dice_slices_modelo13.npy"), dice_slices)
np.save(os.path.join(NPY_DIR, "iou_slices_modelo13.npy"),  iou_slices)

print("Saved in:", NPY_DIR)
print("- Y_val_modelo13.npy")
print("- preds_val_modelo13_bestthr_pp.npy")
print("- dice_slices_modelo13.npy")
print("- iou_slices_modelo13.npy")

# ------------------------------------------------------------------
# GENERATE PNGs WITH ORIGINAL + GT + PREDICTION
# ------------------------------------------------------------------
def save_example_pngs(X_val, Y_val, preds, out_dir, max_examples=20):
    n = min(max_examples, X_val.shape[0])
    for i in range(n):
        img = X_val[i, ..., 0]
        gt  = Y_val[i, ..., 0]
        prd = preds[i, ..., 0]

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        fig.suptitle(f"Prototype 13 - Slice {i}", fontsize=12)

        ax = axes[0]
        ax.imshow(img, cmap='gray')
        ax.set_title("Original")
        ax.axis('off')

        ax = axes[1]
        ax.imshow(img, cmap='gray')
        ax.imshow(gt, alpha=0.4)
        ax.set_title("Ground Truth")
        ax.axis('off')

        ax = axes[2]
        ax.imshow(img, cmap='gray')
        ax.imshow(prd, alpha=0.4)
        ax.set_title("Prediction Prototype 13")
        ax.axis('off')

        fname = os.path.join(out_dir, f"modelo13_slice_{i:03d}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close(fig)

    print(f"Saved {n} example PNGs in: {out_dir}")

save_example_pngs(X_val, Y_val, preds_val_bestthr_pp, PNG_DIR, max_examples=30)

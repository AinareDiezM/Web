#!/usr/bin/env python
import tensorflow as tf
import os
import numpy as np
import random
from glob import glob

from sklearn.model_selection import train_test_split

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# BASIC CONFIGURATION
# ------------------------------------------------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)

IMG_HEIGHT   = 256
IMG_WIDTH    = 256
IMG_CHANNELS = 1   # GRAYSCALE

# ------------------------------------------------------------------
# PATHS OF ADC DATASET
# ------------------------------------------------------------------
IMAGES_DIR = r"C:\Users\Ainare\Downloads\DATASET FINAL ADC-20251124T130843Z-1-001\DATASET FINAL ADC\Imagenes"
MASKS_DIR  = r"C:\Users\Ainare\Downloads\DATASET FINAL ADC-20251124T130843Z-1-001\DATASET FINAL ADC\Mascaras"

# ------------------------------------------------------------------
# FILE LIST (SEARCH IN SUBFOLDERS PER PATIENT)
# ------------------------------------------------------------------
# Searching all PNGs inside all subfolders
image_paths_raw = glob(os.path.join(IMAGES_DIR, "**", "*.png"), recursive=True)
mask_paths_raw  = glob(os.path.join(MASKS_DIR,  "**", "*.png"), recursive=True)

print(f"Images found: {len(image_paths_raw)}")
print(f"Masks found: {len(mask_paths_raw)}")

# ------------------------------------------------------------------
# MATCH IMAGES AND MASKS BY slice/mask PATTERN
# ------------------------------------------------------------------
# Dictionary: filename -> full_path for masks
mask_dict = {os.path.basename(p): p for p in mask_paths_raw}

image_paths = []
mask_paths  = []

for img_path in image_paths_raw:
    img_name = os.path.basename(img_path)
    # Replace 'slice' with 'mask' to build the expected mask name
    mask_name = img_name.replace("_slice_", "_mask_")  # 'p1_adc_mask_001.png'
    
    if mask_name in mask_dict:
        image_paths.append(img_path)
        mask_paths.append(mask_dict[mask_name])
    else:
        print(f"Mask not found for image {img_name} (expected {mask_name})")

n_images = len(image_paths)
print(f"Final number of image-mask pairs: {n_images}")

assert n_images > 0, "Could not match images and masks. Check the filenames."


# ------------------------------------------------------------------
# IMAGE AND MASK LOADING
# ------------------------------------------------------------------
X = np.zeros((n_images, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y = np.zeros((n_images, IMG_HEIGHT, IMG_WIDTH, 1),             dtype=np.float32)

print("Loading and resizing images and masks...")

for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):

    # ----- Image -----
    img = imread(img_path) 
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1) # (H,W) -> (H,W,1)
    elif img.ndim == 3 and img.shape[-1] > 1:
        # If it comes with 3 channels, take only one
        img = img[..., 0:1]

    # Resize for safety (even if already 256x256)
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # NORMALIZATION
    img = img / 255.0

    X[i] = img

    # ----- Mask -----
    mask = imread(mask_path)

    if mask.ndim == 3:
        mask = mask[..., 0]

    # Resize
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    # Binarize (if masks are 0 and 255 or >0)
    mask = (mask > 0).astype(np.float32)

    mask = np.expand_dims(mask, axis=-1)  # (H,W) -> (H,W,1)

    Y[i] = mask

print("Loading completed.")
print("X shape:", X.shape, "Y shape:", Y.shape)

# ------------------------------------------------------------------
# TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------------
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.1, random_state=seed
)

print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val:", X_val.shape, "Y_val:", Y_val.shape)

# ------------------------------------------------------------------
# QUICK VISUALIZATION OF ONE EXAMPLE
# ------------------------------------------------------------------
idx = random.randint(0, len(X_train)-1)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Image")
imshow(X_train[idx].squeeze(), cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Mask")
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
# CALLBACKS (EARLY STOP + TENSORBOARD + CHECKPOINT)
# ------------------------------------------------------------------
checkpointer = tf.keras.callbacks.ModelCheckpoint(
    'unet_adc_256_gray.h5',
    verbose=1,
    save_best_only=True
)

callbacks = [
    checkpointer,
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.TensorBoard(log_dir='logs_adc')
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
# SAVE ARRAYS TO COMPUTE METRICS
# ================================================================
np.save("X_val_adc.npy", X_val)            # validation images
np.save("Y_val_adc.npy", Y_val)            # Ground Truth masks
np.save("preds_val_adc.npy", preds_val_t)  # binarized predictions

print("Arrays saved for metrics:")
print(" - X_val_adc.npy")
print(" - Y_val_adc.npy")
print(" - preds_val_adc.npy")


# ================================================================
# SAVE 10 EXAMPLES AS PNG (IMAGE | GT | PRED)
# ================================================================
import os
import matplotlib.pyplot as plt

output_dir = "resultados_adc_png"
os.makedirs(output_dir, exist_ok=True)

N_SAVE = min(10, len(X_val))

for i in range(N_SAVE):
    # Original image
    plt.imsave(
        os.path.join(output_dir, f"img_{i:03d}.png"),
        X_val[i].squeeze(),
        cmap="gray"
    )
    # Ground Truth mask
    plt.imsave(
        os.path.join(output_dir, f"mask_gt_{i:03d}.png"),
        Y_val[i].squeeze(),
        cmap="gray"
    )
    # Predicted mask
    plt.imsave(
        os.path.join(output_dir, f"mask_pred_{i:03d}.png"),
        preds_val_t[i].squeeze(),
        cmap="gray"
    )

print(f"Saved {N_SAVE} examples as PNG in folder '{output_dir}'")





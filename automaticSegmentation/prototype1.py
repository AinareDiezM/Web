# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Fri Nov 21 13:09:27 2025)---
test_list_tour = [1, 2, 3, 4, 5]
test_dict_tour = {'a': 1, 'b': 2}
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir
%clear
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir
%debugcell -i 0 C:/Users/Ainare/.spyder-py3/temp.py

## ---(Fri Nov 21 15:09:22 2025)---
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir

## ---(Mon Nov 24 14:22:31 2025)---
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir

## ---(Mon Nov 24 15:03:51 2025)---
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir

## ---(Mon Nov 24 18:13:13 2025)---
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir

## ---(Mon Nov 24 18:33:32 2025)---
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir

## ---(Mon Nov 24 18:39:28 2025)---
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir

# ================================================================
# SAVE ARRAYS FOR METRIC COMPUTATION
# ================================================================
np.save("X_val_adc.npy", X_val)            # validation images
np.save("Y_val_adc.npy", Y_val)            # ground truth masks
np.save("preds_val_adc.npy", preds_val_t)  # binarized predictions

print("Arrays saved for metric computation:")
print("   - X_val_adc.npy")
print("   - Y_val_adc.npy")
print("   - preds_val_adc.npy")


# ================================================================
# SAVE 10 EXAMPLES AS PNG (IMAGE | GT | PREDICTION)
# ================================================================
import os
import matplotlib.pyplot as plt

output_dir = "adc_png_results"
os.makedirs(output_dir, exist_ok=True)

N_SAVE = min(10, len(X_val))  # Save up to 10 examples (or fewer if dataset is smaller)

for i in range(N_SAVE):
    # Original image
    plt.imsave(
        os.path.join(output_dir, f"img_{i:03d}.png"),
        X_val[i].squeeze(),
        cmap="gray"
    )
    # Ground truth mask
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

print(f" {N_SAVE} PNG examples saved in folder '{output_dir}'")


# ================================================================
# SAVE ARRAYS FOR METRIC COMPUTATION (REPEATED BLOCK)
# ================================================================
np.save("X_val_adc.npy", X_val)            # validation images
np.save("Y_val_adc.npy", Y_val)            # ground truth masks
np.save("preds_val_adc.npy", preds_val_t)  # binarized predictions

print("Arrays saved for metric computation:")
print("   - X_val_adc.npy")
print("   - Y_val_adc.npy")
print("   - preds_val_adc.npy")


# ================================================================
# SAVE 10 EXAMPLES AS PNG (IMAGE | GT | PREDICTION)
# ================================================================
import os
import matplotlib.pyplot as plt

output_dir = "adc_png_results"
os.makedirs(output_dir, exist_ok=True)

N_SAVE = min(10, len(X_val))  # Save up to 10 examples (or fewer if dataset is smaller)

for i in range(N_SAVE):
    # Original image
    plt.imsave(
        os.path.join(output_dir, f"img_{i:03d}.png"),
        X_val[i].squeeze(),
        cmap="gray"
    )
    # Ground truth mask
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

print(f"✔️ {N_SAVE} PNG examples saved in folder '{output_dir}'")


%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir

## ---(Wed Nov 26 09:33:45 2025)---
import tensorflow as tf
tf.__version__
import keras
keras.__version__
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir

## ---(Wed Nov 26 09:49:28 2025)---
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled1.py --wdir

## ---(Thu Nov 27 17:34:36 2025)---
%runfile C:/Users/Ainare/.spyder-py3/remove_empty.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled1.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled2.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled3.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled4.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/temp.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled5.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled6.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled7.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled8.py --wdir

## ---(Fri Nov 28 09:42:28 2025)---
%runfile 'C:/Users/Ainare/.spyder-py3/enhanced_unet.py' --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir
%runfile 'C:/Users/Ainare/.spyder-py3/EVALUATION_METRICS.py' --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled1.py --wdir
%runfile 'C:/Users/Ainare/.spyder-py3/EVALUATION_METRICS.py' --wdir

## ---(Tue Dec  2 17:16:09 2025)---
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir
%runfile 'C:/Users/Ainare/.spyder-py3/EVALUATION_METRICS.py' --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled1.py --wdir
%runfile C:/Users/Ainare/.spyder-py3/threshold_visualization.py --wdir
%runfile 'C:/Users/Ainare/.spyder-py3/EVALUATION_METRICS.py' --wdir

## ---(Wed Dec  3 11:19:51 2025)---
%runfile 'C:/Users/Ainare/.spyder-py3/optimal_model_with_postprocessing.py' --wdir
%runfile C:/Users/Ainare/.spyder-py3/untitled0.py --wdir


X_train, Y_train = load_dataset(
    images_dirs=[TRAIN_IMAGES_DIR, TRAIN_AUG_IMAGES_DIR],
    masks_dirs=[TRAIN_MASKS_DIR, TRAIN_AUG_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

X_val, Y_val = load_dataset(
    images_dirs=[VAL_IMAGES_DIR],
    masks_dirs=[VAL_MASKS_DIR],
    img_height=IMG_HEIGHT,
    img_width=IMG_WIDTH
)

np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_val.npy",   X_val)
np.save("Y_val.npy",   Y_val)

exit()

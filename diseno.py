# app.py
import os
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd

import base64
from io import BytesIO

# =========================================
# CONFIG – REPO PATHS (NO LOCAL C:\...)
# =========================================

# Root of the repo = folder where this app.py lives
REPO_ROOT = Path(__file__).resolve().parent

# Your repo folder that contains the ZIPs
PHOTOWEB_DIR = REPO_ROOT / "Photoweb"

# Where ZIPs will be extracted (works on Streamlit Cloud)
EXTRACT_DIR = REPO_ROOT / "data_extracted"
EXTRACT_DIR.mkdir(exist_ok=True)

# Map logical keys -> zip filenames (as they are in your repo)
ZIP_MAP = {
    "ORIGINAL": "ORIGINAL.zip",
    "GT_RAS_PNG_RECORTE": "GT_RAS_PNG_RECORTE.zip",
    "MEJOR SEMIAUTOMATICO": "MEJOR SEMIAUTOMATICO.zip",
    "PRUEBAAUTO - ROIM": "PRUEBAAUTO - ROIM.zip",
    "FIGS_ERRORMAPS": "FIGS_ERRORMAPS.zip",
    # Optional (only if you add these zips to the repo):
    # "PRUEBAAUTO_PROBS": "PRUEBAAUTO_PROBS.zip",
    # "PRUEBAAUTO - ROI": "PRUEBAAUTO - ROI.zip",
    # "CODE": "CODE.zip",
    # "lung_app_metrics": "lung_app_metrics.zip",
    # "Index": "Index.zip",
}

VALID_EXTS = (".png", ".jpg", ".jpeg")


def _looks_like_image_folder(p: Path) -> bool:
    if not p.is_dir():
        return False
    for ext in VALID_EXTS:
        if any(p.rglob(f"*{ext}")):
            return True
    return False


def ensure_unzipped(folder_key: str) -> Path:
    target_dir = EXTRACT_DIR / folder_key

    # If already extracted
    if target_dir.exists() and any(target_dir.iterdir()):
        subdirs = [p for p in target_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
        if len(subdirs) == 1:
            return subdirs[0]
        return target_dir

    zip_name = ZIP_MAP.get(folder_key)
    if zip_name is None:
        raise FileNotFoundError(f"No ZIP mapping found for key: {folder_key}")

    zip_path = PHOTOWEB_DIR / zip_name
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found in repo: {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    # Return inner folder if it exists (common zip layout)
    subdirs = [p for p in target_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if len(subdirs) == 1:
        return subdirs[0]

    return target_dir


    zip_name = ZIP_MAP.get(folder_key)
    if zip_name is None:
        raise FileNotFoundError(f"No ZIP mapping found for key: {folder_key}")

    zip_path = PHOTOWEB_DIR / zip_name
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found in repo: {zip_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    # Resolve nested folder case after extraction
    if _looks_like_image_folder(target_dir) or any(target_dir.glob("*.npy")):
        return target_dir

    subdirs = [p for p in target_dir.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return subdirs[0]

    return target_dir

st.sidebar.markdown("### DEBUG (paths)")
st.sidebar.write("BASE_ORIGINAL =", BASE_ORIGINAL)
st.sidebar.write("Exists? ", os.path.isdir(BASE_ORIGINAL))
if os.path.isdir(BASE_ORIGINAL):
    st.sidebar.write("Sample files:", sorted(os.listdir(BASE_ORIGINAL))[:10])
    
# Extracted folders (as strings for os.path.* compatibility)
BASE_ORIGINAL = str(ensure_unzipped("ORIGINAL") / "ORIGINAL")
BASE_MANUAL     = str(ensure_unzipped("GT_RAS_PNG_RECORTE") / "GT_RAS_PNG_RECORTE")
BASE_SEMI       = str(ensure_unzipped("MEJOR SEMIAUTOMATICO") / "MEJOR SEMIAUTOMATICO")
BASE_AUTO_MASKS = str(ensure_unzipped("PRUEBAAUTO - ROIM") / "PRUEBAAUTO - ROIM")
ERRORMAPS_DIR   = str(ensure_unzipped("FIGS_ERRORMAPS") / "FIGS_ERRORMAPS")




# Optional folders (only if you add zips)
BASE_AUTO_PROBS = None
BASE_AUTO_ROI = None

# Optional: if you add a standalone image (not in zip) under photoweb/
SCENE_IMG_PATH = PHOTOWEB_DIR / "2025-10-08-Scene.png"
SCENE_IMG_PATH = str(SCENE_IMG_PATH) if SCENE_IMG_PATH.is_file() else None

# ---------------------------------------------------------
# LANDING PAGE (Project overview) – 4 images to display
# ---------------------------------------------------------
OVERVIEW_IMG_1 = os.path.join(BASE_ORIGINAL, "SCC_P4.png")
OVERVIEW_IMG_2 = os.path.join(BASE_ORIGINAL, "ADC_P3.png")
OVERVIEW_IMG_3 = SCENE_IMG_PATH  # can be None if not in repo
OVERVIEW_IMG_4 = os.path.join(BASE_SEMI, "ADC_P1.png")

# Patients = file stems like ADC_P1, ADC_P2, ..., SCC_P5
ADC_PATIENTS = [f"ADC_P{i}" for i in range(1, 6)]
SCC_PATIENTS = [f"SCC_P{i}" for i in range(1, 6)]

# =========================================
# OPTIONAL: METRICS / CODE PATHS
# (Will work only if you add these folders/zips to the repo)
# =========================================

def resolve_optional_dir(folder_key: str, fallback_rel: str):
    """
    Prefer unzipped zip if present in ZIP_MAP, else prefer repo folder fallback_rel if exists, else return None.
    """
    if folder_key in ZIP_MAP:
        try:
            return str(ensure_unzipped(folder_key))
        except Exception:
            return None
    p = REPO_ROOT / fallback_rel
    return str(p) if p.is_dir() else None

METRICS_DIR = resolve_optional_dir("lung_app_metrics", "lung_app_metrics")
CODE_DIR = resolve_optional_dir("CODE", "CODE")

MODEL_METRICS_FILES = {}
if METRICS_DIR:
    MODEL_METRICS_FILES = {
        "Model 1 – training without augmentation": {
            "dice": os.path.join(METRICS_DIR, "dice_model1.npy"),
            "iou":  os.path.join(METRICS_DIR, "iou_model1.npy"),
        },
        "Model 2 – training with augmentation": {
            "dice": os.path.join(METRICS_DIR, "dice_model2.npy"),
            "iou":  os.path.join(METRICS_DIR, "iou_model2.npy"),
        },
        "Model 3 – BCE + Dice loss": {
            "dice": os.path.join(METRICS_DIR, "dice_model3.npy"),
            "iou":  os.path.join(METRICS_DIR, "iou_model3.npy"),
        },
    }

MODELS_7_9_10_13 = {}
if CODE_DIR:
    MODELS_7_9_10_13 = {
        "Model 7 – U-Net (BCE + Dice + Focal, thr=0.90 + post-processing)": {
            "y_true": os.path.join(CODE_DIR, "Y_val_modelo_bce_dice_focal.npy"),
            "y_pred": os.path.join(CODE_DIR, "preds_val_modelo_bce_dice_focal_bestthr_pp.npy"),
            "threshold": None,
        },
        "Model 9 – Final U-Net (2-phase + oversampling, thr=0.90 + post-processing)": {
            "y_true": os.path.join(CODE_DIR, "Y_val_modelo9_bce_dice_focal.npy"),
            "y_pred": os.path.join(CODE_DIR, "preds_val_modelo9_bce_dice_focal_bestthr_pp.npy"),
            "threshold": None,
        },
        "Model 10 – U-Net (CLAHE + regionprops, thr=0.90 + post-processing)": {
            "y_true": os.path.join(CODE_DIR, "Y_VAL_MODEL10.npy"),
            "y_pred": os.path.join(CODE_DIR, "PREDS_VAL_MODEL10_BEST_PP.npy"),
            "threshold": None,
        },
        "Model 13 – ResUNet (Focal Tversky, thr=0.90 + post-processing)": {
            "y_true": os.path.join(CODE_DIR, "Y_val_modelo13.npy"),
            "y_pred": os.path.join(CODE_DIR, "preds_val_modelo13_bestthr_pp.npy"),
            "threshold": None,
        },
    }

# =========================================
# HELPER FUNCTIONS
# =========================================

def load_image_safe(path: str) -> Image.Image:
    """Load an image and convert 16-bit or unusual modes into something Streamlit can display."""
    img = Image.open(path)
    if img.mode == "I;16":
        img = img.point(lambda i: i * (1 / 256)).convert("L")
    else:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
    return img

def compute_simple_descriptors(mask_img: Image.Image, base_gray_img: Image.Image):
    if mask_img is None or base_gray_img is None:
        return None

    gt_arr = np.array(mask_img.convert("L"))
    base_arr = np.array(base_gray_img.convert("L"))

    mask_bin = gt_arr > 0
    if mask_bin.sum() == 0:
        return {
            "mean_intensity": np.nan,
            "area": 0,
            "perimeter": 0,
            "compactness": np.nan,
            "centroid_row": np.nan,
            "centroid_col": np.nan
        }

    area = int(mask_bin.sum())
    coords = np.column_stack(np.nonzero(mask_bin))
    centroid_row = float(coords[:, 0].mean())
    centroid_col = float(coords[:, 1].mean())

    up = np.roll(mask_bin, -1, axis=0); up[-1, :] = False
    down = np.roll(mask_bin, 1, axis=0); down[0, :] = False
    left = np.roll(mask_bin, -1, axis=1); left[:, -1] = False
    right = np.roll(mask_bin, 1, axis=1); right[:, 0] = False

    neighbors_bg = (~up) | (~down) | (~left) | (~right)
    boundary = mask_bin & neighbors_bg
    perimeter = int(boundary.sum())

    compactness = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else np.nan
    tumour_vals = base_arr[mask_bin]
    mean_intensity = float(tumour_vals.mean()) if tumour_vals.size > 0 else np.nan

    return {
        "mean_intensity": mean_intensity,
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness,
        "centroid_row": centroid_row,
        "centroid_col": centroid_col
    }

@st.cache_data
def list_slices_for_patient(patient: str):
    if not os.path.isdir(BASE_ORIGINAL):
        return []
    slice_files = []
    for f in os.listdir(BASE_ORIGINAL):
        name, ext = os.path.splitext(f)
        if ext.lower() in VALID_EXTS and name.startswith(patient):
            slice_files.append(f)
    return sorted(slice_files)

def build_original_path(patient: str, slice_name: str) -> str:
    return os.path.join(BASE_ORIGINAL, slice_name)

def build_manual_path(patient: str, slice_name: str) -> str:
    return os.path.join(BASE_MANUAL, slice_name)

def build_difference_map(gt_mask_img: Image.Image, pred_mask_img: Image.Image):
    gt_arr = np.array(gt_mask_img.convert("L"))
    pred_arr = np.array(pred_mask_img.convert("L"))

    if gt_arr.shape != pred_arr.shape:
        pred_arr = np.array(
            pred_mask_img.convert("L").resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
        )

    gt_bin = gt_arr > 0
    pred_bin = pred_arr > 0

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    h, w = gt_bin.shape
    diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    diff_rgb[fp] = [255, 0, 0]
    diff_rgb[fn] = [0, 0, 255]
    diff_rgb[tp] = [0, 255, 0]

    return Image.fromarray(diff_rgb)

def build_semi_path(patient: str, slice_name: str) -> str:
    return os.path.join(BASE_SEMI, slice_name)

def build_auto_mask_path(patient: str, slice_name: str):
    base_name, _ = os.path.splitext(slice_name)

    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = os.path.join(BASE_AUTO_MASKS, base_name + ext)
        if os.path.isfile(candidate):
            return candidate

    if not os.path.isdir(BASE_AUTO_MASKS):
        return None

    for fname in os.listdir(BASE_AUTO_MASKS):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in VALID_EXTS:
            continue
        if name.startswith(base_name):
            return os.path.join(BASE_AUTO_MASKS, fname)

    return None

def build_auto_prob_path(patient: str, slice_name: str):
    if not BASE_AUTO_PROBS:
        return None
    return os.path.join(BASE_AUTO_PROBS, slice_name)

def build_auto_roi_path(patient: str, slice_name: str):
    if not BASE_AUTO_ROI:
        return None
    return os.path.join(BASE_AUTO_ROI, slice_name)

def load_metrics_array(model_label: str, metric: str):
    files = MODEL_METRICS_FILES.get(model_label, {})
    path = files.get(metric)
    if path is None or not os.path.isfile(path):
        return None
    return np.load(path)

def list_error_maps_for_patient(patient: str):
    if not os.path.isdir(ERRORMAPS_DIR):
        return []
    files = []
    for fname in os.listdir(ERRORMAPS_DIR):
        f_lower = fname.lower()
        if f_lower.endswith((".png", ".jpg", ".jpeg")) and patient.lower() in f_lower:
            files.append(os.path.join(ERRORMAPS_DIR, fname))
    return sorted(files)

def ensure_3d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 4:
        arr = arr[..., 0]
    return arr

def binarise(arr):
    return (arr > 0).astype(np.uint8)

def compute_dice_iou(y_true, y_pred):
    y_true = ensure_3d(y_true)
    y_pred = ensure_3d(y_pred)
    if y_true.shape != y_pred.shape:
        return None, None

    dices, ious = [], []
    for i in range(y_true.shape[0]):
        yt = binarise(y_true[i])
        yp = binarise(y_pred[i])
        inter = np.sum(yt * yp)
        union = np.sum(yt) + np.sum(yp)
        dice = (2.0 * inter) / (union + 1e-7)
        iou = inter / (np.sum(yt) + np.sum(yp) - inter + 1e-7)
        dices.append(dice)
        ious.append(iou)
    return np.array(dices), np.array(ious)

def pil_to_base64_png(pil_img: Image.Image) -> str:
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =========================================
# PAGE CONFIG & STYLING
# =========================================

st.set_page_config(page_title="Lung Cancer Segmentation Viewer", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; }

:root {
    --primary-color: #003865;
    --secondary-color: #005f9e;
    --accent-color: #f4b000;
    --bg-light: #f7f9fc;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-light);
    border-right: 1px solid #d0d4e4;
}
[data-testid="stSidebar"] * { font-size: 17px !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: var(--primary-color);
}

[data-testid="stSidebar"] label {
    font-size: 17px !important;
    font-weight: 600 !important;
}

h1 {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: var(--primary-color);
}

h2, h3 {
    font-weight: 600 !important;
    color: var(--secondary-color);
}

[data-testid="stMetricValue"] { font-size: 22px !important; }

.main .block-container {
    max-width: 1300px;
    padding-top: 1.5rem;
}

button[data-baseweb="tab"] {
    font-weight: 500;
    color: var(--primary-color);
}
button[data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 2px solid var(--accent-color);
}
</style>
""",
    unsafe_allow_html=True
)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("Lung Cancer Segmentation Viewer")
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

section = st.sidebar.radio("Go to section:", ["Project overview", "Patient exploration", "Model comparison"])
st.sidebar.markdown("---")

# =========================================
# SECTION: PROJECT OVERVIEW
# =========================================

if section == "Project overview":
    st.markdown(
        """
    <div class="hero-author">
        Ainare Díez Madariaga<br>
        <span style="font-size:16px; font-weight:400;">
        Biomedical Engineering – University of Deusto
        </span>
    </div>
    """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .hero-box { background-color: #e9f2ff; padding: 2.2rem 2.4rem; border-radius: 16px; margin-bottom: 2rem; }
        .hero-title { font-size: 54px !important; font-weight: 800 !important; color: #003865; line-height: 1.1; margin-bottom: 2.8rem !important; }
        .hero-subtext { font-size: 20px !important; color: #3a4a58; margin-top: 0.5rem; margin-bottom: 1.5rem; line-height: 1.6; }

        .hero-img-box {
            width: 100%;
            aspect-ratio: 1 / 1;
            overflow: hidden;
            border-radius: 12px;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .hero-img-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }
        .hero-grid-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }
        </style>
    """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="hero-box">', unsafe_allow_html=True)

    col_hero_text, col_hero_images = st.columns([2, 1.3])

    with col_hero_text:
        st.markdown(
            '<div class="hero-title">Multimodal Imaging and Computational Analysis for NSCLC Subtype Differentiation</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            """
        <p class="hero-subtext">
        This project presents an integrated computational framework for the analysis of
        non-small cell lung cancer (NSCLC) using multimodal medical imaging. The pipeline
        combines MRI-based tumour segmentation, multimodal image processing, radiomic
        feature exploration, and deep-learning–based methods to support the differentiation
        of NSCLC subtypes.
        <br><br>
        The proposed framework incorporates manual, semi-automatic, and fully automatic
        segmentation strategies, enabling a systematic comparison of reproducibility,
        tumour characterisation, and clinical interpretability across different levels of
        automation.
        </p>
        """,
            unsafe_allow_html=True
        )

    with col_hero_images:

        def hero_img_html(path: str) -> str:
            if not path or (not os.path.isfile(path)):
                return "<div style='padding:10px; border:1px dashed #ccc; border-radius:12px;'>Missing image</div>"
            img = load_image_safe(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_b64 = pil_to_base64_png(img)
            return f"""
            <div class="hero-img-box">
                <img src="data:image/png;base64,{img_b64}" />
            </div>
            """

        st.markdown(
            f"""
            <div class="hero-grid-row">
                {hero_img_html(OVERVIEW_IMG_1)}
                {hero_img_html(OVERVIEW_IMG_2)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="hero-grid-row">
                {hero_img_html(OVERVIEW_IMG_3)}
                {hero_img_html(OVERVIEW_IMG_4)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

# =========================================
# SECTION: PATIENT EXPLORATION
# =========================================
elif section == "Patient exploration":

    st.markdown(
        """
        <h2 style="margin-bottom:0.2rem;">Patient exploration</h2>
        <p style="color:#4a4a4a;margin-top:0.2rem;margin-bottom:1.2rem;"></p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            width: 100%;
            height: 8px;
            margin-top: -10px;
            margin-bottom: 18px;
            background-color: #e9f2ff;
            border-radius: 6px;">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.subheader("Patient selection")

    tumour_type = st.sidebar.selectbox("Tumour type:", ["Adenocarcinoma (ADC)", "Squamous cell carcinoma (SCC)"])
    patient = st.sidebar.selectbox("Select patient:", ADC_PATIENTS if "Adenocarcinoma" in tumour_type else SCC_PATIENTS)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation layers")

    show_gt = st.sidebar.checkbox("Show manual segmentation (ground truth)", True)
    show_semi = st.sidebar.checkbox("Show semi-automatic segmentation", True)
    show_auto = st.sidebar.checkbox("Show automatic segmentation (U-Net)", True)

    st.write(f"Selected case: **{patient}** – {tumour_type}")

    st.markdown(
        """
<div style="
    background-color: #f7f9fc;
    border-left: 4px solid #003865;
    padding: 0.8rem 1rem;
    margin-top: 0.6rem;
    margin-bottom: 1.2rem;
    border-radius: 6px;
    color: #3a4a58;
    font-size: 15px;
">
<strong>Section overview</strong><br>
In this section, the selected patient case can be explored in detail at slice level.
The user can visualise the original MRI image alongside manual, semi-automatic, and
automatic tumour segmentations, inspect simple radiomic descriptors computed per method,
and analyse segmentation differences. This view is intended to
support qualitative interpretation of tumour morphology and segmentation behaviour on an
individual-case basis.
</div>
""",
        unsafe_allow_html=True
    )

    slice_files = list_slices_for_patient(patient)

    if not slice_files:
        st.warning(
            "No images found for this patient in the ORIGINAL folder.\n\n"
            f"Checked folder: `{BASE_ORIGINAL}` with prefix `{patient}`"
        )
    else:
        if len(slice_files) == 1:
            slice_name = slice_files[0]
        else:
            idx = st.slider("Slice index", 0, len(slice_files) - 1, 0)
            slice_name = slice_files[idx]
            st.markdown(f"Current file: `{slice_name}`")

        orig_path = build_original_path(patient, slice_name)
        manual_path = build_manual_path(patient, slice_name)
        semi_path = build_semi_path(patient, slice_name)
        auto_mask_path = build_auto_mask_path(patient, slice_name)
        auto_prob_path = build_auto_prob_path(patient, slice_name)

        st.markdown("### Original MRI")

        base_img = load_image_safe(orig_path)
        base_img_gray = base_img.convert("L")

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(base_img, use_container_width=True)

        st.markdown("### Segmentation masks")

        mask_gt_img = load_image_safe(manual_path) if os.path.isfile(manual_path) else None
        mask_semi_img = load_image_safe(semi_path) if os.path.isfile(semi_path) else None
        mask_auto_img = load_image_safe(auto_mask_path) if (auto_mask_path and os.path.isfile(auto_mask_path)) else None

        col_gt, col_semi, col_auto = st.columns(3)

        with col_gt:
            st.markdown("<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Manual (ground truth)</div>", unsafe_allow_html=True)
            if show_gt and mask_gt_img is not None:
                st.image(mask_gt_img, use_container_width=True, clamp=True)
            elif show_gt:
                st.info("No manual mask available for this slice.")
            else:
                st.info("Layer disabled.")

        with col_semi:
            st.markdown("<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Semi-automatic</div>", unsafe_allow_html=True)
            if show_semi and mask_semi_img is not None:
                st.image(mask_semi_img, use_container_width=True, clamp=True)
            elif show_semi:
                st.info("No semi-automatic mask available for this slice.")
            else:
                st.info("Layer disabled.")

        with col_auto:
            st.markdown("<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Automatic (U-Net)</div>", unsafe_allow_html=True)
            if show_auto and mask_auto_img is not None:
                st.image(mask_auto_img, use_container_width=True, clamp=True)
            elif show_auto:
                st.info("No automatic mask available for this slice.")
            else:
                st.info("Layer disabled.")

        st.markdown("---")
        st.subheader("Radiomics preview per method")

        def fmt(v, nd=1):
            return "N/A" if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), nd)

        def fmt_centroid(r, c):
            if r is None or c is None or (isinstance(r, float) and np.isnan(r)) or (isinstance(c, float) and np.isnan(c)):
                return "N/A"
            return f"({r:.1f}, {c:.1f})"

        data = []
        d_gt = compute_simple_descriptors(mask_gt_img, base_img_gray) if mask_gt_img is not None else None
        d_semi = compute_simple_descriptors(mask_semi_img, base_img_gray) if mask_semi_img is not None else None
        d_auto = compute_simple_descriptors(mask_auto_img, base_img_gray) if mask_auto_img is not None else None

        rows = [("Manual (GT)", d_gt), ("Semi-automatic", d_semi), ("Automatic (U-Net)", d_auto)]

        for label, d in rows:
            if d is None:
                data.append({
                    "Method": label,
                    "Mean intensity (a.u.)": "N/A",
                    "Tumour area (px)": "N/A",
                    "Perimeter (px)": "N/A",
                    "Compactness (4πA/P²)": "N/A",
                    "Centroid (row, col)": "N/A",
                })
            else:
                data.append({
                    "Method": label,
                    "Mean intensity (a.u.)": fmt(d["mean_intensity"], 1),
                    "Tumour area (px)": int(d["area"]),
                    "Perimeter (px)": int(d["perimeter"]),
                    "Compactness (4πA/P²)": fmt(d["compactness"], 3),
                    "Centroid (row, col)": fmt_centroid(d["centroid_row"], d["centroid_col"]),
                })

        df = pd.DataFrame(data)

        st.markdown(
            """
        **Radiomics descriptors (slice-level):**

        - **Mean intensity (a.u.)**: Average MRI signal intensity within the segmented tumour region.
        - **Tumour area (px)**: Number of pixels classified as tumour in the selected slice.
        - **Perimeter (px)**: Approximate length of the tumour boundary, estimated using 4-connectivity.
        - **Compactness (4πA/P²)**: Shape descriptor quantifying how close the tumour is to a circular geometry.
        - **Centroid (row, col)**: Geometric centre of the segmented tumour region in image coordinates.
        """
        )

        st.dataframe(df, use_container_width=True, hide_index=True)

        missing = []
        if mask_gt_img is None: missing.append("Manual (GT)")
        if mask_semi_img is None: missing.append("Semi-automatic")
        if mask_auto_img is None: missing.append("Automatic (U-Net)")
        if missing:
            st.info("Missing masks for this slice: " + ", ".join(missing))

        st.markdown("---")
        st.subheader("GT vs U-Net differences (FP / FN / TP)")

        if (mask_gt_img is None) or (mask_auto_img is None) or (mask_semi_img is None):
            st.info("GT + masks are required to compute the difference maps.")
        else:
            col_auto, col_semi, col_text = st.columns([1, 1, 1.2])

            with col_auto:
                st.markdown("**GT vs U-Net**")
                diff_auto = build_difference_map(mask_gt_img, mask_auto_img)
                st.image(diff_auto, width=380)

            with col_semi:
                st.markdown("**GT vs Semi-automatic**")
                diff_semi = build_difference_map(mask_gt_img, mask_semi_img)
                st.image(diff_semi, width=380)

            with col_text:
                st.markdown(
                    """
                <strong>Colour legend:</strong>
                <ul>
                    <li><span style="color:red; font-weight:600;">Red</span>: False positives</li>
                    <li><span style="color:blue; font-weight:600;">Blue</span>: False negatives</li>
                    <li><span style="color:green; font-weight:600;">Green</span>: True positives</li>
                </ul>

                <p style="font-size: 14.5px; color: #4a5a68; margin-top: 0.4rem; line-height: 1.45;">
                <strong>Interpretation.</strong><br>
                These maps compare the manual ground truth with semi-automatic and U-Net-based
                segmentations at pixel level.
                </p>
                """,
                    unsafe_allow_html=True
                )

# =========================================
# SECTION: MODEL COMPARISON
# =========================================
elif section == "Model comparison":

    st.title("Model comparison")

    if not MODELS_7_9_10_13:
        st.warning(
            "Model comparison is disabled because CODE_DIR was not found in the repo.\n\n"
            "To enable it, add a CODE folder or CODE.zip to your repository (and map it in ZIP_MAP)."
        )
        st.stop()

    st.markdown(
        """
    This section compares the final candidate models selected for the demo.
    It reports global performance, metric distributions and a ranking table
    to support the selection of the most reliable segmentation configuration.
    """
    )

    with st.expander("Model descriptions"):
        st.markdown(
            """
        **Model 7 – U-Net with composite loss and post-processing**  
        Composite loss (BCE + Dice + Focal), threshold=0.90, post-processing (largest component).

        **Model 9 – Final U-Net (two-phase training with oversampling)**  
        Two-phase training + oversampling for small tumours, threshold=0.90, post-processing.

        **Model 10 – U-Net with enhanced preprocessing (CLAHE + region-based refinement)**  
        Adds CLAHE preprocessing and refinement.

        **Model 13 – ResUNet with Focal Tversky loss**  
        Residual architecture variant with Focal Tversky loss.
        """
        )

    def safe_load_npy(path: str):
        if path and os.path.isfile(path):
            try:
                return np.load(path, allow_pickle=False)
            except Exception:
                return None
        return None

    def dice_np(y_true, y_pred, smooth=1e-6):
        y_true = (y_true > 0).astype(np.uint8).ravel()
        y_pred = (y_pred > 0).astype(np.uint8).ravel()
        inter = np.sum(y_true * y_pred)
        return (2.0 * inter + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)

    def iou_np(y_true, y_pred, smooth=1e-6):
        y_true = (y_true > 0).astype(np.uint8).ravel()
        y_pred = (y_pred > 0).astype(np.uint8).ravel()
        inter = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - inter
        return (inter + smooth) / (union + smooth)

    def compute_metrics_from_arrays(y_arr, pred_arr):
        if y_arr is None or pred_arr is None:
            return None, None
        if y_arr.ndim == 4:
            y_arr = y_arr[..., 0]
        if pred_arr.ndim == 4:
            pred_arr = pred_arr[..., 0]

        n = min(len(y_arr), len(pred_arr))
        dice_s = np.zeros(n, dtype=np.float32)
        iou_s = np.zeros(n, dtype=np.float32)

        for i in range(n):
            dice_s[i] = dice_np(y_arr[i], pred_arr[i])
            iou_s[i] = iou_np(y_arr[i], pred_arr[i])

        return dice_s, iou_s

    model_labels = list(MODELS_7_9_10_13.keys())
    selected_model = st.selectbox("Select model:", model_labels)

    tab1, tab2, tab3 = st.tabs(["Summary metrics", "Metric distributions", "Ranking table (7 vs 9 vs 10 vs 13)"])

    cfg = MODELS_7_9_10_13[selected_model]

    # Load arrays
    Y_val = safe_load_npy(cfg["y_true"])
    preds_val = safe_load_npy(cfg["y_pred"])

    dice_arr, iou_arr = (None, None)
    if Y_val is not None and preds_val is not None:
        dice_arr, iou_arr = compute_dice_iou(Y_val, preds_val)

    with tab1:
        st.subheader("Global performance summary")
        if dice_arr is None or iou_arr is None:
            st.error("Metrics could not be loaded/computed. Check that CODE data exists in the repo.")
            st.write("Paths checked:", cfg)
        else:
            dice_mean = float(np.mean(dice_arr))
            dice_std = float(np.std(dice_arr))
            iou_mean = float(np.mean(iou_arr))
            iou_std = float(np.std(iou_arr))
            n = int(min(len(dice_arr), len(iou_arr)))

            c1, c2, c3 = st.columns([1, 1, 1.2])
            c1.metric("Dice (mean ± std)", f"{dice_mean:.3f} ± {dice_std:.3f}")
            c2.metric("IoU  (mean ± std)", f"{iou_mean:.3f} ± {iou_std:.3f}")
            c3.metric("Evaluated slices", f"{n}")

    with tab2:
        st.subheader("Metric distributions")
        if dice_arr is None or iou_arr is None:
            st.warning("No metric arrays available for the selected model.")
        else:
            colA, colB = st.columns(2)
            with colA:
                fig_dice = px.histogram(x=dice_arr, nbins=20, title="Dice distribution")
                fig_dice.update_layout(xaxis_title="Dice", yaxis_title="Count")
                st.plotly_chart(fig_dice, use_container_width=True)

            with colB:
                fig_iou = px.histogram(x=iou_arr, nbins=20, title="IoU distribution")
                fig_iou.update_layout(xaxis_title="IoU", yaxis_title="Count")
                st.plotly_chart(fig_iou, use_container_width=True)

            st.markdown("---")
            fig_line = px.line(x=np.arange(len(dice_arr)), y=dice_arr, title="Dice per slice")
            fig_line.update_layout(xaxis_title="Slice index", yaxis_title="Dice")
            st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        st.subheader("Model ranking (Models 7, 9, 10 and 13)")
        rows = []
        for model_name, cfg_rank in MODELS_7_9_10_13.items():
            Y_rank = safe_load_npy(cfg_rank["y_true"])
            P_rank = safe_load_npy(cfg_rank["y_pred"])
            if Y_rank is None or P_rank is None:
                rows.append({"Model": model_name, "Dice mean": None, "Dice std": None, "IoU mean": None, "IoU std": None, "N slices": None})
                continue
            d_rank, i_rank = compute_metrics_from_arrays(Y_rank, P_rank)
            if d_rank is None or len(d_rank) == 0:
                rows.append({"Model": model_name, "Dice mean": None, "Dice std": None, "IoU mean": None, "IoU std": None, "N slices": None})
                continue
            rows.append({
                "Model": model_name,
                "Dice mean": round(float(d_rank.mean()), 4),
                "Dice std": round(float(d_rank.std()), 4),
                "IoU mean": round(float(i_rank.mean()), 4),
                "IoU std": round(float(i_rank.std()), 4),
                "N slices": int(len(d_rank))
            })

        df_rank = pd.DataFrame(rows).sort_values(by="Dice mean", ascending=False, na_position="last").reset_index(drop=True)
        st.dataframe(df_rank, use_container_width=True, hide_index=True)

        if df_rank["Dice mean"].notna().sum() >= 1:
            best_model = df_rank.iloc[0]["Model"]
            st.success(f"Best model based on mean Dice: **{best_model}**")

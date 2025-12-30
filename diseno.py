# app.py
import os
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

REPO_ROOT = Path(__file__).resolve().parent

# Folder in your repo that contains the datasets
PHOTOWEB_DIR = REPO_ROOT / "Photoweb"

# Data folders inside Photoweb (direct PNGs inside)
BASE_ORIGINAL_DIR   = PHOTOWEB_DIR / "ORIGINAL"
BASE_MANUAL_DIR     = PHOTOWEB_DIR / "GT_RAS_PNG_RECORTE"
BASE_SEMI_DIR       = PHOTOWEB_DIR / "MEJOR SEMIAUTOMATICO"
BASE_AUTO_MASKS_DIR = PHOTOWEB_DIR / "PRUEBAAUTO - ROIM"
ERRORMAPS_DIR_PATH  = PHOTOWEB_DIR / "FIGS_ERRORMAPS"

# Convert to strings for os.path compatibility
BASE_ORIGINAL   = str(BASE_ORIGINAL_DIR)
BASE_MANUAL     = str(BASE_MANUAL_DIR)
BASE_SEMI       = str(BASE_SEMI_DIR)
BASE_AUTO_MASKS = str(BASE_AUTO_MASKS_DIR)
ERRORMAPS_DIR   = str(ERRORMAPS_DIR_PATH)

# Optional folders
BASE_AUTO_PROBS = None
BASE_AUTO_ROI = None

VALID_EXTS = (".png", ".jpg", ".jpeg")

# Optional: standalone image for landing page
SCENE_IMG_PATH = PHOTOWEB_DIR / "2025-10-08-Scene.png"
SCENE_IMG_PATH = str(SCENE_IMG_PATH) if SCENE_IMG_PATH.is_file() else None

# ---------------------------------------------------------
# LANDING PAGE (Project overview) – 4 images to display
# ---------------------------------------------------------
OVERVIEW_IMG_1 = os.path.join(BASE_ORIGINAL, "SCC_P4.png")
OVERVIEW_IMG_2 = os.path.join(BASE_ORIGINAL, "ADC_P3.png")
OVERVIEW_IMG_3 = SCENE_IMG_PATH  # can be None
OVERVIEW_IMG_4 = os.path.join(BASE_SEMI, "ADC_P1.png")

# Patients
ADC_PATIENTS = [f"ADC_P{i}" for i in range(1, 6)]
SCC_PATIENTS = [f"SCC_P{i}" for i in range(1, 6)]

# =========================================
# METRICS / MODEL COMPARISON PATHS
# =========================================

def resolve_optional_dir_case_insensitive(*candidates: str):
    """
    Find a directory in REPO_ROOT matching any of the candidate names.
    Works on Streamlit Cloud/Linux (case sensitive) and Windows (case insensitive).
    Returns str path or None.
    """
    # 1) Direct check (fast path)
    for name in candidates:
        p = REPO_ROOT / name
        if p.is_dir():
            return str(p)

    # 2) Case-insensitive search in repo root
    try:
        entries = [p for p in REPO_ROOT.iterdir() if p.is_dir()]
        name_map = {p.name.lower(): p for p in entries}
        for name in candidates:
            hit = name_map.get(name.lower())
            if hit and hit.is_dir():
                return str(hit)
    except Exception:
        pass

    return None

# Your .npy folder is called Metrics in GitHub (but we also accept CODE/metrics)
METRICS_NPY_DIR = resolve_optional_dir_case_insensitive("Metrics", "metrics", "CODE", "code")

# Build the model config only if the folder exists
MODELS_7_9_10_13 = {}
if METRICS_NPY_DIR:
    MODELS_7_9_10_13 = {
        "Model 7 – U-Net (BCE + Dice + Focal, thr=0.90 + post-processing)": {
            "y_true": os.path.join(METRICS_NPY_DIR, "Y_val_modelo_bce_dice_focal.npy"),
            "y_pred": os.path.join(METRICS_NPY_DIR, "preds_val_modelo_bce_dice_focal_bestthr_pp.npy"),
            "threshold": None,
        },
        "Model 9 – Final U-Net (2-phase + oversampling, thr=0.90 + post-processing)": {
            "y_true": os.path.join(METRICS_NPY_DIR, "Y_val_modelo9_bce_dice_focal.npy"),
            "y_pred": os.path.join(METRICS_NPY_DIR, "preds_val_modelo9_bce_dice_focal_bestthr_pp.npy"),
            "threshold": None,
        },
        "Model 10 – U-Net (CLAHE + regionprops, thr=0.90 + post-processing)": {
            "y_true": os.path.join(METRICS_NPY_DIR, "Y_VAL_MODEL10.npy"),
            "y_pred": os.path.join(METRICS_NPY_DIR, "PREDS_VAL_MODEL10_BEST_PP.npy"),
            "threshold": None,
        },
        "Model 13 – ResUNet (Focal Tversky, thr=0.90 + post-processing)": {
            "y_true": os.path.join(METRICS_NPY_DIR, "Y_val_modelo13.npy"),
            "y_pred": os.path.join(METRICS_NPY_DIR, "preds_val_modelo13_bestthr_pp.npy"),
            "threshold": None,
        },
    }

# =========================================
# HELPER FUNCTIONS
# =========================================

def load_image_safe(path: str) -> Image.Image:
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
    files = []
    for f in os.listdir(BASE_ORIGINAL):
        name, ext = os.path.splitext(f)
        if ext.lower() in VALID_EXTS and name.startswith(patient):
            files.append(f)
    return sorted(files)

def build_original_path(patient: str, slice_name: str) -> str:
    return os.path.join(BASE_ORIGINAL, slice_name)

def build_manual_path(patient: str, slice_name: str) -> str:
    return os.path.join(BASE_MANUAL, slice_name)

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

# Optional: quick debug (safe)
with st.sidebar.expander("Debug paths"):
    st.write("REPO_ROOT:", str(REPO_ROOT))
    st.write("Photoweb exists:", PHOTOWEB_DIR.is_dir())
    st.write("Metrics folder resolved:", METRICS_NPY_DIR)
    if METRICS_NPY_DIR and os.path.isdir(METRICS_NPY_DIR):
        st.write("Sample Metrics files:", sorted(os.listdir(METRICS_NPY_DIR))[:10])

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

    st.sidebar.subheader("Patient selection")
    tumour_type = st.sidebar.selectbox("Tumour type:", ["Adenocarcinoma (ADC)", "Squamous cell carcinoma (SCC)"])
    patient = st.sidebar.selectbox("Select patient:", ADC_PATIENTS if "Adenocarcinoma" in tumour_type else SCC_PATIENTS)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Segmentation layers")
    show_gt = st.sidebar.checkbox("Show manual segmentation (ground truth)", True)
    show_semi = st.sidebar.checkbox("Show semi-automatic segmentation", True)
    show_auto = st.sidebar.checkbox("Show automatic segmentation (U-Net)", True)

    st.write(f"Selected case: **{patient}** – {tumour_type}")

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

        st.markdown("### Original MRI")
        base_img = load_image_safe(orig_path)
        base_img_gray = base_img.convert("L")
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
        st.subheader("GT vs U-Net differences (FP / FN / TP)")

        if (mask_gt_img is None) or (mask_auto_img is None) or (mask_semi_img is None):
            st.info("GT + masks are required to compute the difference maps.")
        else:
            col_auto, col_semi = st.columns(2)
            with col_auto:
                st.markdown("**GT vs U-Net**")
                st.image(build_difference_map(mask_gt_img, mask_auto_img), use_container_width=True)
            with col_semi:
                st.markdown("**GT vs Semi-automatic**")
                st.image(build_difference_map(mask_gt_img, mask_semi_img), use_container_width=True)

# =========================================
# SECTION: MODEL COMPARISON
# =========================================
elif section == "Model comparison":

    st.title("Model comparison")

    if not METRICS_NPY_DIR or not MODELS_7_9_10_13:
        st.warning(
            "Model comparison is disabled because the metrics folder was not found in the repo.\n\n"
            "Expected a folder named `Metrics/` (or `metrics/`), or alternatively `CODE/`, in the repository root.\n\n"
            f"Current resolved path: `{METRICS_NPY_DIR}`"
        )
        st.stop()

    # Validate that at least one expected file exists
    any_exists = False
    for m, cfg in MODELS_7_9_10_13.items():
        if os.path.isfile(cfg["y_true"]) and os.path.isfile(cfg["y_pred"]):
            any_exists = True
            break
    if not any_exists:
        st.error(
            "Metrics folder was found, but the expected .npy files were not found.\n\n"
            "Check filenames in `Metrics/` and ensure they match exactly (case-sensitive on Streamlit Cloud)."
        )
        st.write("Metrics folder:", METRICS_NPY_DIR)
        st.write("Sample files:", sorted(os.listdir(METRICS_NPY_DIR))[:50])
        st.stop()

    st.markdown(
        """
    This section compares the final candidate models selected for the demo.
    It reports global performance, metric distributions and a ranking table
    to support the selection of the most reliable segmentation configuration.
    """
    )

    model_labels = list(MODELS_7_9_10_13.keys())
    selected_model = st.selectbox("Select model:", model_labels)

    tab1, tab2, tab3 = st.tabs(["Summary metrics", "Metric distributions", "Ranking table (7 vs 9 vs 10 vs 13)"])
    cfg = MODELS_7_9_10_13[selected_model]

    Y_val = safe_load_npy(cfg["y_true"])
    preds_val = safe_load_npy(cfg["y_pred"])

    dice_arr, iou_arr = compute_metrics_from_arrays(Y_val, preds_val)

    with tab1:
        st.subheader("Global performance summary")
        if dice_arr is None or iou_arr is None:
            st.error("Metrics could not be loaded/computed. Verify the .npy files in Metrics/.")
            st.write("Paths checked:", cfg)
        else:
            c1, c2, c3 = st.columns([1, 1, 1.2])
            c1.metric("Dice (mean ± std)", f"{float(np.mean(dice_arr)):.3f} ± {float(np.std(dice_arr)):.3f}")
            c2.metric("IoU  (mean ± std)", f"{float(np.mean(iou_arr)):.3f} ± {float(np.std(iou_arr)):.3f}")
            c3.metric("Evaluated slices", f"{int(len(dice_arr))}")

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
            d_rank, i_rank = compute_metrics_from_arrays(Y_rank, P_rank)

            if d_rank is None or i_rank is None:
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

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 13:00:42 2025

@author: Usuario
"""

from __future__ import annotations
from pathlib import Path
import os
import zipfile

import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import plotly.express as px

# =============================================================================
# HUGGING FACE DATASET
# =============================================================================
HF_REPO = "AinareDiez/AinareDiezInternship"
HF_REPO_TYPE = "dataset"
LOCAL_DATA_DIR = Path("data")

# =============================================================================
# AUTO-DOWNLOAD + UNZIP
# =============================================================================
def _extract_zip_to_data(zip_path: str) -> None:
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(LOCAL_DATA_DIR)

def ensure_folder_from_zip(zip_name_in_repo: str, expected_dir: Path, required: bool = True) -> None:
    """
    zip_name_in_repo: ruta dentro del dataset HF, ej 'Photoweb/ORIGINAL.zip'
    expected_dir: carpeta que debe existir tras extraer, ej Path('data/ORIGINAL') o Path('data/DEMO/ORIGINAL')
    """
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Si ya existe lo esperado, no hacemos nada
    if expected_dir.exists():
        return

    # Descarga (al cache de HF) + extrae en ./data
    try:
        zip_path = hf_hub_download(
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE,
            filename=zip_name_in_repo,
        )
    except Exception:
        if required:
            raise
        return

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(LOCAL_DATA_DIR)

def ensure_demo_data() -> None:
    ensure_folder_from_zip("Photoweb/ORIGINAL.zip", LOCAL_DATA_DIR / "ORIGINAL", required=True)
    ensure_folder_from_zip("Photoweb/GT_RAS_PNG_RECORTE.zip", LOCAL_DATA_DIR / "GT_RAS_PNG_RECORTE", required=True)
    ensure_folder_from_zip("Photoweb/MEJOR_SEMIAUTOMATICO.zip", LOCAL_DATA_DIR / "MEJOR_SEMIAUTOMATICO", required=True)
    ensure_folder_from_zip("Photoweb/PRUEBAAUTO - ROIM.zip", LOCAL_DATA_DIR / "PRUEBAAUTO - ROIM", required=True)

    # opcionales
    ensure_folder_from_zip("Photoweb/PRUEBAAUTO_PROBS.zip", LOCAL_DATA_DIR / "PRUEBAAUTO_PROBS", required=False)
    ensure_folder_from_zip("Photoweb/FIGS_ERRORMAPS.zip", LOCAL_DATA_DIR / "FIGS_ERRORMAPS", required=False)
    ensure_folder_from_zip("Photoweb/assets.zip", LOCAL_DATA_DIR / "assets", required=False)

try:
    ensure_demo_data()
except Exception as e:
    st.error(f"Error loading demo data: {type(e).__name__}: {e}")
    st.stop()

from pathlib import Path
import streamlit as st

LOCAL_DATA_DIR = Path("data")

# Debug (temporal): qué hay en /data
st.write("Contenido /data:", sorted([p.name for p in LOCAL_DATA_DIR.iterdir()]) if LOCAL_DATA_DIR.exists() else "NO EXISTE /data")

# Autodetección: busca carpeta ORIGINAL dentro de /data
original_dirs = list(LOCAL_DATA_DIR.rglob("ORIGINAL"))
st.write("ORIGINAL encontrados:", [str(p) for p in original_dirs[:10]])

if not original_dirs:
    st.error("No se encontró ninguna carpeta 'ORIGINAL' tras extraer. Revisa el ZIP / la extracción.")
    st.stop()

BASE_DIR = original_dirs[0].parent  # <-- clave
st.write("BASE_DIR detectado:", str(BASE_DIR))





# =============================================================================
# PATHS 
# =============================================================================


BASE_ORIGINAL = BASE_DIR / "ORIGINAL"
BASE_MANUAL   = BASE_DIR / "GT_RAS_PNG_RECORTE"
BASE_SEMI     = BASE_DIR / "MEJOR_SEMIAUTOMATICO"
BASE_AUTO_MASKS = BASE_DIR / "PRUEBAAUTO - ROIM"
BASE_AUTO_PROBS = BASE_DIR / "PRUEBAAUTO_PROBS"
ERRORMAPS_DIR   = BASE_DIR / "FIGS_ERRORMAPS"


VALID_EXTS = (".png", ".jpg", ".jpeg")

# =============================================================================
# HELPERS
# =============================================================================

def load_image_safe(path: Path) -> Image.Image:
    """Load an image and convert 16-bit or unusual modes into something Streamlit can display."""
    img = Image.open(path)
    if img.mode == "I;16":
        img = img.point(lambda i: i * (1 / 256)).convert("L")  # 16-bit -> 8-bit
    elif img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img


@st.cache_data
def list_slices_for_patient(patient: str) -> list[str]:
    """List available filenames for a given patient, based on ORIGINAL."""
    if not BASE_ORIGINAL.is_dir():
        return []
    slice_files: list[str] = []
    for p in BASE_ORIGINAL.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.stem.startswith(patient):
            slice_files.append(p.name)
    return sorted(slice_files)


def build_original_path(slice_name: str) -> Path:
    return BASE_ORIGINAL / slice_name


def build_manual_path(slice_name: str) -> Path:
    return BASE_MANUAL / slice_name


def build_semi_path(slice_name: str) -> Path:
    return BASE_SEMI / slice_name


def build_auto_mask_path(slice_name: str) -> Path | None:
    base_name = Path(slice_name).stem
    for ext in [".png", ".jpg", ".jpeg"]:
        candidate = BASE_AUTO_MASKS / f"{base_name}{ext}"
        if candidate.is_file():
            return candidate
    if not BASE_AUTO_MASKS.is_dir():
        return None
    for p in BASE_AUTO_MASKS.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and p.stem.startswith(base_name):
            return p
    return None

def build_auto_prob_path(slice_name: str) -> Path:
    return BASE_AUTO_PROBS / slice_name


def load_metrics_array(model_label: str, metric: str) -> np.ndarray | None:
    """Load a 1D numpy array of a given metric ("dice" or "iou") for the selected model."""
    path = MODEL_METRICS_FILES.get(model_label, {}).get(metric)
    if path is None or not Path(path).is_file():
        return None
    return np.load(path)


def list_error_maps_for_patient(patient: str) -> list[Path]:
    """List error-map figures associated with a patient id (e.g. 'ADC_P1')."""
    if not ERRORMAPS_DIR.is_dir():
        return []
    patient_lower = patient.lower()
    out: list[Path] = []
    for p in ERRORMAPS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in VALID_EXTS and patient_lower in p.name.lower():
            out.append(p)
    return sorted(out)


def missing_paths_warning() -> None:
    """Show a friendly warning if the expected folders are not present."""
    missing = []
    for p in [BASE_DIR, BASE_ORIGINAL, BASE_MANUAL, BASE_SEMI, BASE_AUTO_MASKS]:
        if not p.exists():
            missing.append(p)
    if missing:
        st.warning(
            "Some expected folders/files are missing.\n\n"
            "Make sure your repository contains the `DEMO/` folder (with the subfolders described in README), "
            "or set the environment variable `DEMO_DIR`.\n\n"
            + "\n".join([f"- Missing: `{m}`" for m in missing])
        )

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

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

[data-testid="stSidebar"] { background-color: var(--bg-light); border-right: 1px solid #d0d4e4; }
[data-testid="stSidebar"] * { font-size: 17px !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 20px !important; font-weight: 700 !important; color: var(--primary-color);
}
[data-testid="stSidebar"] label { font-size: 17px !important; font-weight: 600 !important; }

h1 { font-size: 32px !important; font-weight: 700 !important; color: var(--primary-color); }
h2, h3 { font-weight: 600 !important; color: var(--secondary-color); }

[data-testid="stMetricValue"] { font-size: 22px !important; }

.main .block-container { max-width: 1300px; padding-top: 1.5rem; }

button[data-baseweb="tab"] { font-weight: 500; color: var(--primary-color); }
button[data-baseweb="tab"][aria-selected="true"] { border-bottom: 2px solid var(--accent-color); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("Lung Cancer Segmentation Viewer")
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

section = st.sidebar.radio("Go to section:", ["Project overview", "Patient exploration", "Model comparison"])

st.sidebar.markdown("---")

# =============================================================================
# SECTION: PROJECT OVERVIEW
# =============================================================================

if section == "Project overview":
    st.markdown(
        """
        <style>
        .hero-box { background-color: #e9f2ff; padding: 2.2rem 2.4rem; border-radius: 16px; margin-bottom: 2rem; }
        .hero-title { font-size: 54px !important; font-weight: 800 !important; color: #003865;
                      line-height: 1.1; margin-bottom: 2.8rem !important; }
        .hero-subtext { font-size: 20px !important; color: #3a4a58; margin-top: 0.5rem; margin-bottom: 1.5rem;
                        line-height: 1.6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="hero-box">', unsafe_allow_html=True)

    col_hero_text, col_hero_images = st.columns([2, 1.3])

    with col_hero_text:
        st.markdown(
            '<div class="hero-title">Multimodal Imaging and Computational Analysis for NSCLC Subtype Differentiation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <p class="hero-subtext">
        This project integrates MRI-based tumour segmentation, multimodal image processing,
        radiomic extraction and deep learning pipelines to support the differentiation of
        non-small cell lung cancer (NSCLC) subtypes.
        <br><br>
        The framework combines manual, semi-automatic and automatic segmentation methods
        to enhance reproducibility, tumour characterisation and clinical interpretability.
        </p>
        """,
            unsafe_allow_html=True,
        )

    with col_hero_images:
        st.markdown(" ")

        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        with c1:
            st.caption("Segmentation (RGB)")
            if HERO_SEG_RGB_IMAGE.is_file():
                st.image(load_image_safe(HERO_SEG_RGB_IMAGE), use_container_width=True)
            else:
                st.info("Add `assets/segmentation_rgb.png` (optional)")

        with c2:
            st.caption("MRI + mask")
            if HERO_MRI_MASK_IMAGE.is_file():
                st.image(load_image_safe(HERO_MRI_MASK_IMAGE), use_container_width=True)
            else:
                st.info("Add `assets/mri_mask.png` (optional)")

        with c3:
            st.caption("Example Case 1")
            if OVERVIEW_ADC_IMAGE.is_file():
                st.image(load_image_safe(OVERVIEW_ADC_IMAGE), use_container_width=True)
            else:
                st.info("Add `assets/ADC_P5.png` (optional)")

        with c4:
            st.caption("Example Case 2")
            if OVERVIEW_SCC_IMAGE.is_file():
                st.image(load_image_safe(OVERVIEW_SCC_IMAGE), use_container_width=True)
            else:
                st.info("Add `assets/SCC_P5.png` (optional)")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")
    missing_paths_warning()

# =============================================================================
# SECTION: PATIENT EXPLORATION
# =============================================================================

elif section == "Patient exploration":
    st.markdown(
        """
        <h2 style="margin-bottom:0.2rem;">Patient exploration</h2>
        <p style="color:#4a4a4a;margin-top:0.2rem;margin-bottom:1.2rem;"></p>
        """,
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Patient selection")

    tumour_type = st.sidebar.selectbox("Tumour type:", ["Adenocarcinoma (ADC)", "Squamous cell carcinoma (SCC)"])

    if "Adenocarcinoma" in tumour_type:
        patient = st.sidebar.selectbox("Select patient:", ADC_PATIENTS)
    else:
        patient = st.sidebar.selectbox("Select patient:", SCC_PATIENTS)

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
        missing_paths_warning()
    else:
        if len(slice_files) == 1:
            slice_name = slice_files[0]
            st.markdown(f"*Only one image found for this patient:* `{slice_name}`")
        else:
            idx = st.slider("Slice index", 0, len(slice_files) - 1, 0)
            slice_name = slice_files[idx]
            st.markdown(f"Current file: `{slice_name}`")

        orig_path = build_original_path(slice_name)
        manual_path = build_manual_path(slice_name)
        semi_path = build_semi_path(slice_name)
        auto_mask_path = build_auto_mask_path(slice_name)
        auto_prob_path = build_auto_prob_path(slice_name)  # optional

        st.markdown("### Original MRI")
        base_img = load_image_safe(orig_path)
        base_img_rgb = base_img.convert("RGB")
        base_img_gray = base_img.convert("L")

        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            st.image(base_img, use_container_width=True)

        st.markdown("### Segmentation masks")

        mask_gt_img = load_image_safe(manual_path) if manual_path.is_file() else None
        mask_semi_img = load_image_safe(semi_path) if semi_path.is_file() else None
        mask_auto_img = load_image_safe(auto_mask_path) if (auto_mask_path and auto_mask_path.is_file()) else None

        col_gt, col_semi, col_auto = st.columns(3)

        with col_gt:
            st.markdown(
                "<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Manual (ground truth)</div>",
                unsafe_allow_html=True,
            )
            if show_gt and mask_gt_img is not None:
                st.image(mask_gt_img, use_container_width=True, clamp=True)
            elif show_gt:
                st.info("No manual mask available for this slice.")
            else:
                st.info("Layer disabled.")

        with col_semi:
            st.markdown(
                "<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Semi-automatic</div>",
                unsafe_allow_html=True,
            )
            if show_semi and mask_semi_img is not None:
                st.image(mask_semi_img, use_container_width=True, clamp=True)
            elif show_semi:
                st.info("No semi-automatic mask available for this slice.")
            else:
                st.info("Layer disabled.")

        with col_auto:
            st.markdown(
                "<div style='text-align:center; font-weight:600; margin-bottom:0.5rem;'>Automatic (U-Net)</div>",
                unsafe_allow_html=True,
            )
            if show_auto and mask_auto_img is not None:
                st.image(mask_auto_img, use_container_width=True, clamp=True)
            elif show_auto:
                st.info("No automatic mask available for this slice.")
            else:
                st.info("Layer disabled.")

        st.markdown("---")
        st.subheader("Radiomics preview")

        if mask_gt_img is None:
            st.info("Radiomic descriptors are based on the manual mask. No GT mask available for this slice.")
        else:
            gt_arr = np.array(mask_gt_img.convert("L"))
            base_arr = np.array(base_img_gray)

            gt_bin = gt_arr > 0
            if gt_bin.sum() == 0:
                st.info("The ground-truth mask is empty. No radiomic descriptors can be computed.")
            else:
                area = int(gt_bin.sum())
                coords = np.column_stack(np.nonzero(gt_bin))
                centroid_row = float(coords[:, 0].mean())
                centroid_col = float(coords[:, 1].mean())

                up = np.roll(gt_bin, -1, axis=0); up[-1, :] = False
                down = np.roll(gt_bin, 1, axis=0); down[0, :] = False
                left = np.roll(gt_bin, -1, axis=1); left[:, -1] = False
                right = np.roll(gt_bin, 1, axis=1); right[:, 0] = False

                neighbors_bg = (~up) | (~down) | (~left) | (~right)
                boundary = gt_bin & neighbors_bg
                perimeter = int(boundary.sum())

                compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else np.nan
                mean_intensity = float(base_arr[gt_bin].mean())

                c1, c2, c3 = st.columns(3)
                c1.metric("Mean intensity (a.u.)", f"{mean_intensity:.1f}")
                c2.metric("Tumour area (pixels)", f"{area}")
                c3.metric("Perimeter (pixels)", f"{perimeter}")

                c4, c5 = st.columns(2)
                with c4:
                    st.metric("Compactness (4πA/P²)", f"{compactness:.3f}" if not np.isnan(compactness) else "N/A")
                with c5:
                    st.metric("Centroid (row, col)", f"({centroid_row:.1f}, {centroid_col:.1f})")

        st.markdown("---")
        st.subheader("GT vs U-Net differences (FP / FN / TP)")

        if (mask_gt_img is None) or (mask_auto_img is None):
            st.info("Both GT and automatic masks are required to compute the difference map.")
        else:
            gt_arr = np.array(mask_gt_img.convert("L"))
            auto_arr = np.array(mask_auto_img.convert("L"))

            if gt_arr.shape != auto_arr.shape:
                auto_arr = np.array(
                    mask_auto_img.convert("L").resize((gt_arr.shape[1], gt_arr.shape[0]), Image.NEAREST)
                )

            gt_bin = gt_arr > 0
            auto_bin = auto_arr > 0

            tp = gt_bin & auto_bin
            fp = (~gt_bin) & auto_bin
            fn = gt_bin & (~auto_bin)

            h, w = gt_bin.shape
            diff_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            diff_rgb[fp] = [255, 0, 0]
            diff_rgb[fn] = [0, 0, 255]
            diff_rgb[tp] = [0, 255, 0]

            diff_img = Image.fromarray(diff_rgb)

            col_diff, col_legend = st.columns([2, 1])
            with col_diff:
                st.image(diff_img, use_container_width=True)
            with col_legend:
                st.markdown(
                    """
                **Colour legend:**
                - <span style="color:#ff0000; font-weight:600;">Red</span>: False positives (U-Net oversegmentation)  
                - <span style="color:#0000ff; font-weight:600;">Blue</span>: False negatives (U-Net undersegmentation)  
                - <span style="color:#00aa00; font-weight:600;">Green</span>: True positives (correctly segmented tumour region)
                """,
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.subheader("Confidence map (U-Net prediction probabilities)")

        if not auto_prob_path.is_file():
            st.info(
                "No probability map was found for this slice. "
                "If you generate them, save them in "
                f"`{BASE_AUTO_PROBS}` with the same filename as the original image."
            )
        else:
            prob_img = load_image_safe(auto_prob_path).convert("L")
            prob_arr = np.array(prob_img).astype(np.float32)

            p_min, p_max = prob_arr.min(), prob_arr.max()
            prob_norm = (prob_arr - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prob_arr)

            if prob_norm.shape != np.array(base_img_rgb).shape[:2]:
                prob_resized = prob_img.resize(base_img_rgb.size, Image.BILINEAR)
                prob_norm = np.array(prob_resized).astype(np.float32)
                p_min, p_max = prob_norm.min(), prob_norm.max()
                prob_norm = (prob_norm - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(prob_norm)

            base_arr_rgb = np.array(base_img_rgb).astype(np.float32) / 255.0

            heat_r = 1.0
            heat_g = 0.8 - 0.3 * prob_norm
            heat_b = 0.9 - 0.5 * prob_norm
            heat_rgb = np.stack([heat_r, heat_g, heat_b], axis=-1)

            alpha = 0.6 * prob_norm[..., None]
            overlay = (1 - alpha) * base_arr_rgb + alpha * heat_rgb
            overlay_img = Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))

            col_conf1, col_conf2 = st.columns(2)
            with col_conf1:
                st.image(prob_img, caption="Raw probability map (grayscale)", use_container_width=True)
            with col_conf2:
                st.image(overlay_img, caption="Confidence overlay (pastel heatmap on MRI)", use_container_width=True)

        st.caption(
            "All radiomic descriptors shown here are simple illustrative measurements to support qualitative "
            "interpretation of tumour morphology and model behaviour. They are not intended as a full radiomics pipeline."
        )

# =============================================================================
# SECTION: MODEL COMPARISON
# =============================================================================

elif section == "Model comparison":
    st.title("Model Comparison")

    model_name = st.selectbox("Select model:", list(MODEL_METRICS_FILES.keys()))
    st.write(f"You selected: **{model_name}**")

    mc_tab1, mc_tab2, mc_tab3 = st.tabs(["Summary metrics", "Plots (Dice / IoU)", "ADC vs SCC"])

    with mc_tab1:
        st.subheader("Global performance summary")

        dice_array = load_metrics_array(model_name, "dice")
        iou_array = load_metrics_array(model_name, "iou")

        if dice_array is None or iou_array is None:
            st.warning(
                "Metric files not found for this model.\n\n"
                f"Expected under: `{METRICS_DIR}`\n\n"
                "Tip: add `lung_app_metrics/` to the repo or set env var `METRICS_DIR` in Streamlit Cloud."
            )
        else:
            c1, c2 = st.columns(2)
            c1.metric("Dice (mean ± std)", f"{float(np.mean(dice_array)):.3f} ± {float(np.std(dice_array)):.3f}")
            c2.metric("IoU  (mean ± std)", f"{float(np.mean(iou_array)):.3f} ± {float(np.std(iou_array)):.3f}")
            st.markdown(f"Number of evaluated slices: **{len(dice_array)}**")

    with mc_tab2:
        st.subheader("Metric distributions (Plotly)")

        dice_array = load_metrics_array(model_name, "dice")
        iou_array = load_metrics_array(model_name, "iou")

        if dice_array is None or iou_array is None:
            st.warning("Metric files not found for this model. Please check `lung_app_metrics/`.")
        else:
            fig_dice = px.histogram(x=dice_array, nbins=20, title="Dice coefficient distribution")
            fig_dice.update_layout(xaxis_title="Dice", yaxis_title="Count")
            st.plotly_chart(fig_dice, use_container_width=True)

            fig_iou = px.histogram(x=iou_array, nbins=20, title="IoU distribution")
            fig_iou.update_layout(xaxis_title="IoU", yaxis_title="Count")
            st.plotly_chart(fig_iou, use_container_width=True)

            indices = np.arange(len(dice_array))
            fig_line = px.line(x=indices, y=dice_array, title="Dice per slice")
            fig_line.update_layout(xaxis_title="Slice index", yaxis_title="Dice")
            st.plotly_chart(fig_line, use_container_width=True)

    with mc_tab3:
        st.subheader("ADC vs SCC comparison")
        st.markdown(
            """
Here you can present a comparative analysis between **adenocarcinoma (ADC)** and **squamous cell carcinoma (SCC)**, for example:

- Mean Dice and IoU for ADC vs SCC  
- Boxplots of Dice by subtype  
- Differences in over-/under-segmentation patterns  

To implement it, you can:
- Compute separate metric arrays for ADC and SCC cases  
- Save them as `dice_model1_adc.npy`, `dice_model1_scc.npy`, etc.  
- Load and visualise them in this panel
        """
        )
        st.info("ADC vs SCC detailed plots will appear here once you provide separate metric files for each subtype.")
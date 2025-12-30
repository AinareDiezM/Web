# Lung Cancer Segmentation Viewer

This repository contains an interactive **Streamlit web application** for the visualisation and quantitative assessment of lung tumour segmentation results based on multimodal medical imaging.

The project focuses on **non-small cell lung cancer (NSCLC)** and enables comparison between **manual (ground truth)**, **semi-automatic**, and **fully automatic (deep learning)** segmentation approaches.

## Author
**Ainare Díez Madariaga**  
Biomedical Engineering – University of Deusto  
International internship at Bialystok University of Technology

## Repository Structure

```text
.
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
│
├── Photoweb/              # Image data used by the app
│   ├── ORIGINAL/           # Original MRI slices
│   ├── GT_RAS_PNG_RECORTE/ # Manual ground truth masks
│   ├── MEJOR SEMIAUTOMATICO/ # Semi-automatic masks
│   ├── PRUEBAAUTO - ROIM/  # Automatic segmentation masks
│   └── FIGS_ERRORMAPS/     # Error map visualisations
│
└── Metrics/               # Model evaluation data (.npy)
    ├── Y_val_modelo*.npy
    └── preds_val_modelo*.npy

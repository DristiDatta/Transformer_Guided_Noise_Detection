# Transformer-Guided Noise Detection and Correction in Remote Sensing Data for Enhanced Soil Organic Carbon Estimation

This repository contains the implementation of a Transformer-based noise detection and correction framework for improving Soil Organic Carbon (SOC) estimation using satellite reflectance data. The proposed method combines deep learning and machine learning techniques to detect and reconstruct noisy soil samples from both Landsat 8 (L8) and Sentinel-2 (S2) datasets.

📄 **Preprint Available**: [SSRN - Link to Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5205193)  
📚 **Under Review**: *Precision Agriculture*, Springer

---

## 📁 Repository Contents

- `data_L8.xlsx` – Landsat 8 reflectance data with SOC values
- `data_S2.xlsx` – Sentinel-2 reflectance data with SOC values
- `code_L8.py` – Full pipeline for detecting and correcting noise in L8 data
- `code_S2.py` – Full pipeline for detecting and correcting noise in S2 data

---## ▶️ How to Run

To process and correct noisy samples, run:


python code_L8.py   # Run pipeline on Landsat 8 data
python code_S2.py   # Run pipeline on Sentinel-2 data


## 🧠 Method Overview

The proposed framework consists of the following components:

1. **Transformer Network**: Extracts global spectral features across bands.
2. **Principal Component Analysis (PCA)**: Reduces dimensionality while retaining essential information.
3. **Isolation Forest**: Identifies noisy samples in the reflectance data.
4. **Conditional GAN (cGAN)**: Reconstructs the reflectance of noisy samples to align with clean data.
5. **Exported Outputs**: Clean + corrected dataset for downstream SOC estimation.

---

## ⚙️ Dependencies

- Python 3.8+
- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `openpyxl`
- `pykrige`
- `matplotlib`

Each script performs the following steps:

- Noise detection using Isolation Forest  
- Noise correction using Kriging (for baseline) or cGAN (for proposed)  
- Output saving of corrected datasets for SOC modeling

---

## 📬 Contact

**Dristi Datta**  
PhD Candidate, Charles Sturt University  
Email: ddatta@csu.edu.au
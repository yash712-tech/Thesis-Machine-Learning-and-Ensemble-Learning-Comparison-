# Flood Detection Dataset

This repository contains NumPy (`.npy`) files for a flood detection task, including essential feature layers (**X**) and corresponding flood maps (**Y**). The data is derived from satellite imagery and is intended for use in training and evaluating machine learning models focused on flood classification or segmentation.

## Files and Structure

- **X_best3.npy**  
  - **Shape**: Typically `(n, 3, H, W)` .
  - **Contents**:  
    1. **Coh VV** – Coherence in VV polarization  
    2. **VVPost** – Post-flood VV band  
    3. **VVPre** – Pre-flood VV band  
  - These three layers represent the “best” feature combination for flood detection, identified through model experiments.

- **Y_*.npy** (e.g., `Y_data.npy`, Mask derived using Sentinel-2)  
  - **Shape**: Usually `(1, H, W)` for segmentation tasks.  
  - **Contents**:  
    - Binary maps indicating flood-affected areas (1) vs. non-flooded areas (0).  


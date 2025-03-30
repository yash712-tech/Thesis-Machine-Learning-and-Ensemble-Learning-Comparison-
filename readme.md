# Flood Detection Dataset

This repository contains NumPy (`.npy`) files for a flood detection task, including essential feature layers (**X**) and corresponding flood maps (**Y**). The data is derived from satellite imagery and is intended for use in training and evaluating machine learning models focused on flood classification or segmentation.

## Files and Structure

- **X_best3.npy**  
  - **Shape**: Typically `(n, 3, H, W)` or `(n, H, W, 3)` (depending on your model requirements).
  - **Contents**:  
    1. **Coh VV** – Coherence in VV polarization  
    2. **VVPost** – Post-flood VV band  
    3. **VVPre** – Pre-flood VV band  
  - These three layers represent the “best” feature combination for flood detection, identified through model experiments.

- **Y_*.npy** (e.g., `Y_data.npy`, adjust name to match your actual file)  
  - **Shape**: Usually `(n, H, W)` for segmentation tasks or `(n,)` for classification.  
  - **Contents**:  
    - Binary or multi-class label maps indicating flood-affected areas (1) vs. non-flooded areas (0).  

> **Tip**: Confirm exact file names and shapes in your local environment—this is just an example.

## Usage Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

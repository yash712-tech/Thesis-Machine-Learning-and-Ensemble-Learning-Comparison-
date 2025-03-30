Dataset Overview
This repository provides NumPy (.npy) files containing feature layers (X variables) and corresponding classification maps (Y variables) for a flood detection project. Each file stores a specific slice of the data, ready to be loaded directly in Python for model training, validation, or inference.

File Descriptions
X_best3.npy

Dimensions: The file typically has the shape (n, 3, H, W) or (n, H, W, 3) depending on how you stacked the features.

Contents:

Coh_VV – Coherence in VV polarization

VVPost – Post-flood VV

VVPre – Pre-flood VV

These three layers represent the “best” performing feature set for the flood detection task.

Y_* (e.g., Y_data.npy)

Dimensions: Usually (n, H, W) for segmentation labels, or (n,) for classification, depending on your task.

Contents:

Binary or multi-class map indicating whether a pixel/region is flooded (1) or not flooded (0), or potentially more classes.

Note: Replace the above placeholders (e.g., Y_data.npy, (n, 3, H, W)) with the exact names and shapes you use.

Usage
Clone or download this repository to your local machine.

Load the NumPy arrays in Python:

python
Copy
Edit
import numpy as np

X_best3 = np.load("X_best3.npy")  # Shape: (n, 3, H, W)
Y_data  = np.load("Y_data.npy")   # Shape: (n, H, W)
Integrate with your pipeline:

Pass X_best3 into your model or data preprocessing pipeline.

Use Y_data for training labels, validation accuracy checks, or generating flood maps.

Data Format Details
Coordinate System: If applicable, specify how the image coordinates map to geospatial references (e.g., Sentinel-1 or Sentinel-2 lat/lon grids).

Scaling/Normalization: Mention any preprocessing steps (like normalization, standardization, or clipping of input values).

Missing Values: If certain regions are masked due to cloud cover or sensor issues, indicate how these are represented (e.g., zero or NaN).

Attribution and Licensing
If the data is based on publicly available satellite imagery or from a specific project, provide appropriate attribution and licensing details here.

Clarify usage rights and any restrictions on data distribution.

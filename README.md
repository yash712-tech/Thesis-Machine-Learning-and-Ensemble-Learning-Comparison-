# README

## Thesis Code for Model Evaluation and Comparison
This repository contains Python code for training and evaluating machine learning models for flood detection using satellite imagery. The project involves data preprocessing, model training using Random Forest, SVM, and Gradient Boosting classifiers, and performance evaluation with metrics such as accuracy, IoU, and F1-score. Additionally, ensemble learning has been applied to combine multiple ML models, and a comparative analysis of their performance has been conducted.

## Features
 
- Model inference using trained classifiers
- Performance evaluation with k-fold cross-validation
- Visualization of contingency maps
- Ensemble learning for model combination and performance comparison

## Installation

### Prerequisites

Ensure you have Python installed (version 3.7+ recommended).

### Install Dependencies
Install required packages

## Usage

### Training the Model

To train a model with your dataset, run:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)
```

```python
from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', C=1.0, random_state=42)
model_svm.fit(X_train, Y_train)
```

```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model_xgb.fit(X_train, Y_train)
```

```python
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[('rf', model_rf), ('svm', model_svm), ('xgb', model_xgb)],
    voting='hard'
)
ensemble_model.fit(X_train, Y_train)
```

### Model Inference

Run inference using a trained model:

```python
prediction = model_inference(X_test, Y_test, 'path_to_model')
```

### Visualization

To visualize contingency maps:

```python
contingency_maps(Y_true, Y_pred, orig_img_size)
```

## Dataset

Paths to datasets are defined within the code, adjust them as needed to fit your directory structure.

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

This project is for academic research and follows the author's usage guidelines.

---

# Requirements.txt

```
os
pickle
numpy
scikit-learn
matplotlib
seaborn
pandas
scipy
```
## Required Version

```
matplotlib==3.5.2
numpy==1.21.5
pandas==1.4.4
scikit_learn==1.0.2
seaborn==0.11.2
```

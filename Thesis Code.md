```python
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import cross_val_score
from scipy.ndimage import median_filter

```


```python
import os
os.getcwd()
```


```python
def calculate_metrics(y_true, y_pred):
    # Calculate Overall Accuracy (OA)
    oa = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # Calculate Producer's Accuracy (PA) and User's Accuracy (UA) for each class
    pa = cm.diagonal() / cm.sum(axis=1)
    ua = cm.diagonal() / cm.sum(axis=0)
    
    # Calculate Intersection over Union (IoU) for each class
    iou = jaccard_score(y_true, y_pred, average=None)
    
    # Calculate F1 Score for each class
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Print the metrics
    print(f"Overall Accuracy (OA): {oa:.4f}")
    for i, (p, u, iou_score, f1_score_val) in enumerate(zip(pa, ua, iou, f1)):
        print(f"Class {i}: PA = {p:.4f}, UA = {u:.4f}, IoU = {iou_score:.4f}, F1 Score = {f1_score_val:.4f}")
        
def model_inference(X, Y,model_path):
    model_path =  os.path.join(model_path,'Pointswala.sav1')
    with open(model_path, 'rb') as f:
            model_load = pickle.load(f)

    prediction = model_load.predict(X)
    calculate_metrics(Y,prediction)
    return prediction


def model_inference_filtering(X, Y,model_path, orig_img_size):
    model_path =  os.path.join(model_path,'Pointswala.sav1')
    with open(model_path, 'rb') as f:
            model_load = pickle.load(f)

    prediction = model_load.predict(X)
    
    origina = prediction.reshape(orig_img_size)

    # Apply a median filter with size 20
    filtered_image = median_filter(origina, size=20)

    # Flatten the image back to a 1D array (though for plotting this step isn't needed)
    Y_all_aqqala_filtered = filtered_image.flatten()
    
    calculate_metrics(Y,Y_all_aqqala_filtered)
    contingency_maps(Y,Y_all_aqqala_filtered, orig_img_size)
    

def model_inference_filtering_modelinram(X, Y,model_load,orig_img_size):
    prediction = model_load.predict(X)
    origina = prediction.reshape(orig_img_size)

    # Apply a median filter with size 20
    filtered_image = median_filter(origina, size=20)
    # Flatten the image back to a 1D array (though for plotting this step isn't needed)
    Y_all_aqqala_filtered = filtered_image.flatten()
    
    calculate_metrics(Y,Y_all_aqqala_filtered)
    contingency_maps(Y,Y_all_aqqala_filtered, orig_img_size)
    



def contingency_maps(y_true, y_pred, orig_img_size):
    # Reshape the flattened prediction and true labels back to their original image dimensions
    original_image = y_true.reshape(orig_img_size)
    prediction_image = y_pred.reshape(orig_img_size)

    # Initialize an output image with dimensions (height, width, 3) for RGB
    output_image = np.zeros((orig_img_size[0], orig_img_size[1], 3))

    # Assign colors based on the classification
    # Convert lists to numpy arrays for element-wise division
    # Define colors using RGB values normalized to [0, 1]
    cornflower_blue = np.array([100, 149, 237]) / 255.0  # Cornflower blue for TP
    red = np.array([255, 0, 0]) / 255.0                 # Red for FP
    white = np.array([255, 255, 255]) / 255.0           # White for TN
    black = np.array([0, 0, 0]) / 255.0                 # Black for FN

    # Assign colors based on the classification
    output_image[(original_image == 1) & (prediction_image == 1)] = cornflower_blue
    output_image[(original_image == 0) & (prediction_image == 0)] = white
    output_image[(original_image == 0) & (prediction_image == 1)] = red
    output_image[(original_image == 1) & (prediction_image == 0)] = black

    # Plotting
    plt.imshow(output_image)
    plt.title('Contingency Map')
    plt.axis('off')
    plt.show()


def K_fold_scores(X_sis, Y_sis, model_path,num_k):
    # Get indices where Y is 1 and 0
    indices_y1 = np.where(Y_sis == 1)[0]
    indices_y0 = np.where(Y_sis == 0)[0]

    # Sample 1000 indices where Y is 1
    sampled_indices_y1 = np.random.choice(indices_y1, num_k, replace=False)

    # Sample 500 indices where Y is 0
    sampled_indices_y0 = np.random.choice(indices_y0, num_k, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_indices_y1, sampled_indices_y0])

    # Create sampled X and Y
    X_sampled = X_sis[sampled_indices]
    Y_sampled = Y_sis[sampled_indices]

    # Perform k-fold cross-validation
    model_path2 =  os.path.join(model_path,'Pointswala.sav1')
    with open(model_path2, 'rb') as f:
            model_load = pickle.load(f)
    cv_scores = cross_val_score(model_load, X_sampled, Y_sampled.reshape(-1), cv=10, scoring='accuracy')

    print(f"Accuracy scores for each fold: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean()}")
    print(f"Standard deviation of accuracy: {cv_scores.std()}")
    


def points_sampling(X_sis, Y_sis,num_k):
    # Get indices where Y is 1 and 0
    indices_y1 = np.where(Y_sis == 1)[0]
    indices_y0 = np.where(Y_sis == 0)[0]

    # Sample 1000 indices where Y is 1
    sampled_indices_y1 = np.random.choice(indices_y1, num_k, replace=False)

    # Sample 500 indices where Y is 0
    sampled_indices_y0 = np.random.choice(indices_y0, num_k, replace=False)

    # Combine sampled indices
    sampled_indices = np.concatenate([sampled_indices_y1, sampled_indices_y0])

    # Create sampled X and Y
    X_sampled = X_sis[sampled_indices]
    Y_sampled = Y_sis[sampled_indices]
    
    return [X_sampled,Y_sampled]

```


```python
# All layers
dset_all_layers_sis = '/yash/flood/Sistan_floods/RF/Coh+AMP_PrePost1June21_filtersize20/'
dset_all_layers_aqqala = '/yash/flood/Aqqala_floods/RFv2/Coh+AMP_PrePost1June13/'
dset_all_layers_pak = '/yash/flood/Pakistan_2020/RF/Coh+AMP_PrePost1June13/'




dset_best3_layers_sis = '/yash/flood/Sistan_floods/RF/CohVV+VVPostpreJune21_filtersize20/'
dset_best3_layers_aqqala = '/yash/flood/Aqqala_floods/RFv2/CohVV+VVPostpreJune13/'
dset_best3_layers_pak = '/yash/flood/Pakistan_2020/RF/CohVV+VVPostpreJune13/'
```


```python
dset_aqqala = '/yash/flood/Aqqala_floods/RFv2/'
dset_aqqala_best3 = os.path.join(dset_aqqala,'CohVV+VVPostpreJune13')
dset_aqqala_vh_prepost = os.path.join(dset_aqqala,'VH_PrePost1June21')
dset_aqqala_alleight = os.path.join(dset_aqqala,'Coh+AMP_PrePost1June13')
dset_aqqala_vv_prepost = os.path.join(dset_aqqala,'VVPost_VVPre1June13')
dset_aqqala_vv_post_vh_post = os.path.join(dset_aqqala,'VVPost_VHPost1June13')
dset_aqqala_cohv_vv_coh_vh = os.path.join(dset_aqqala,'COhVV_CohVH1June13')
dset_aqqala_vvvh_prepost = os.path.join(dset_aqqala,'VVVH_PrePost1June13')
dset_aqqala_cohvvvh_amp_vvvh_prepost = os.path.join(dset_aqqala,'Cohpost_AmpPrePost1June13')
```


```python
X_best3_aqqala = np.load('X_best3_aqqala.npy')
Y_best3_aqqala = np.load('Y_best3_aqqala.npy')
X_best3_pak = np.load('X_best3_pak.npy')
Y_best3_pak=np.load('Y_best3_pak.npy')
X_best3_sis = np.load('X_best3_sis.npy')
Y_best3_sis = np.load('Y_best3_sis.npy')
```


```python
X_all_sis = np.load('X_full_sis.npy')
Y_all_sis = np.load('Y_full_sis.npy')
X_all_aqqala = np.load('X_full_aqqala.npy')
Y_all_aqqala = np.load('Y_full_aqqala.npy')
X_all_pak = np.load('X_full_pak.npy')
Y_all_pak = np.load('Y_full_pak.npy')
```


```python
columns = ['CohVV', 'CohVH', 'CohVV_Pre', 'CohVH_Pre','VVPre', 'VVPost', 'VHPre', 'VHPost']  # Assuming there are 8 layers

# Create an index map from column names to indices
index_map = {name: idx for idx, name in enumerate(columns)}

# Define the scenarios with the correct column names
scenarios = {
    'coh_amp_prepost' : ['CohVV','CohVH','VVPre','VVPost','VHPre','VHPost'],
    'vv_vh_prepost' : ['VVPre','VVPost','VHPre','VHPost'],
    'coh_vvvh' : ['CohVV','CohVH'],
    'vv_vh_post' :['VHPost','VVPost'],
    'vv_prepost' : ['VVPre','VVPost'],
    'cohprepost_ampprepost' : ['CohVV','CohVH','CohVV_Pre','CohVH_Pre','VVPre','VVPost','VHPre','VHPost'],
    'cohvv_vv_prepost' : ['CohVV','VVPre','VVPost'],
    'vh_prepost':['VHPre','VHPost']
   
}

# Dictionary to hold the resulting matrices for each scenario
X_scenarios_aqqala = {}
X_scenarios_sistan ={}
X_scenarios_pakistan ={}

# Extract columns based on scenarios and store in dictionary
for scenario, cols in scenarios.items():
    X_scenarios_aqqala[scenario] = X_all_aqqala[:, [index_map[col] for col in cols]]

for scenario, cols in scenarios.items():
    X_scenarios_sistan[scenario] = X_all_sis[:, [index_map[col] for col in cols]]

    
for scenario, cols in scenarios.items():
    X_scenarios_pakistan[scenario] = X_all_pak[:, [index_map[col] for col in cols]]
```


```python
print(np.shape(X_scenarios_aqqala['coh_amp_prepost']))
print(np.shape(X_scenarios_sistan['coh_amp_prepost']))
print(np.shape(X_scenarios_pakistan['coh_amp_prepost']))
```


```python
# Assuming 'flattened_image' is your input flattened array.
# Replace the random data with your actual flattened image data.

# Reshape the flattened image back to its original size
original_image = Y_all_aqqala.reshape((2396, 3058))

# Apply a median filter with size 20
filtered_image = median_filter(original_image, size=20)

# Flatten the image back to a 1D array (though for plotting this step isn't needed)
Y_all_aqqala_filtered = filtered_image.flatten()

# Plot the original and the filtered images
plt.figure(figsize=(12, 8))

# Plotting the original image
plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plotting the filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()

original_image_pak = Y_all_pak.reshape((931,720))

# Apply a median filter with size 20
filtered_image = median_filter(original_image_pak, size=20)

# Flatten the image back to a 1D array (though for plotting this step isn't needed)
Y_all_pak_filtered = filtered_image.flatten()

# Plot the original and the filtered images
plt.figure(figsize=(12, 8))

# Plotting the original image
plt.subplot(1, 2, 1)
plt.imshow(original_image_pak, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plotting the filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
plt.colorbar()

plt.show()



original_image_sis = Y_all_sis.reshape((3905,3335))

# Apply a median filter with size 20
filtered_image = median_filter(original_image_sis, size=20)

# Flatten the image back to a 1D array (though for plotting this step isn't needed)
Y_all_sis_filtered = filtered_image.flatten()

# Plot the original and the filtered images
plt.figure(figsize=(12, 8))

# Plotting the original image
plt.subplot(1, 2, 1)
plt.imshow(original_image_sis, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Plotting the filtered image
plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')
plt.colorbar()

plt.show()
```


```python
a1 = points_sampling(X_scenarios_aqqala['coh_amp_prepost'],Y_all_aqqala_filtered,5000)
b1 = points_sampling(X_scenarios_sistan['coh_amp_prepost'],Y_all_sis_filtered,5000)
c1 = points_sampling(X_scenarios_pakistan['coh_amp_prepost'],Y_all_pak_filtered,5000)

X_training_aqqala = []
X_training_sistan = []
X_training_pakistan = []
Y_training_aqqala = []
Y_training_sistan = []
Y_training_pakistan = []


X_training_aqqala = a1 [0]; Y_training_aqqala = a1[1]
X_training_sistan = b1 [0]; Y_training_sistan = b1[1]
X_training_pakistan = c1 [0]; Y_training_pakistan = c1[1]

gen_X_training = np.concatenate((X_training_aqqala, X_training_sistan, X_training_pakistan), axis=0)
gen_Y_training = np.concatenate((Y_training_aqqala, Y_training_sistan, Y_training_pakistan), axis=0)


# Check the result
print(f"Shape of combined array: {gen_Y_training.shape}")  # Should be (3000, 6)
```


```python
### Training using 2 and testing on the third 

# gen_X_training = np.concatenate((X_training_aqqala, X_training_sistan), axis=0)
# gen_Y_training = np.concatenate((Y_training_aqqala, Y_training_sistan), axis=0)

# gen_X_training = np.concatenate((X_training_aqqala, X_training_pakistan), axis=0)
# gen_Y_training = np.concatenate((Y_training_aqqala, Y_training_pakistan), axis=0)

gen_X_training = np.concatenate((X_training_pakistan, X_training_sistan), axis=0)
gen_Y_training = np.concatenate((Y_training_pakistan, Y_training_sistan), axis=0)
```


```python
#Training a Random Forest model from these points 

model = RandomForestClassifier(n_estimators = 100, random_state = 42,oob_score=True,n_jobs=-1)
model.fit(gen_X_training, gen_Y_training) #For sklearn no one hot encoding
```


```python

model = GradientBoostingClassifier(
    n_estimators=100, 
    random_state=42
)
model.fit(gen_X_training, gen_Y_training)
```


```python
#Running Inference on this model
orig_img_size =  (2396, 3058)

model_inference_filtering_modelinram(X_scenarios_aqqala['coh_amp_prepost'], Y_all_aqqala_filtered, model,(2396, 3058))
model_inference_filtering_modelinram(X_scenarios_sistan['coh_amp_prepost'], Y_all_sis_filtered, model,(3905,3335))
model_inference_filtering_modelinram(X_scenarios_pakistan['coh_amp_prepost'], Y_all_pak_filtered, model,(931,720))

```


```python


# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
svm_model = SVC(random_state=42)
xgb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train and evaluate each model
models = [rf_model, svm_model, xgb_model]
model_names = ['Random Forest', 'SVM', 'XGBoost']

for name, model in zip(model_names, models):
    print(f"\nResults for {name}:")
    model.fit(gen_X_training, gen_Y_training)
    
    # Your inference code for each region
    model_inference_filtering_modelinram(X_scenarios_aqqala['coh_amp_prepost'], Y_all_aqqala_filtered, model, (2396, 3058))
    model_inference_filtering_modelinram(X_scenarios_sistan['coh_amp_prepost'], Y_all_sis_filtered, model, (3905,3335))
    model_inference_filtering_modelinram(X_scenarios_pakistan['coh_amp_prepost'], Y_all_pak_filtered, model, (931,720))

```


```python

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Comparison Across Regions')

# Plot results for each model
models = [rf_model, svm_model, xgb_model]
model_names = ['Random Forest', 'SVM', 'XGBoost']

for idx, (model, name) in enumerate(zip(models, model_names)):
   results = {
       'Aqqala': model_inference_filtering_modelinram(X_scenarios_aqqala['coh_amp_prepost'], Y_all_aqqala_filtered, model, (2396, 3058)),
       'Sistan': model_inference_filtering_modelinram(X_scenarios_sistan['coh_amp_prepost'], Y_all_sis_filtered, model, (3905,3335)),
       'Pakistan': model_inference_filtering_modelinram(X_scenarios_pakistan['coh_amp_prepost'], Y_all_pak_filtered, model, (931,720))
   }
   
   axes[idx].bar(results.keys(), results.values())
   axes[idx].set_title(name)
   axes[idx].set_ylim(0, 1)
   axes[idx].set_ylabel('Score')

plt.tight_layout()
plt.show()
```

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
import os
import pickle

cwd = os.getcwd()

def extract_intensity_features(image):
    """
    Extract intensity-based features from a image.
    """
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    std_intensity = np.std(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    return [mean_intensity, median_intensity, std_intensity, min_intensity, max_intensity]

def extract_texture_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Extract texture-based features from a image using Gray-Level Co-occurrence Matrix (GLCM).
    """
    # Ensure image is 2D
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)

    texture_features = []
    for distance in distances:
        for angle in angles:
            glcm = graycomatrix(image, [distance], [angle], levels=levels, symmetric=True, normed=True)
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
                texture_features.append(graycoprops(glcm, prop)[0, 0])
    
    return texture_features


def extract_lbp_features(image, P=8, R=1, method='uniform'):
    """
    Extract Local Binary Pattern features from a image.
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    
    lbp = local_binary_pattern(image, P, R, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    return hist.tolist()


def extract_features(images):
    """
    Extract intensity-based, texture-based, and LBP-based features.
    """
    features = []
    for image in images:
        intensity_features = extract_intensity_features(image)
        texture_features = extract_texture_features(image)
        lbp_features = extract_lbp_features(image)
        combined_features = intensity_features + texture_features + lbp_features
        features.append(combined_features)
    
    return np.array(features)


def feature_selection(X, y):
    """
    Perform feature selection using XGBoost.
    """
    xgb_model = XGBClassifier(n_jobs=-1, random_state=42)
    xgb_model.fit(X, y)
    feature_importances = xgb_model.feature_importances_
    important_features = np.where(feature_importances > np.percentile(feature_importances, 75))[0]
    X_selected = X[:, important_features]
    
    return X_selected, important_features


def hyperparameter_tuning(X, y):
    """
    Perform hyperparameter tuning using GridSearchCV.
    """
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.001]
    }
    
    xgb_model = XGBClassifier(n_jobs=-1, random_state=42)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    
    return best_model

# Load your preprocessed images and labels from NumPy arrays or pickle files
X_train = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_train.npy")
y_train = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_train.npy")
X_valid = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_valid.npy")
y_valid = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_valid.npy")
X_test = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\X_test.npy")
y_test = np.load(r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification\Pre-process files\y_test.npy")

# Extract features from the training set
X_train_features = extract_features(X_train)

# Perform feature selection
X_train_selected, selected_indices = feature_selection(X_train_features, y_train)

# Perform hyperparameter tuning
best_model = hyperparameter_tuning(X_train_selected, y_train)

# Extract features from the validation and test sets using the selected features
X_valid_features = extract_features(X_valid)[:, selected_indices]
X_test_features = extract_features(X_test)[:, selected_indices]

# Evaluate the best model on the validation and test sets
valid_accuracy = best_model.score(X_valid_features, y_valid)
test_accuracy = best_model.score(X_test_features, y_test)

print(f"Validation accuracy: {valid_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print(f"Validation accuracy: {valid_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

output_file = cwd

# Save data as NumPy arrays or pickle files
try:
    np.save("X_train_features.npy", X_train_selected)
    np.save("X_valid_features.npy", X_valid_features)
    np.save("X_test_features.npy", X_test_features)
    np.save('extracted_features.npy', selected_indices)
except Exception as e:
    # If an exception occurs, print the error message
    print(f"An error occurred while saving the file: {e}")
else:
    current_dir = os.getcwd()
    print(f"Array saved successfully at: {current_dir}")

# Or save selected indices as a Python list using pickle
with open('extracted_features.pkl', 'wb') as f:
    pickle.dump(selected_indices.tolist(), f)

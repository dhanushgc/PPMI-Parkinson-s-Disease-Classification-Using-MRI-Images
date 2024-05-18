import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pydicom
from PIL import Image
from pywt import wavedec2, waverec2
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern

# Assuming the previous functions and models are already defined and loaded
# Load your model
with open('finalized_model.pkl', 'rb') as file:
    model = pickle.load(file)

import os
import cv2

def wavelet_preprocess(image, wavelet='bior3.7', levels=3):
    if image.ndim == 3 and image.shape[2] == 3:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Assume image is already grayscale
        gray = image
    
    # Apply DWT
    coeffs = wavedec2(gray, wavelet, level=levels)
    
    # Reconstruct the image from the wavelet coefficients
    reconstructed = waverec2(coeffs, wavelet)
    
    # Histogram equalization
    preprocessed = cv2.equalizeHist(np.uint8(reconstructed))
    
    return preprocessed

def resize_image(image, target_size):
    """
    Resize the input image to a fixed size by padding or cropping.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the aspect ratio
    aspect_ratio = width / height

    # Calculate the new dimensions while preserving aspect ratio
    new_width = target_width
    new_height = int(target_width / aspect_ratio)

    # Resize the image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Determine the number of channels
    if len(resized.shape) == 2:
        channels = 1
        resized = np.expand_dims(resized, axis=2)
    else:
        channels = resized.shape[2]

    # Create a new image with the target size and fill it with black
    padded = np.zeros((target_height, target_width, channels), dtype=np.uint8)

    # Calculate the starting point for cropping or padding
    start_y = (target_height - new_height) // 2
    start_x = 0

    # Crop or pad the image
    if new_height > target_height:
        # Crop the image
        padded = resized[0:target_height, :]
    else:
        # Pad the image
        padded[start_y:start_y+new_height, :] = resized

    return padded

def extract_intensity_features(image):
    """
    Extract intensity-based features from a single image.
    """
    mean_intensity = np.mean(image)
    median_intensity = np.median(image)
    std_intensity = np.std(image)
    min_intensity = np.min(image)
    max_intensity = np.max(image)
    
    return [mean_intensity, median_intensity, std_intensity, min_intensity, max_intensity]

def extract_texture_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Extract texture-based features from a single image using Gray-Level Co-occurrence Matrix (GLCM).
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
    Extract Local Binary Pattern features from a single image.
    """
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.squeeze(image, axis=2)
    
    lbp = local_binary_pattern(image, P, R, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
    
    return hist.tolist()


def dicom_to_png(input_dir):
    try:
        # Load all DICOM files that end with .dcm
        dicoms = [pydicom.dcmread(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if f.endswith('.dcm')]
        
        # Sort DICOMs by Instance Number
        dicoms.sort(key=lambda x: int(x.InstanceNumber))
        
        # Select the middle DICOM file
        if not dicoms:
            print(f"No DICOM files found in {input_dir}")
            return None
        middle_index = len(dicoms) // 2
        middle_dicom = dicoms[middle_index]

        # Convert pixel data to a NumPy array
        pixel_array = middle_dicom.pixel_array

        # Normalize and convert to uint8
        pixel_array = ((pixel_array - np.min(pixel_array)) * 255) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array = pixel_array.astype(np.uint8)

        # Convert to PIL Image and return
        image = Image.fromarray(pixel_array)
        return image
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function to process a single image
def process_and_predict(dicom_path):
    # Load image
    image = dicom_to_png(dicom_path)
    if image is None:
        return "Image not found or unable to read."
    image = np.array(image)
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image

    # Preprocess the image (as per your preprocessing function)
    preprocessed = wavelet_preprocess(image)  # Example preprocessing function
    resized = resize_image(preprocessed, (256, 256))  # Resizing to the input size used during training

    # Extract features
    intensity_features = extract_intensity_features(resized)
    texture_features = extract_texture_features(resized)
    lbp_features = extract_lbp_features(resized)
    features = np.array(intensity_features + texture_features + lbp_features).reshape(1, -1)

    # Feature selection (Apply the same selection indices obtained during training)
    # Load feature indices if saved or ensure they are available in the environment
    selected_indices = np.load('extracted_features.npy')  # Load indices if saved
    selected_features = features[:, selected_indices]

    # Predict the class
    prediction = model.predict(selected_features)
    class_name = "PD" if prediction[0] == 0 else "Control"
    
    plt.imshow(image_rgb)
    plt.title(f'Predicted Class: {class_name}')
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()
    return class_name

# Example usage update the path
image_path = r"C:\Users\dhanu\Downloads\control_test1\PPMI\3635\AX_PD_+_T2\2013-02-07_07_45_45.0\I362044"
prediction = process_and_predict(image_path)

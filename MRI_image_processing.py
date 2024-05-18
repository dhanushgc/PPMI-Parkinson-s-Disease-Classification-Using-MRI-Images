import os
import cv2
import numpy as np
from pywt import wavedec2, waverec2
from sklearn.model_selection import train_test_split
import albumentations as A
import pickle

cwd = os.getcwd()

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

def augment_images(images, labels, augmentations):
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        augmented = augmentations(image=image)['image']
        augmented_images.append(augmented)
        augmented_labels.append(label)
    
    return augmented_images, augmented_labels

def load_images(data_dir, categories):
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)
                labels.append(categories.index(category))
    
    return images, labels

def preprocess_and_save(data_dir, categories, target_size, output_file):
    images, labels = load_images(data_dir, categories)

    # Preprocess images
    preprocessed_images = []
    for image in images:
        preprocessed = wavelet_preprocess(image)
        resized = resize_image(preprocessed, target_size)
        preprocessed_images.append(resized)
        
    for i, image in enumerate(preprocessed_images):
        print(f"Preprocessed image {i+1} shape: {image.shape}")
        
    # Data augmentation
    augmentations = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    ])


    augmented_images, augmented_labels = augment_images(preprocessed_images, labels, augmentations)
    preprocessed_images.extend(augmented_images)
    labels.extend(augmented_labels)


    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Convert to NumPy arrays
    X_train = np.array(X_train)
    X_valid = np.array(X_valid)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)

    # Save data as NumPy arrays or pickle files
    np.save(f"{output_file}_X_train.npy", X_train)
    np.save(f"{output_file}_y_train.npy", y_train)
    np.save(f"{output_file}_X_valid.npy", X_valid)
    np.save(f"{output_file}_y_valid.npy", y_valid)
    np.save(f"{output_file}_X_test.npy", X_test)
    np.save(f"{output_file}_y_test.npy", y_test)

    # Alternatively, save data as pickle files
    with open(f"{output_file}_data.pkl", "wb") as f:
         pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test), f)

# Example usage
data_dir = r"C:\Users\dhanu\Desktop\Projects\PPMI PD Classification"
categories = ["PD", "Control"]
target_size = (256, 256)
output_file = cwd

preprocess_and_save(data_dir, categories, target_size, output_file)
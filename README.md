# PPMI-Parkinson-s-Disease-Classification-Using-MRI-Images
Implemented XGBoost, SVM, and Random Forest classifiers to classify PD from MRI images with 78% accuracy. Applied wavelet transformations and GLCM techniques to enhance MRI image quality and extract diagnostic features.

STEPS TO RUN THE CODE

Prerequisites: Ensure you have Python installed on your system and the required libraries listed 
below. Install these using pip command.
➢ os, pandas, pydicom, PIL, numpy, glob, skimage, sklearn, xgboost, cv2, pickle, matplotlib, 
pywt, albumentations.

Step-1: Download Data
➢ Download the PD and Control MRI images from the PPMI database. Save them in separate 
folders named PD and Control. Also, download the corresponding metadata files for each 
category.

Step-2: Convert DICOM to PNG
➢ Before running the dicom_to_png.py script, update the metadata file paths in the script to point 
to your downloaded metadata files. Then execute the script to convert the DICOM MRI images 
to PNG format and save the images in their respective folders.

Step-3: Image Processing
➢ Update the MRI_image_processing.py script with the paths to your PD and Control folders. 
After updating, run the script to apply image preprocessing techniques. Enhanced images will 
be saved as NumPy files.

Step-4: Feature Extraction
➢ Ensure the feature_extraction.py script is updated with the paths to the NumPy files generated 
in Step 3. After updating, run the script to extract features, which will be saved as new NumPy 
arrays.

Step-5: Model Training
➢ Update the model_training.py script to load the feature NumPy files from Step 4. After 
updating the paths, run the script to train the model.

Step-6: Model Testing
➢ Update the model_testing.py script with the path to your saved model and the directory 
containing unseen DICOM files. Ensure the input path is correctly set to where the unseen 
DICOM files are stored. Run the script to test the PD classification.

Make sure all scripts are correctly pointing to the necessary directories and files where your data 
and outputs are stored. Adjust the paths in the scripts as necessary to match your local environment.

import os
import pandas as pd
import pydicom
from PIL import Image
import numpy as np
from glob import glob

cwd = os.getcwd()

def convert_middle_dicom_to_image(input_dir, output_dir, image_id, group):
    try:
        # Load all DICOM files
        dicoms = [pydicom.dcmread(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if f.endswith('.dcm')]
        
        # Sort DICOMs by Instance Number
        dicoms.sort(key=lambda x: int(x.InstanceNumber))
        
        if not dicoms:
            print(f"No DICOM files found in {input_dir}")
            return
        middle_index = len(dicoms) // 2
        middle_dicom = dicoms[middle_index]

        pixel_array = middle_dicom.pixel_array

        # Normalize and convert to uint8
        pixel_array = ((pixel_array - np.min(pixel_array)) * 255) / (np.max(pixel_array) - np.min(pixel_array))
        pixel_array = pixel_array.astype(np.uint8)

        # Create group-specific output directory
        group_dir = os.path.join(output_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        # Save the image
        image = Image.fromarray(pixel_array)
        output_file = os.path.join(group_dir, f"{image_id}.png")
        image.save(output_file)
        print(f"Image saved successfully: {output_file}")
    except FileNotFoundError:
        print(f"File not found in {input_dir}, check the file path and permissions.")
    except PermissionError:
        print("Permission denied while accessing the files, check the file permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Load metadata Replace the path
metadata_path = r"C:\Users\dhanu\Downloads\Control_sgittal_mri_image_5_06_2024.csv"  
metadata = pd.read_csv(metadata_path)

# Define output directory
output_dir = cwd

# Process each entry in the metadata
for index, row in metadata.iterrows():
    description = row['Description'].replace(' ', '_')
    # Define the base path up to the description replace th path 
    base_path = f"C:/Users/dhanu/Downloads/Control_sgittal_mri_image/PPMI/{row['Subject']}/{description}/"

    # find the first subdirectory that matches any pattern, assumed to be the next directory
    sub_dirs = glob(os.path.join(base_path, '*/'))

    if not sub_dirs:
        print(f"No subdirectories found for {base_path}")
        continue

    #find the specific Image Data ID folder
    input_dir = os.path.join(sub_dirs[0], str(row['Image Data ID']))

    convert_middle_dicom_to_image(input_dir, output_dir, str(row['Image Data ID']), row['Group'])
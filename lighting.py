import os
import cv2
import numpy as np

# Define the target brightness level
TARGET_BRIGHTNESS = 128
TARGET_CONTRAST_STD = 64

def calculate_brightness(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the average brightness
    brightness = np.mean(gray_image)
    return brightness

def calculate_contrast(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the standard deviation of the pixel values
    contrast = np.std(gray_image)
    return contrast

def adjust_brightness_and_contrast(image, target_brightness=TARGET_BRIGHTNESS, target_contrast_std=TARGET_CONTRAST_STD):
    # Calculate current brightness and contrast
    current_brightness = calculate_brightness(image)
    current_contrast = calculate_contrast(image)
    
    # Adjust brightness
    brightness_ratio = target_brightness / current_brightness
    bright_adjusted_image = cv2.convertScaleAbs(image, alpha=brightness_ratio, beta=0)
    
    # Adjust contrast
    mean = np.mean(bright_adjusted_image)
    current_std = np.std(bright_adjusted_image)
    contrast_ratio = target_contrast_std / current_std
    contrast_adjusted_image = cv2.convertScaleAbs(bright_adjusted_image, alpha=contrast_ratio, beta=(1-contrast_ratio) * mean)
    
    return contrast_adjusted_image


# Clean pictures in the folder 'datasets'
def data_cleaning():
    
    # Getting current directory
    current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    
    # Getting path of the image file under the folder 'datasets'
    path_folder_pics = os.path.join(current_dir, 'bias_datasets')
    
    # Defining valid image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic']
    
    # Loop through all directories and files in the dataset
    for root, dirs, files in os.walk(path_folder_pics):
        for filename in files:
            
            # Renaming file path
            f = os.path.join(root, filename)
            
            # Verifying if it is a file and valid image extension
            if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in image_extensions):
                # Read image
                img = cv2.imread(f)
                
                # Adjust the brightness
                adjusted_img = adjust_brightness_and_contrast(img)
                
                # Convert to grayscale
                gray_img = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
                
                # Resize to 48x48 pixels
                resized_img = cv2.resize(gray_img, (48, 48))
                
                # Save updated image under the same name (overwrite previous pic)
                cv2.imwrite(f, resized_img)
                
    print("Data cleaning complete.")
    
data_cleaning()

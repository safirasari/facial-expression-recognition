import os
import cv2

# Clean pictures in the folder 'datasets'
def data_cleaning():
    
    # Getting current directory
    current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    
    # Getting path of the image file under the folder 'datasets'
    path_folder_pics = os.path.join(current_dir, 'datasets')
    
    # Defining valid image extensions
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.heic']
    
    # Loop through all directories and files in the dataset
    for root, dirs, files in os.walk(path_folder_pics):
        for filename in files:
            
            # Renaming file path
            f = os.path.join(root, filename)
            
            # Verifying if it is a file and valid image extension
            if os.path.isfile(f) and any(f.lower().endswith(ext) for ext in image_extensions):
                # Read image
                img = cv2.imread(f)
                
                # Convert to grayscale
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Resize to 48x48 pixels
                resized_img = cv2.resize(gray_img, (48, 48))
                
                # Save updated image under the same name (overwrite previous pic)
                cv2.imwrite(f, resized_img)
                
    print("Data cleaning complete.")
    
data_cleaning()

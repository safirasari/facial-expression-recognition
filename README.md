# COMP472 - SmartClass A.I.ssistant
Link: https://github.com/SafiraSari/COMP472

## Members of Group FS_15
- Alexanne Marcil - 40248764
- Safira Sari - 40249017
- Nguyen-My-Linh Tang - 40229505


## Part 1: Data Collection, Cleaning, Labeling & Preliminary Analysis
This project aims to create an AI for a Deep Learning Convolutional Neural Network (CNN) to analyze facial expressions, notably neutral, focused, angry & happy. In part 1, datasets have been taken of various classes, used for training and testing the model. To ensure that the pictures in the dataset are consistent, data cleaning was applied, ensuring that all images are in grayscale and the same dimension (48x48 pixels) for standardization. Data is also visualized.


## Features
- Datasets of 4 facial expressions: neutral, focused, angry & happy
- Data cleaning
- Data visualization


## Deliverables
- Python Code: Includes the scripts for data cleaning (`data_cleaning.py`) and data visualization (`barChart.py`)
- Dataset: Sourced from Kaggle - https://www.kaggle.com/datasets/msambare/fer2013/data
- README: Overview of the project containing instructions on running the program
- Report: Project report detailing each section (PDF)
- Originality Form: 1 for each member included in the report
  

## Setup
This Python project was made using Spyder, Anaconda Prompt and the virtual environment created in class named `comp472`
For the scripts to run correctly, the datasets need to be in the same relative directory to the script as in this repository :
    ```
    project_directory/
    ├─ datasets/
    │  ├─ test/
    │  │  ├─ angry/
    │  │  ├─ happy/
    │  │  ├─ focused/
    │  │  ├─ neutral/
    │  ├─ train/
    │  │  ├─ angry/
    │  │  ├─ happy/
    │  │  ├─ neutral/
    │  │  ├─ focused/
    ```

1. Activate the virtual environment:
   ```
   $ conda activate comp472
   ```
   
2. Install the required libraries:
   - OpenCV
   ```
   $ conda install -c conda-forge opencv
   ```
   - Matplotlib
   ```
   $ conda install matplotlib
   ```
   - Scikit-learn
    ```
    $ conda install scikit-learn
    ```

3. Open the repository in which the Python script is in:
    ```
    $ cd "repository"
    ```

4. Run the Python script for data cleaning and standardization:
    ```
    $ python data_cleaning.py
    ```
   
5.  Run the Python scripts for data visualization:
    ```
    $ python barChart.py
    ```
    ```
    $ python aggregated_pixel_intensity.py
    ```
    ```
    $ python sample_images_neutral.py
    $ python sample_images_angry.py
    $ python sample_images_focused.py
    $ python sample_images_happy.py
    ```

   

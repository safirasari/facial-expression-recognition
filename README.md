# COMP472 - SmartClass A.I.ssistant
Link: https://github.com/SafiraSari/COMP472

## Members of Group FS_15
- Alexanne Marcil - 40248764
- Safira Sari - 40249017
- Nguyen-My-Linh Tang - 40229505


## Project
This project aims to create an AI for a Deep Learning Convolutional Neural Network (CNN) to analyze facial expressions, notably neutral, focused, angry & happy. 

### Part 1:  Data Collection, Cleaning, Labeling & Preliminary Analysis
Datasets have been taken of various classes, used for training and testing the model. To ensure that the pictures in the dataset are consistent, data cleaning was applied, ensuring that all images are in grayscale and the same dimension (48x48 pixels) for standardization. Data is also visualized, including the class distribution, pixel intensity disribution and sample images. 

### Part 2: Basic CNN Model & Evaluation
Using PyTorch, the CNN architecture is defined through the optimization of various hyper-parameters, such as the number of epochs, kernel size, learning rate, number of convolution layers, and more. To prevent overfitting, dropout values were included, along with early stopping. The model underwent a training phase, a validation phase and a testing phase - the overall test accuracy was then measured to assess its prediction on unseen data of facial expressions. 3 types of models are saved: Main Model, Variant 1 and Variant 2. It is then evaluated through precision metrics and plotting the confusion matrix of each class.


## Features
- Datasets of 4 facial expressions: neutral, focused, angry & happy
- Data cleaning & visualization
- CNN Model & Evaluation (precision metrics and confusion matrices)


## Deliverables
- Python Code: Includes the scripts for data cleaning and data visualization
- Dataset: Sourced from Kaggle - https://www.kaggle.com/datasets/msambare/fer2013/data
- README: Overview of the project containing instructions on running the program
- Report: Project report detailing each section (PDF)
- Originality Form: 1 for each member included in the report
  

## Executing the code
### Setup
This Python project was made using Spyder, Anaconda Prompt and the virtual environment created in class named `comp472`
For the scripts to run correctly, the datasets need to be in the same relative directory to the script and the folders set up the same way as in this repository :

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
    - PyTorch
    ```
    $ conda install pytorch torchvision -c pytorch
    ```
    - Seaborn
    ```
    $ conda install seaborn
    ```

3. Open the repository in which the Python script is in:
    ```
    $ cd "repository"
    ```
### Part 1
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
### Part 2
6. Run the Python script to train the main model:
    ```
    $ python cnn.py
    ```
    The main model will be saved in the _models_ folder under the name _best_main_model.ph_.
7. Run the Python scripts to train the variant models:
    ```
  
    ```
    ```
  
    ```
8. Run the Python script to load and run the models:
    ```
    $ python load_and_run.py --Model MODEL --Data DATA
    ```
   - The Model parameter (-m or --Model) expects _main_, _v1_ or _v2_. This is the model that will run.
   - The Data parameter (-d or --Data) expects either _test_, _validation_, or any image name. Providing an image name will run the model on the image and output the prediction, and the other two will evaluate the model on the input set.
   

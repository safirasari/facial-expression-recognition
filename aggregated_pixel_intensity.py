import matplotlib.pyplot as plot
import numpy as np
import os
import cv2

classes = ["angry", "happy", "focused", "neutral"]
currentDir = os.path.normpath(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd())
imageExt = ['.jpg', '.jpeg', '.png', '.heic']
datasetPath = os.path.join(currentDir, 'datasets')

figure, axis = plot.subplots(2, 2)
subplotAxis = [(0,0), (0,1), (1,0), (1,1)]
i = 0;

def directory_images_to_array(path):
    x = []
    for file in os.scandir(path):
        filePath = file.path
        fileName = file.name
        # Verifying if it is a file and valid image extension
        if os.path.isfile(filePath) and any(fileName.lower().endswith(ext) for ext in imageExt):
            currentImage = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            x = np.concatenate((x,currentImage.ravel()))
    return x

            
def histClass(imageClass):
    global i
    classTrainPath = os.path.join(datasetPath, "train", imageClass)
    classTestPath = os.path.join(datasetPath, "test", imageClass)
    classTrainArray = directory_images_to_array(classTrainPath)
    classTestArray = directory_images_to_array(classTestPath)
    classArray = np.concatenate((classTrainArray, classTestArray))
    x, y = subplotAxis[i]
    i += 1
    axis[x, y].hist(x=classArray, bins=256, range=[0,256])
    axis[x,y].set_title(imageClass.capitalize())
    axis[x,y].set_ylabel("Number of Pixels")
    axis[x,y].set_xlabel("Pixel Intensity")

    
for emotionClass in classes:
    histClass(emotionClass)
plot.show()
from barChart import total_count
from random import randint
import os
import matplotlib.pyplot as plot
import cv2

def generate_15_random_numbers(maxValue):
    randomNumbers = []
    for x in range(15):
        currentNumber = randint(0, maxValue-1)
        while currentNumber in randomNumbers:
            currentNumber = randint(0, maxValue-1)
        randomNumbers.append(currentNumber)
    return randomNumbers

def create_plot(imageClass, datasetPath):
    figure, axis = plot.subplots(5, 6)
    subplotAxis = [(0,0), (0,2), (0,4), (1,0), (1,2), (1,4), (2,0), (2,2), (2,4), (3,0), (3,2), (3,4), (4,0), (4,2), (4,4)]
    axisIndex = 0;
    totalNumberOfImages = total_count(imageClass)
    imagesToPlot = generate_15_random_numbers(totalNumberOfImages)
    classPath = os.path.join(datasetPath, imageClass)
    imageFiles = list(os.scandir(classPath))
    selectedFiles = [imageFiles[i] for i in imagesToPlot]
    
    for file in selectedFiles:
        create_image_plots(file.path, subplotAxis, axisIndex, figure, axis)
        axisIndex += 1
    return figure

def create_image_plots(imagePath, subplotAxis, axisIndex, figure, axis):
    x, imageY = subplotAxis[axisIndex]
    histY = imageY + 1
    currentImage = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    axis[x, imageY].imshow(currentImage, cmap='gray')
    axis[x, imageY].axis('off')
    axis[x, histY].hist(x=currentImage.ravel(), bins=256, range=[0,256])
    axis[x,histY].set_ylabel("Number of Pixels")
    axis[x,histY].set_xlabel("Pixel Intensity")


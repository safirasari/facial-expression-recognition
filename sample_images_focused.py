import matplotlib.pyplot as plot
import os
from sample_images import create_plot

imageClass = "focused"
currentDir = os.path.normpath(os.path.dirname(__file__) if "__file__" in locals() else os.getcwd())
imageExt = ['.jpg', '.jpeg', '.png', '.heic']
datasetPath = os.path.join(currentDir, 'datasets')

if __name__ == "__main__":
    figure = create_plot(imageClass, datasetPath)
    figure.suptitle(imageClass.capitalize())
    plot.show()
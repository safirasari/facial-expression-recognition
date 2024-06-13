import matplotlib.pyplot as plt
import os
import glob

def count_images_in_folder(folder_path):
    # Define the image extensions we want to count
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.heic']
    
    # Initialize the image count
    image_count = 0
    
    # Iterate over each extension and count matching files
    for extension in image_extensions:
        # Use glob to find files with the current extension
        files = glob.glob(os.path.join(folder_path, extension))
        # Add the number of files found to the total count
        image_count += len(files)
    return image_count
    

def total_count(emotion):
    # Compute total of images per class
    dir_path = os.path.join(os.getcwd(), 'datasets', emotion) 
    image_number = count_images_in_folder(dir_path)
    return image_number
    
# variables for the graph
x = ["Neutral", "Focused", "Angry", "Happy"]
y = [total_count('neutral'), total_count("focused"), total_count("angry"), total_count("happy")]

# function to add value labels (https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/)
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

# plotting the graph with its characteristics
if __name__ == "__main__":
    plt.bar(x,y)
    addlabels(x,y)
    plt.title("Number of Images per Class in our Dataset")
    plt.xlabel("Emotion Class")
    plt.ylabel("Number of Images")
    plt.show()




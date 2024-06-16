import argparse
import os
import torch
from PIL import Image
import data_loaders as data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from cnn import CNN


def plot_table(modelName, acc, prec, recall, f1, prec_micro, recall_micro, f1_micro): 
    table_data = [["Model","Macro P","Macro R","Macro F","Micro P","Micro R","Micro F","Accuracy"], 
                [modelName, prec, recall, f1, prec_micro, recall_micro, f1_micro, acc]]
    fig, ax = plt.subplots() 
    ax.axis("off") #remove axis
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    for (i, j), cell in table.get_celld().items(): #formatting in bold the first row
        if i == 0:
            cell.set_text_props(fontweight="bold")
    plt.rcParams.update({'font.size': 16})
    plt.show()

def evaluate_model(model, dataLoader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in dataLoader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy()) #
            y_pred.extend(predicted.numpy()) #
    
    cm = confusion_matrix(y_true, y_pred) #generate confusion matrix
    accuracy = accuracy_score(y_true, y_pred) #compute accuracy score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro') #compute P, R, F for macro
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro') #compute P, R, F for micro
    
    #put values to 2 decimals
    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    precision_micro = round(precision_micro, 4)
    recall_micro = round(recall_micro, 4)
    f1_micro = round(f1_micro, 4)

    return cm, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro

#Generate confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.dataset.classes, yticklabels=data.classes) #better visualization https://www.shiksha.com/online-courses/articles/heatmap-in-seaborn/#:~:text=The%20primary%20purpose%20of%20the,the%20features%20in%20the%20data.
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Find the image within the given folder and return its folder and its path
def find_image(dataPath, imageName):
    for dirPath, _, fileNames in os.walk(dataPath):
        for fileName in fileNames:
            if fileName == imageName:
                imagePath = os.path.join(dirPath, fileName)
                imageClass = dirPath.split('\\')[-1]
                return imagePath, imageClass
    return None
            
if __name__ == '__main__':

    # Define script arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--Model", default='main', choices=['main', 'v1', 'v2'], help="Select model to run")
    parser.add_argument("-d", "--Data", default='test', help="Select between test and validation to evaluate the model, or select an image file to run the model on")
    args = parser.parse_args()
    
    #Load chosen model
    if args.Model == "main":
        model = CNN()
        model.load_state_dict(torch.load("./models/best_main_model.pt"))
        modelName = "Main Model"
    elif args.Model == "v1":
        model = torch.load("v1.pt")
        modelName = "Variant 1"
    elif args.Model == "v2":
        model = torch.load("v2.pt")
        modelName = "Variant 2"

    classesInOrder = ('angry', 'focused', 'happy', 'neutral')
    if args.Data != "test" and args.Data != "validation":
        # Find and load the image
        imageName = args.Data
        print(modelName,"predicting the class of image",imageName)
        current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        path_folder_pics = os.path.join(current_dir, 'datasets')
        imagePath, imageClass = find_image(path_folder_pics, imageName)
        if imagePath is None:
            print("Image",imageName,"not found.")
        imageToPredict = Image.open(imagePath)
        imageToPredict = data.transform(imageToPredict).unsqueeze(0) # not sure about unsqueeze, adds batch dimension to image

        # Predict image class with loaded model
        model.eval()
        with torch.no_grad():
            output = model(imageToPredict)
            _, prediction = torch.max(output, 1)
        print("Actual class:",imageClass)
        print(modelName,"predicted: ",classesInOrder[prediction.item()])

    # Load data loaders if test or validation is chosen
    else:
        if args.Data == "test":
            dataLoader = data.test_loader
        elif args.Data == "validation":
            dataLoader = data.val_loader

         # Evaluate models
        cm, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro = evaluate_model(model, dataLoader)

        # Plot confusion matrices
        plot_confusion_matrix(cm, title='Confusion Matrix - ' + modelName)
        plot_table(modelName, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro)




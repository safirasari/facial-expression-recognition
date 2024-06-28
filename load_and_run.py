import argparse
import os
import torch
from PIL import Image
import data_loaders as data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from mainmodel import CNN
from variant1 import CNN_V1
from variant2 import CNN_V2


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
    print(cm)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.dataset.classes, yticklabels=data.dataset.classes) #better visualization https://www.shiksha.com/online-courses/articles/heatmap-in-seaborn/#:~:text=The%20primary%20purpose%20of%20the,the%20features%20in%20the%20data.
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

    parser.add_argument("-m", "--Model", default='main', choices=['main', 'v1', 'v2', 'fold1', 'fold2', 'fold3', 'fold4','fold5','fold6','fold7', 'fold8', 'fold9', 'fold1','fold10'], help="Select model to run")
    parser.add_argument("-d", "--Data", default='test', help="Select between test and validation to evaluate the model, select an image file to run the model on, or select a demographic group to evaluate the model on")
    args = parser.parse_args()
    
    #Load chosen model
    if args.Model == "main":
        model = CNN()
        model.load_state_dict(torch.load("./models/best_main_model.pt"))
        modelName = "Main Model"
    elif args.Model == "v1":
        model = CNN_V1()
        model.load_state_dict(torch.load("./models/best_v1.pt"))
        modelName = "Variant 1"
    elif args.Model == "v2":
        model = CNN_V2()
        model.load_state_dict(torch.load("./models/best_v2.pt"))
        modelName = "Variant 2"
    elif args.Model == "fold1":
        model = CNN_F1()
        model.load_state_dict(torch.load("./models/model_fold_1.pt"))
        modelName = "Fold 1"
    elif args.Model == "fold2":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_2.pt"))
        modelName = "Fold 2"
    elif args.Model == "fold3":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_3.pt"))
        modelName = "Fold 3"
    elif args.Model == "fold4":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_4.pt"))
        modelName = "Fold 4"
    elif args.Model == "fold5":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_5.pt"))
        modelName = "Fold 5"
    elif args.Model == "fold6":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_6.pt"))
        modelName = "Fold 6"
    elif args.Model == "fold7":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_7.pt"))
        modelName = "Fold 7"
    elif args.Model == "fold8":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_8.pt"))
        modelName = "Fold 8"
    elif args.Model == "fold9":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_9.pt"))
        modelName = "Fold 9"
    elif args.Model == "fold10":
        model = CNN()
        model.load_state_dict(torch.load("./models/model_fold_10.pt"))
        modelName = "Fold 10"

    classesInOrder = ('angry', 'focused', 'happy', 'neutral')
    dataChoices = ("test", "validation", "middle-aged", "senior", "young", "female", "male", "other")
    if args.Data not in dataChoices:
        # Find and load the image
        imageName = args.Data
        print(modelName,"predicting the class of image",imageName)
        current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
        path_folder_pics = os.path.join(current_dir, 'datasets')
        imagePath, imageClass = find_image(path_folder_pics, imageName)
        if imagePath is None:
            print("Image",imageName,"not found.")
        imageToPredict = Image.open(imagePath)
        imageToPredict = data.transform(imageToPredict).unsqueeze(0)

        # Predict image class with loaded model
        model.eval()
        with torch.no_grad():
            output = model(imageToPredict)
            _, prediction = torch.max(output, 1)
        print("Actual class:",imageClass)
        print(modelName,"predicted: ",classesInOrder[prediction.item()])

    # Load data loaders if test or validation is chosen
    else:
        needcm = False
        if args.Data == "test":
            dataLoader = data.test_loader
            needcm = True
        elif args.Data == "validation":
            dataLoader = data.val_loader
            needcm = True
        elif args.Data == "middle-aged":
            dataLoader = data.middleaged_loader
        elif args.Data == "senior":
            dataLoader = data.senior_loader
        elif args.Data == "young":
            dataLoader = data.young_loader
        elif args.Data == "female":
            dataLoader = data.female_loader
        elif args.Data == "male":
            dataLoader = data.male_loader
        elif args.Data == "other":
            dataLoader = data.othergender_loader

         # Evaluate models
        cm, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro = evaluate_model(model, dataLoader)

        # Plot confusion matrices
        if needcm:
            plot_confusion_matrix(cm, title='Confusion Matrix - ' + modelName)
        else:
            print("Number of images in",args.Data,":",len(dataLoader.sampler.indices))
        plot_table(modelName, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro)




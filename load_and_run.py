import argparse
import os
import torch
from PIL import Image as img
from cnn import transform

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--Model", default='main', choices=['main', 'v1', 'v2'], help="Select model to run")
parser.add_argument("-d", "--Data", default='test', help="Select between test and validation to evaluate the model, or select an image file to run the model on")
args = parser.parse_args()


if args.Model == "main":
    model = torch.load("main.pt")
elif args.Model == "v1":
    model = torch.load("v1.pt")
elif args.Model == "v2":
    model = torch.load("v2.pt")

if args.Data != "test" and args.Data != "validation":
    current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    path_folder_pics = os.path.join(current_dir, 'datasets')
    imageName = args.Data


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy()) #
            y_pred.extend(predicted.numpy()) #
    
    cm = confusion_matrix(y_true, y_pred) #generate confusion matrix
    accuracy = accuracy_score(y_true, y_pred) #compute accuracy score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro') #compute P, R, F for macro
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro') #compute P, R, F for micro
    
    #put values to 2 decimals
    accuracy = round(accuracy, 2)
    precision = round(precision, 2)
    recall = round(recall, 2)
    f1 = round(f1, 2)
    precision_micro = round(precision_micro, 2)
    recall_micro = round(recall_micro, 2)
    f1_micro = round(f1_micro, 2)

    return cm, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro

#function to generate confusion matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes) #better visualization https://www.shiksha.com/online-courses/articles/heatmap-in-seaborn/#:~:text=The%20primary%20purpose%20of%20the,the%20features%20in%20the%20data.
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Evaluate models
cm_main, acc_main, prec_main, recall_main, f1_main, prec_micro_main, recall_micro_main,  f1_micro_main = evaluate_model(modelA, test_loader)
#cm_variant1, acc_variant1, prec_variant1, recall_variant1, f1_variant1, prec_micro_variant1, recall_micro_variant1, f1_micro_variant1 = evaluate_model(variant1_model, test_loader)
#cm_variant2, acc_variant2, prec_variant2, recall_variant2, f1_variant2, prec_micro_variant2, recall_micro_variant2, f1_micro_variant2 = evaluate_model(variant2_model, test_loader)

# Plot confusion matrices
plot_confusion_matrix(cm_main, title='Confusion Matrix - Main Model')
#plot_confusion_matrix(cm_variant1, title='Confusion Matrix - Variant 1')
#plot_confusion_matrix(cm_variant2, title='Confusion Matrix - Variant 2')


def plot_table(): 
    table_data = [["Model","Macro P","Macro R","Macro F","Micro P","Micro R","Micro F","Accuracy"], 
                ["Main Model", prec_main, recall_main, f1_main, prec_micro_main, recall_micro_main, f1_micro_main, acc_main], 
                ["Variant 1", 5, 6, 7, 1, 2, 3, 8], 
                ["Variant 2", 9, 10, 11, 1, 2, 3, 12]]
    fig, ax = plt.subplots() 
    ax.axis("off") #remove axis
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    for (i, j), cell in table.get_celld().items(): #formatting in bold the first row
        if i == 0:
            cell.set_text_props(fontweight="bold")
    plt.rcParams.update({'font.size': 16})
    plt.show()

plot_table()

def find_subfolder(imageName):

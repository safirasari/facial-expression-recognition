import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from skorch import NeuralNetClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from mainmodel import CNN
import data_loaders as data
from load_and_run import evaluate_model

def kfold_cross_validation(model, dataset):
    
    # Set random seed
    torch.manual_seed(42)
    
    num_epochs = 10
    learning_rate = 0.003
    kfold_num = 10
    batch_size = 32
    
    best_val_loss = None
    best_epoch = 0 
    patience = 5
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Evaluation metrics
    accuracy_fold = []
    precision_micro_fold = []
    recall_micro_fold = []
    f1_micro_fold = []

    precision_macro_fold = []
    recall_macro_fold = []
    f1_macro_fold = []

    kf = KFold(n_splits=kfold_num, shuffle=True, random_state=0)

    fold_num = 1
    

    # Create a directory to save the models
    if not os.path.exists('./models'):
        os.makedirs('./models')

    # K-fold cross-validation
    for train_idx, test_idx in kf.split(dataset):
        
        print(f"FOLD: {fold_num}/{kfold_num}")
        
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)
       
        # Splitting training data --> train and validation subsets
        val_split = int(0.15 * len(train_dataset))
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [len(train_dataset) - val_split, val_split])
       
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        total_step = len(data.train_loader)
        loss_list = []
        acc_list = []
        total = 0
        
        # Initialize a new model before training each fold
        model = CNN()
        model.train()
            
        print('\nTRAINING PHASE:')
        for epoch in range(num_epochs):
            
            for i, (images, labels) in enumerate(train_loader):
    
                # Forward pass
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                loss_list.append(loss.item())
                
                # Backprop and optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Train accuracy
                total = labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).sum().item()
                acc_list.append(correct / total)
                
                if (i + 1) % 10 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),(correct / total) * 100))

            # Validation
            model.eval()
            val_correct = 0
            val_loss = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    val_loss += criterion(outputs, labels).item() * labels.size(0)
            val_loss /= val_total
            
            val_acc = (val_correct / val_total) * 100
            print('Validation Accuracy of the model on the validation images: {} %'.format(val_acc))
            print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss))

            # Early stopping
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                # torch.save(model.state_dict(), "./models/kfold.pt")
                
            elif epoch - best_epoch > patience:
                print("Stopped training at epoch ",epoch + 1)
                print("Main model saved at epoch ",best_epoch)
                break

         # Save the model for the current fold
        torch.save(model.state_dict(), f"./models/model_fold_{fold_num}.pt")
        print(f"Fold {fold_num}: Training completed and model saved.\n")


        # Set model to evaluation
        print('\nTESTING PHASE: ')
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            correct = 0
            total = 0
            class_correct = [0 for i in range(4)]
            class_total = [0 for i in range(4)]
                   
            for images, labels in test_loader:
                
                # Prediction
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                y_true.extend(labels.numpy()) #
                y_pred.extend(predicted.numpy()) #
                
                # Calculating accuracy of each class
                for label, pred in zip(labels, predicted):
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1
                        
            # Displaying accuracy overall
            accuracy = (correct / total) * 100
            print(f'Correct: {correct}')
            print(f'Total: {total}')
            print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
            

        # Performance metrics
        accuracy_fold.append(accuracy_score(y_true, y_pred))
        
        # Micro metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        precision_micro_fold.append(precision_micro)
        recall_micro_fold.append(recall_micro)
        f1_micro_fold.append(f1_micro)
        
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        precision_macro_fold.append(precision_macro)
        recall_macro_fold.append(recall_macro)
        f1_macro_fold.append(f1_macro)
        
        print("Micro values: ")
        print(f"Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
        print("Macro values: ")
        print(f"Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        
        accuracy_fold.append(accuracy)
        precision_micro_fold.append(precision_micro)
        recall_micro_fold.append(recall_micro)
        f1_micro_fold.append(f1_micro)
        precision_macro_fold.append(precision_macro)
        recall_macro_fold.append(recall_macro)
        f1_macro_fold.append(f1_macro)
        
        print(f'Fold {fold_num} Test Accuracy: {accuracy:.2f}%')
        
        # Update fold number
        fold_num += 1
        
    
    # Calculate average of performance metrics
    avg_accuracy = sum(accuracy_fold) / len(accuracy_fold)
    avg_precision_micro = sum(precision_micro_fold) / len(precision_micro_fold)
    avg_recall_micro = sum(recall_micro_fold) / len(recall_micro_fold)
    avg_f1_micro = sum(f1_micro_fold) / len(f1_micro_fold)

    avg_precision_macro = sum(precision_macro_fold) / len(precision_macro_fold)
    avg_recall_macro = sum(recall_macro_fold) / len(recall_macro_fold)
    avg_f1_macro = sum(f1_macro_fold) / len(f1_macro_fold)

    print("Average values:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print("Micro values: ")
    print(f"Average Precision: {avg_precision_micro:.4f}, Average Recall: {avg_recall_micro:.4f}, Average F1: {avg_f1_micro:.4f}")
    print("Macro values: ")
    print(f"Average Precision: {avg_precision_macro:.4f}, Average Recall: {avg_recall_macro:.4f}, Average F1: {avg_f1_macro:.4f}")


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # 1 channel for grayscale
        transforms.ToTensor(),                          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))            # Center data around 0 (instead of [0,1])
    ])

    # Loading the dataset
    dataset = ImageFolder(root='./datasets', transform=transform)

    # Initialize the model
    model = CNN()
    model.load_state_dict(torch.load("./models/best_main_model.pt"))

    # Train the model
    kfold_cross_validation(model, dataset)

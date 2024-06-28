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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


from mainmodel import CNN
import data_loaders as data
from load_and_run import evaluate_model

def kfold_cross_validation(model, dataset):
    
    # Set random seed
    torch.manual_seed(42)
    
    num_epochs = 5
    learning_rate = 0.003
    patience = 5
    kfold_num = 5
    batch_size = 32
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
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
    
    # K-fold cross-validation
    for train, test in kf.split(dataset):

        total_step = len(data.train_loader)
        loss_list = []
        acc_list = []
        total = 0

        print(f"Fold: {fold_num}/{kfold_num}")
           
        print('\nTRAINING PHASE:')
        for epoch in range(num_epochs):
            
            model.train()              # Training mode
            
            for i, (images, labels) in enumerate(data.train_loader):
                
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
                for images, labels in data.val_loader:
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

        print(f"Fold {fold_num}: Training completed.")
        
    
        
        # Evaluation
        print('\nTESTING PHASE: ')
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            class_correct = [0 for i in range(4)]
            class_total = [0 for i in range(4)]
                   
            for images, labels in data.test_loader:
                
                # Prediction
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
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

        # Performance metrics for 1 fold
        accuracy = accuracy_score(all_labels, all_preds)
        precision_micro = precision_score(all_labels, all_preds, average='micro')
        recall_micro = recall_score(all_labels, all_preds, average='micro')
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        precision_macro = precision_score(all_labels, all_preds, average='macro')
        recall_macro = recall_score(all_labels, all_preds, average='macro')
        f1_macro = f1_score(all_labels, all_preds, average='macro')

        # Adding to list
        accuracy_fold.append(accuracy)
        precision_micro_fold.append(precision_micro)
        recall_micro_fold.append(recall_micro)
        f1_micro_fold.append(f1_micro)
        precision_macro_fold.append(precision_macro)
        recall_macro_fold.append(recall_macro)
        f1_macro_fold.append(f1_macro)
        
        print(f'Fold {fold_num} Test Accuracy: {accuracy * 100:.2f}%')
        
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

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


from mainmodel import CNN
import data_loaders as data
from load_and_run import evaluate_model

def kfold_cross_validation(model, dataset):
    
    # Set random seed
    torch.manual_seed(42)
    
    num_epochs = 10
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
        
        
        train_data = torch.utils.data.Subset(dataset, train)
        test_data = torch.utils.data.Subset(dataset, test)
        
        # Data Loaders:
        train_load = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_load = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


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
        
        
        # Evaluating the fold
        
        model.eval()
        predictions = []
        actual = []
        with torch.no_grad():
            for images, labels in test_load:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
                actual.extend(labels.cpu().numpy())

        # Performance metrics for 1 fold
        
        # cm, accuracy, precision, recall, f1, precision_micro, recall_micro, f1_micro = evaluate_model(model, dataLoader)
        

      
        # Update fold number
        fold_num += 1
        
    
    # Calculate average of performance metrics



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

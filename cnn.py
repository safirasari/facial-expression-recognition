import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    
    # Hyper-parameters
    num_epochs = 15         # Min of 10 epochs
    num_classes = 4         # 4 classes: neutral, focused, angry, happy
    learning_rate = 0.001
    
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 1000
    
    # Transformation to ensure consistency
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),    # 1 channel for grayscale
        transforms.ToTensor(),                          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))            # Center data around 0 (instead of [0,1])
    ])
    

    # Load the dataset
    dataset = ImageFolder(root='./datasets', transform=transform)
    
    # Split dataset into sets of train, validation, and test
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Sizes of each split
    num_samples = len(dataset)
    num_train = int(train_ratio * num_samples)
    num_val = int(val_ratio * num_samples)
    num_test = num_samples - num_train - num_val
    
    # Create indices for each split
    indices = list(range(num_samples))
    train_indices, remaining_indices = train_test_split(indices, test_size=(val_ratio + test_ratio), random_state=42)
    val_indices, test_indices = train_test_split(remaining_indices, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
    
    # Create data samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Define batch sizes
    train_batch_size = 32
    val_batch_size = 32
    test_batch_size = 1000
    
    # Create DataLoaders for each set
    train_loader = DataLoader(dataset, batch_size=train_batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=val_batch_size, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=test_sampler)
    
    classes = ('neutral', 'focused', 'angry', 'happy')

    print("Number of images in dataset:", len(dataset))

    # Defining different layers of the network
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv_layer = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
    
            self.fc_layer = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(12 * 12 * 64, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1),
                nn.Linear(512, 4)       # 4 classes
            )
            
        def forward(self, x):
            # conv layers
            x = self.conv_layer(x)
            # flatten
            x = x.view(x.size(0), -1)
            # fc layer
            x = self.fc_layer(x)
            
            return x
    # End of CNN subclass


    # Creating the model
    modelA = CNN()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelA.parameters(), lr=learning_rate)
    
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    
    # Training the model
    for epoch in range(num_epochs):
        
        for i, (images, labels) in enumerate(train_loader):
            
            # Forward pass
            outputs = modelA(images)
            
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
             
                
    # Set model to evaluation
    modelA.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = [0 for i in range(4)]
        class_total = [0 for i in range(4)]
               
        for images, labels in test_loader:
            
            # Prediction
            outputs = modelA(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Calculating accuracy of each class
            '''
            for i in range(num_classes):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    class_correct[label] += 1
                class_total[label] += 1
            '''  
            
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
                    
        # Displaying accuracy overall
        accuracy = (correct / total) * 100
        print(f'Correct: {correct}')
        print(f'Total: {total}')
        print('Test Accuracy of the model on the test images: {} %'.format(accuracy))
        
        # Displaying accuracy for each class
        for i in range(4):
            print(f'Class_correct: {class_correct[i]}')
            print(f'Class_total: {class_total[i]}')
            
            if class_total[i] != 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                print(f'Accuracy of {classes[i]}: {accuracy} %')
            else:
                print(f'No instances of class {classes[i]} in the test set.')

#function to evaluate the model
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

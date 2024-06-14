import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
# import seaborn as sns

if __name__ == '__main__':
    
    # Hyper-parameters
    num_epochs = 15         # Min of 10 epochs
    num_classes = 4         # 4 classes: neutral, focused, angry, happy
    learning_rate = 0.0005
    
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
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            self.fc_layer = nn.Sequential(
                nn.Dropout(p=0.5),
                # nn.Linear(64 * 10 * 10, 1000),         # kernel 2x2
                nn.Linear(12 * 12 * 64, 1000),         # kernel 3x3, 5x5, 7x7
                nn.ReLU(inplace=True),
                nn.Linear(1000, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
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
    print('\nTRAINING PHASE:')
    for epoch in range(num_epochs):
        
        modelA.train()              # Training mode
        
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
                
        # Validation phase
        modelA.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = modelA(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = (val_correct / val_total) * 100
        print('Validation Accuracy of the model on the validation images: {} %'.format(val_acc))
        
                
    # Set model to evaluation
    print('\nTESTING PHASE: ')
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

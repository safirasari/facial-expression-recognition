import torch
import torch.nn as nn
import data_loaders as data

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
            nn.Dropout(p=0.25),
            # nn.Linear(64 * 10 * 10, 1000),         # kernel 2x2
            nn.Linear(12 * 12 * 64, 1000),         # kernel 3x3, 5x5, 7x7
            nn.ReLU(inplace=True),
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
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

if __name__ == '__main__':
    
    # Hyper-parameters
    num_epochs = 15         # Min of 10 epochs
    num_classes = 4         # 4 classes: neutral, focused, angry, happy
    learning_rate = 0.0003

    classes = ('neutral', 'focused', 'angry', 'happy')

    print("Number of images in dataset:", len(data.dataset))

    # Creating the model
    modelA = CNN()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(modelA.parameters(), lr=learning_rate)
    
    total_step = len(data.train_loader)
    loss_list = []
    acc_list = []
    total = 0

    best_val_loss = None
    best_epoch = 0
    patience = 5

    # Training the model
    print('\nTRAINING PHASE:')
    for epoch in range(num_epochs):
        
        modelA.train()              # Training mode
        
        for i, (images, labels) in enumerate(data.train_loader):
            
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
        val_loss = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in data.val_loader:
                outputs = modelA(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += criterion(outputs, labels).item() * labels.size(0)
        val_loss /= val_total
        
        val_acc = (val_correct / val_total) * 100
        print('Validation Accuracy of the model on the validation images: {} %'.format(val_acc))
        print('Epoch [{}/{}], Validation Loss: {:.4f}'
                .format(epoch + 1, num_epochs, val_loss))
        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(modelA.state_dict(), "./models/best_main_model.pt")
        elif epoch - best_epoch > patience:
            print("Stopped training at epoch ",epoch + 1)
            print("Main model saved at epoch ",best_epoch)
            break
        
        
                
    # Set model to evaluation
    print('\nTESTING PHASE: ')
    modelA.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        class_correct = [0 for i in range(4)]
        class_total = [0 for i in range(4)]
               
        for images, labels in data.test_loader:
            
            # Prediction
            outputs = modelA(images)
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
        
        # Displaying accuracy for each class
        for i in range(4):
            print(f'Class_correct: {class_correct[i]}')
            print(f'Class_total: {class_total[i]}')
            
            if class_total[i] != 0:
                accuracy = 100.0 * class_correct[i] / class_total[i]
                print(f'Accuracy of {classes[i]}: {accuracy} %')
            else:
                print(f'No instances of class {classes[i]} in the test set.')

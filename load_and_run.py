import argparse
import os
import torch

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--Model", default='main', choices=['main', 'v1', 'v2'], help="Select model to run")
parser.add_argument("-d", "--Data", default='test', help="Select between test and validation to evaluate the model, or select an image file to run the model on")
args = parser.parse_args()

if args.Data != "test" and args.Data != "validation":
    current_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    path_folder_pics = os.path.join(current_dir, 'datasets')
    imageName = args.Data
    # find image and load

if args.Model == "main":
    model = torch.load("main.pt")
elif args.Model == "v1":
    model = torch.load("v1.pt")
elif args.Model == "v2":
    model = torch.load("v2.pt")


model.eval()
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
        
        
        for label, pred in zip(labels, predicted):
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1
                
    # Displaying accuracy overall
    accuracy = (correct / total) * 100
    print(f'Correct: {correct}')
    print(f'Total: {total}')
    print('Test Accuracy of the model on the test images: {} %'.format(accuracy))

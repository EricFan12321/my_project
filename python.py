import sys
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from resnet import ResNet
sys.path.append(os.path.dirname(os.path.abspath(__file__))) #Add the Current Directory to the Python Path
# Print the Python path to verify
print("Python Path:", sys.path)

# Define a function to load pretrained ResNet20 model
def load_pretrained_resnet20(filepath):
    checkpoint = torch.load(filepath)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 10)
    classifier.load_state_dict(state)
    return classifier

def load_cifar10_data(batch_size):
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# Load Data
trainloader, testloader = load_cifar10_data(batch_size=32)

# Load Model
model = load_pretrained_resnet20(filepath='cifar10-r20.pth.tar')

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Compute the accuracy of the model on the test set
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)  # Move images to the same device as the model
        labels = labels.to(device)  # Move labels to the same device as the model
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

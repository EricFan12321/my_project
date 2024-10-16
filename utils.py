
# Define a function to load cifar10 data

import torch
import torchvision
import torchvision.transforms as transforms
def load_cifar10_data(batch_size=32):
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


# Define a function to load pretrained ResNet20 model
import torch
import torch.nn as nn
import torchvision.models as models

def load_pretrained_resnet20(filepath='cifar10-r20.pth.tar'):
    # Define the ResNet-20 model architecture
    class ResNet20(nn.Module):
        def __init__(self):
            super(ResNet20, self).__init__()
            self.model = models.resnet18()  # Placeholder for ResNet-20 architecture
            # Modify the architecture to match ResNet-20 specifics if needed

        def forward(self, x):
            return self.model(x)
    
    # Initialize the model
    model = ResNet20()
    
    # Load the pretrained weights
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model



# Compute lpips distance 
import lpips
import torch

def compute_lpips(original_images, adversarial_images):
    # Load the LPIPS model
    loss_fn = lpips.LPIPS(net='alex')
    
    # Ensure the images are in the correct format (N, C, H, W) and normalized
    original_images = torch.tensor(original_images).permute(0, 3, 1, 2).float() / 255.0
    adversarial_images = torch.tensor(adversarial_images).permute(0, 3, 1, 2).float() / 255.0
    
    # Compute LPIPS distance
    distances = loss_fn(original_images, adversarial_images)
    
    return distances


# calculate the attack success rate
def compute_attack_success_rate(true_labels, original_preds, adversarial_preds):
    # Convert inputs to tensors if they are not already
    true_labels = torch.tensor(true_labels)
    original_preds = torch.tensor(original_preds)
    adversarial_preds = torch.tensor(adversarial_preds)
    
    # Calculate successful attacks
    successful_attacks = (original_preds == true_labels) & (adversarial_preds != true_labels)
    
    # Compute attack success rate
    attack_success_rate = successful_attacks.sum().item() / len(true_labels)
    
    return attack_success_rate


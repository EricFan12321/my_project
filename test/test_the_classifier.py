## This script is to test the accuracy of the pretrained model 
# case 1: pretrained resnet20 on cifar10
# case 2: pretrained resnet20 on cifar100
# case 3: pretrained resnet50 on imagenet_val

#import the neccessary packages
import torch
import torchattacks
import yaml
import sys
import os
from attacks import ssah_attack 
from collections import OrderedDict
from utils.utils import *
from model.resnet import ResNet 


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils import *

# Load configurations
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: Load the data based on the dataset configuration
dataset_root = config.get('dataset_root') # If dataset_root is not specified, it defaults to './data'
batch_size = config.get('bs')    # If bs is not specified, it defaults to 32.
num_workers = config.get('workers') # If workers is not specified, it defaults to 2.

# Dataset: Load the validation data based on the dataset configuration
if config['dataset'] == 'cifar10':
    data, num_images = load_cifar10(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'cifar100':
    data, num_images = load_cifar100(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'imagenet_val':
    data, num_images = load_imagenet_val(dataset_root, batch_size=16, num_workers=4)

#Classifier: Load the classifier based on the dataset
if config['dataset'] == 'cifar10':
    path = "checkpoints/cifar10-r20.pth.tar"
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 10)
    classifier.load_state_dict(state)
elif config['dataset'] == 'cifar100':
    path = "checkpoints/cifar100-r20.pth.tar"
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 100)
    new_state_dict = OrderedDict()
    for k, v in state.items():
        if 'module.' in k:
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)
elif config['dataset'] == 'imagenet_val':
    classifier = torchvision.models.resnet50(pretrained=True)
classifier.eval()                  
classifier = classifier.to(device) 

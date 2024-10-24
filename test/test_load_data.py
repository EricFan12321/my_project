import yaml
from utils.utils import *

# Load configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: Load the data based on the dataset configuration
dataset_root = config.get('dataset_root', './data') # If dataset_root is not specified, it defaults to './data'
batch_size = config.get('bs', 32)    # If bs is not specified, it defaults to 32.
num_workers = config.get('workers', 2) # If workers is not specified, it defaults to 2.

# Dataset: Load the data based on the dataset configuration
if config['dataset'] == 'cifar10':
    data, num_images = load_cifar10(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'cifar100':
    data, num_images = load_cifar100(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'imagenet_val':
    data, num_images = load_imagenet_val(dataset_root, batch_size, num_workers)

# Check if data is successfully loaded
try:
    sample_data, sample_labels = next(iter(data))
    print(f"Data successfully loaded. Sample data shape: {sample_data.shape}, Sample labels shape: {sample_labels.shape}")
except Exception as e:
    print(f"Failed to load data: {e}")
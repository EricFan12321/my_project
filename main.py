#import the neccessary packages
import torch
import torchattacks
from attacks import ssah_attack # where the SSAH is defined
import yaml

from utils.utils import *
from model.resnet import ResNet # where ResNet() is defined


# Load configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset: Load the data based on the dataset configuration
dataset_root = config.get('dataset_root', './data') # If dataset_root is not specified, it defaults to './data'
batch_size = config.get('bs', 32)    # If bs is not specified, it defaults to 32.
num_workers = config.get('workers', 2) # If workers is not specified, it defaults to 2.

# what if for different dataset, (batchsize, num_workers) is different???

# Dataset: Load the data based on the dataset configuration
if config['dataset'] == 'cifar10':
    data, num_images = load_cifar10(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'cifar100':
    data, num_images = load_cifar100(dataset_root, batch_size, num_workers)
elif config['dataset'] == 'imagenet_val':
    data, num_images = load_imagenet_val(dataset_root, batch_size, num_workers)

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
classifier.eval()                  # Switch the classifier to evaluation mode, so its weights will not change
classifier = classifier.to(device) # Load the classifier to the device



# Load and Apply the attacks
for attack_config in config['attacks']:
    if attack_config['mode'] == 'SSAH':
        atk = SSAH(model=classifier,
                   num_iteration=attack_config['num_iteration'],
                   learning_rate=attack_config['learning_rate'],
                   device=device,
                   Targeted=True,
                   dataset=config['dataset'],
                   m=attack_config['m'],
                   alpha=attack_config['alpha'],
                   lambda_lf=attack_config['lambda_lf'],
                   wave=attack_config['wave'])
    elif attack_config['mode'] == 'PGD':
        atk = torchattacks.PGD(model, steps=attack_config['eps'], alpha=attack_config['alpha'], steps=attack_config['steps'],random_start=attack_config['random_start'] )
    elif attack_config['mode'] == 'CW':
        atk = torchattacks.CW(model, c=attack_config['c'], kappa=attack_config['kappa'], steps=attack_config['steps'], lr=attack_config['lr'])
    elif attack_config['mode'] == 'JSMA':
        atk = torchattacks.JSMA(model, theta=attack_config['theta'], gamma=attack_config['gamma'])
       
atk.set_mode_targeted_by_label()

#Get the target labels
new_labels = (labels + 1) % 10


# Evaluation
# Intialize some parameters
total_img = 0
att_suc_img = 0
for batch, (inputs, labels) in enumerate(data): # enumerate(data) provides both the batch index (batch) and the batch data (inputs and labels).
    inputs = inputs.to(device)
    labels = labels.to(device)
    common_id = common(labels, predict(classifier, inputs, opt)) # common_id stores the indices of the correctly classified images by comparing the true labels (labels) with the predictions from the classifier.
    total_img += len(common_id)          # total_img is incremented by the number of correctly classified images.
    inputs = inputs[common_id].cuda()   # The inputs and labels are filtered to include only the correctly classified images and moved to the GPU.
    labels = labels[common_id].cuda()

    # attack
    adv = att(inputs)

    att_suc_id = attack_success(labels, predict(classifier, adv, opt)) # return the id where "predicted labels != true labels"
    att_suc_img += len(att_suc_id)

    adv = adv[att_suc_id]         # only keep the adversarial images where the attack is successful
    inputs = inputs[att_suc_id]   # only keep the original images where the attack is successful

# Compute the overall SAR for the entire test dataset
sar = 100.0 * att_suc_img / total_img
print(f"Overall Success Attack Rate (SAR): {sar:.2f}%")



# define a parse_arg(), so we could 
# 1.swith between attack methods "SSAH", "PGD", "C&W" .etc
# 2.change the parameters: 
# -----batchsize, 
#------dataset('cifar10', ''cifar100, 'imagenet'), 
#------classifier('resnet20_cifar10', 'resnet20_cifar100', 'resnet50_imagenet')
#------perceptual metric('ssim', 'lpips', 'l2' .etc)
#------target mode('targeted' or 'untargeted'), if targeted , pick up a way to generate target labels




# should I set the following parameters same for all attacks?
# number_of_iterations
# alpha
# lambda
# 

# Set some parameters specific to each attack
# if opt.perturb-mode === 'SSAH', we want to pop a window and ask:
# (1) wavelet
# (2) experiment_name

# if opt.perturb-mode === 'SSAH', we want to pop a window and ask:
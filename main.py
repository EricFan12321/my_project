#import the neccessary packages
import torch
import argparse
import resnet # where ResNet() is defined
import torchattacks
import ssah_attack # where the SSAH is defined


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
# 

def parse_arg():
    parser = argparse.ArgumentParser(description='attack with feature layer and frequency constraint')  
    parser.add_argument('--bs', type=int, default=10000, help="batch size")   
    parser.add_argument('--dataset-root', type=str, default='dataset', help='dataset path')
    parser.add_argument('--dataset', type=str, default='cifar10', help='data to attack')
    parser.add_argument('--classifier', type=str, default='resnet20', help='model to attack')
    parser.add_argument('--seed', type=int, default=18, help='random seed')
    parser.add_argument('--perturb-mode', type=str, default='SSAH', help='attack method')
    parser.add_argument('--max-epoch', type=int, default=1, help='always 1 in attack')
    parser.add_argument('--workers', type=int, default=8, help='num workers to load img')
    parser.add_argument('--wavelet', type=str, default='haar', choices=['haar', 'Daubechies', 'Cohen'])
    parser.add_argument('--test-fid', action='store_true', help='test fid value')

    # SSAH Attack Parameters
    parser.add_argument('--num-iteration', type=int, default=150, help='MAX NUMBER ITERATION')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='LEARNING RATE')
    parser.add_argument('--m', type=float, default=0.2, help='MARGIN')
    parser.add_argument('--alpha', type=float, default=1.0, help='HYPER PARAMETER FOR ADV COST')
    parser.add_argument('--lambda-lf', type=float, default=0.1, help='HYPER PARAMETER FOR LOW FREQUENCY CONSTRAINT')
    parser.add_argument('--outdir', type=str, default='result', help='dir to save the attack examples')
    parser.add_argument('--exp-name', type=str, default='SSAH', help='Experiment Name')

    args = parser.parse_args()

    return args



# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
if opt.dataset == 'cifar10':
    data, num_images = load_cifar10(opt)
elif opt.dataset == 'cifar100':
    data, num_images = load_cifar100(opt)
else:
    data, num_images = load_imagenet_val(opt)


# Load the classifier
if opt.classifier == 'resnet20' and opt.dataset == 'cifar10':
    path = 'checkpoints/cifar10-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 10)
    classifier.load_state_dict(state)
elif opt.classifier == 'resnet20' and opt.dataset == 'cifar100':
    path = 'checkpoints/cifar100-r20.pth.tar'
    checkpoint = torch.load(path)
    state = checkpoint['state_dict']
    classifier = ResNet(20, 100)
    new_state_dict = OrderedDict()  #new_state_dict = OrderedDict(): Creates a new ordered dictionary.
    for k, v in state.items():
        if 'module.' in k:
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    classifier.load_state_dict(new_state_dict)
elif opt.classifier == 'resnet50' and opt.dataset == 'imagenet_val':
    classifier = torchvision.models.resnet50(pretrained=True)
classifier.eval()                  # Switch the classifier to evaluation mode, so its weights will not change
classifier = classifier.to(device) # Load the classifier to the device



opt.num_iteration = 150
opt.learning_rate = 0.01



  # bs=5000 \
  # max-epoch=1 \
  # wavelet='haar' \
  # m=0.2 \
  # alpha=1 \
  # lambda-lf=0.1\
  # seed=8\
  # workers=32\
  # test-fid


# Load the attack
if opt.perturb_mode == 'SSAH':

  atk = SSAH(model=opt.classifier,
            num_iteration=opt.num_iteration,
            learning_rate=opt.learning_rate,
            device=device,
            Targeted=False,    # Have to change the code of SSAH to make it works with targeted is True
            dataset=opt.dataset,
            m=opt.m,
            alpha=opt.alpha,
            lambda_lf=opt.lambda_lf,
            wave=opt.wavelet)
elif opt.perturb_mode == 'DeepFool':
  atk = torchattacks.DeepFool(model, steps=50, overshoot=0.02) 
elif opt.perturb_mode == 'PGD':
  atk = torchattacks.PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
elif opt.perturb_mode == 'C&W':
  atk = torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)
elif opt.perturb_mode == 'JSMA':
  atk = torchattacks.JSMA(model, theta=1.0, gamma=0.1)


# Apply the attack
for batch, (inputs, targets) in enumerate(data):
  adv_images = atk(inputs, targets)
# set the target mode?


# Evaluation
# Intialize some parameters
total_img = 0
att_suc_img = 0
for batch, (inputs, targets) in enumerate(data): # enumerate(data) provides both the batch index (batch) and the batch data (inputs and targets).
    inputs = inputs.to(device)
    targets = targets.to(device)
    common_id = common(targets, predict(classifier, inputs, opt)) # common_id stores the indices of the correctly classified images by comparing the true labels (targets) with the predictions from the classifier.
    total_img += len(common_id)          # total_img is incremented by the number of correctly classified images.
    inputs = inputs[common_id].cuda()   # The inputs and targets are filtered to include only the correctly classified images and moved to the GPU.
    targets = targets[common_id].cuda()

    # attack
    adv = att(inputs)

    att_suc_id = attack_success(targets, predict(classifier, adv, opt)) # return the id where "predicted labels == targeted labels"
    att_suc_img += len(att_suc_id)

    adv = adv[att_suc_id]         # only keep the adversarial images where the attack is successful
    inputs = inputs[att_suc_id]   # only keep the original images where the attack is successful

# Compute the overall SAR for the entire test dataset
sar = 100.0 * att_suc_img / total_img
print(f"Overall Success Attack Rate (SAR): {sar:.2f}%")
#import the neccessary packages
import torch
import argparse
import resnet # where ResNet() is defined
import torchattacks
import ssah_attack

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


# Load the attack
if opt.perturb_mode == 'SSAH':
  atk = SSAH(model=classifier,
            num_iteration=opt.num_iteration,
            learning_rate=opt.learning_rate,
            device=device,
            Targeted=False,
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
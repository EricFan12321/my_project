import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pywt
import numpy as np
import math

from DWT import *
from torch.nn import Module
from torch.autograd import Variable, Function
from torchattacks import Attack

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from utils import *



class SSAH(Attack):
    """"
    Parameters:
    -----------

    """

    def __init__(self,
                 model: nn.Module,
                 num_iteration: int = 150,
                 learning_rate: float = 0.001,
                 device: torch.device = torch.device('cuda'),
                 Targeted: bool = False,
                 dataset: str = 'cifar10',
                 m: float = 0.2,
                 alpha: float = 1,
                 lambda_lf: float = 0.1,
                 wave: str = 'haar',) -> None:
        super().__init__("SSAH", model)
        self.model = model
        self.device = device
        self.lr = learning_rate
        self.target = Targeted
        self.num_iteration = num_iteration
        self.dataset = dataset
        self.m = m
        self.alpha = alpha
        self.lambda_lf = lambda_lf

        self.encoder_fea = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        self.encoder_fea= nn.DataParallel(self.encoder_fea)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.model = nn.DataParallel(self.model)

        self.normalize_fn = normalize_fn(self.dataset)

        self.DWT = DWT_2D_tiny(wavename= wave)
        self.IDWT = IDWT_2D_tiny(wavename= wave)

    def fea_extract(self, inputs: torch.Tensor) -> torch.Tensor:
        fea = self.encoder_fea(inputs)
        b, c, h, w = fea.shape
        fea = self.avg_pool(fea).view(b, c)
        return fea

    def cal_sim(self, adv, inputs):
        adv = F.normalize(adv, dim=1)
        inputs = F.normalize(inputs, dim=1)

        r, c = inputs.shape
        sim_matrix = torch.matmul(adv, inputs.T)
        mask = torch.eye(r, dtype=torch.bool).to(self.device)
        pos_sim = sim_matrix[mask].view(r, -1)
        neg_sim = sim_matrix.view(r, -1)
        return pos_sim, neg_sim

    def select_setp1(self, pos_sim, neg_sim):
        neg_sim, indices = torch.sort(neg_sim, descending=True)
        pos_neg_sim = torch.cat([pos_sim, neg_sim[:, -1].view(pos_sim.shape[0], -1)], dim=1)
        return pos_neg_sim, indices

    def select_step2(self, pos_sim, neg_sim, indices):
        hard_sample = indices[:, -1]
        ones = torch.sparse.torch.eye(neg_sim.shape[1]).to(self.device)
        hard_one_hot = ones.index_select(0, hard_sample).bool()
        hard_sim = neg_sim[hard_one_hot].view(neg_sim.shape[0], -1)
        pos_neg_sim = torch.cat([pos_sim, hard_sim], dim=1)
        return pos_neg_sim

    # Forward method to perform the attack
    def forward(self, inputs: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        # Extract features from the inputs without gradients
        with torch.no_grad():   #  to disable gradient calculation. 
            inputs_fea = self.fea_extract(self.normalize_fn(inputs))

        # Get low frequency components of the inputs
        inputs_ll = self.DWT(inputs)
        inputs_ll = self.IDWT(inputs_ll)

        # Initialize the modifier with arctanh transformation
        eps = 3e-7
        modifier = torch.arctanh(inputs * (2 - eps * 2) - 1 + eps) # what is modifier?
        modifier = Variable(modifier, requires_grad=True)
        modifier = modifier.to(self.device)
        optimizer = optim.Adam([modifier], lr=self.lr)

        lowFre_loss = nn.SmoothL1Loss(reduction='sum')

        # Iterate through the optimization steps
        for step in range(self.num_iteration):
            optimizer.zero_grad()
            self.encoder_fea.zero_grad()

            # Generate adversarial example
            adv = 0.5 * (torch.tanh(modifier) + 1)
            adv_fea = self.fea_extract(self.normalize_fn(adv))

            # Get low frequency components of the adversarial example
            adv_ll = self.DWT(adv)
            adv_ll = self.IDWT(adv_ll)

            # Calculate similarities
            pos_sim, neg_sim = self.cal_sim(adv_fea, inputs_fea)
            
            # If targeted attack, use target features for similarity calculation
            if self.target:
                assert target is not None, "Target must be provided for targeted attack"
                target_fea = self.fea_extract(self.normalize_fn(target))
                target_sim = F.normalize(target_fea, dim=1)
                
                r, c = target_sim.shape
                sim_matrix = torch.matmul(adv_fea, target_sim.T)
                mask = torch.eye(r, dtype=torch.bool).to(self.device)
                pos_sim = sim_matrix[mask].view(r, -1)
                neg_sim = sim_matrix.view(r, -1)

            # Select the most dissimilar one in the first iteration
            if step == 0:
                pos_neg_sim, indices = self.select_setp1(pos_sim, neg_sim)

            # Record the most dissimilar ones by indices and calculate similarity
            else:
                pos_neg_sim = self.select_step2(pos_sim, neg_sim, indices)

            sim_pos = pos_neg_sim[:, 0]
            sim_neg = pos_neg_sim[:, -1]

            # Calculate weights and costs
            w_p = torch.clamp_min(sim_pos.detach() - self.m, min=0)
            w_n = torch.clamp_min(1 + self.m - sim_neg.detach(), min=0)

            adv_cost = torch.sum(torch.clamp(w_p * sim_pos - w_n * sim_neg, min=0))
            lowFre_cost = lowFre_loss(adv_ll, inputs_ll)
            total_cost = self.alpha * adv_cost + self.lambda_lf * lowFre_cost

            optimizer.zero_grad()
            total_cost.backward()
            optimizer.step()

        # Return the final adversarial example
        adv = 0.5 * (torch.tanh(modifier.detach()) + 1)
        return adv
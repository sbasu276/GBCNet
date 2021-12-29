
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import transforms as T
import utils
import torchvision.models as models
from dataloader import GbDataset

class Resnet(nn.Module):
    def __init__(self, num_cls=3, pretrain=True):
        super(Resnet, self).__init__()
        # get the pretrained Resnet50 network
        self.net = models.resnet50(pretrained=pretrain)
        # get the avg pool of the features stem
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # placeholder for the gradients
        self.gradients = None
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, num_cls)
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        # register the hook
        h = x.register_hook(self.activations_hook)
        # apply the remaining pooling
        #x = self.avg_pool(x)
        x = self.net.avgpool(x)
        x = x.view((x.size()[0], -1))
        x = self.net.fc(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        return x

    def eval(self):
        self.net.eval()

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))


class Resnet50(nn.Module):
    def __init__(self, num_cls=3, last_layer=False, pretrain=True):
        super(Resnet50, self).__init__()
        # get the pretrained Resnet50 network
        self.net = models.resnet50(pretrained=pretrain)
        num_ftrs = self.net.fc.in_features
        if last_layer:
            for param in self.net.parameters():
                param.requires_grad = False

        self.net.fc = nn.Linear(num_ftrs, num_cls)
        """
        self.net.fc = nn.Sequential(
                              nn.Linear(num_ftrs, 256), 
                              nn.ReLU(inplace=True), 
                              nn.Dropout(0.4),
                              nn.Linear(256, num_cls)
                            )
        """
    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = x.view((x.size()[0], -1))
        x = self.net.fc(x)
        return x
    
    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))

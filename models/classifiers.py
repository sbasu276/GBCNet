import torch
import torch.nn as nn
import torchvision.models as models
from .res2net import res2net50
from .gbcnet import gbcnet


class GbcNet(nn.Module):
    def __init__(self, num_cls=3, last_layer=False, pretrain=False, att_mode="1", head="2"):
        super(GbcNet, self).__init__()
        attpos = [['0']*2+[att_mode],['0']*3+[att_mode],['0']*5+[att_mode],['0']*2+[att_mode]]
        self.net = gbcnet(pretrained=pretrain, att_position=attpos, GSoP_mode=1)
        num_ftrs = self.net.fc.in_features
        if last_layer:
            for param in self.net.parameters():
                param.requires_grad = False
        
        if head=="1":
            self.net.fc = nn.Linear(num_ftrs, num_cls)
        else:
            self.net.fc = nn.Sequential(
                              nn.Linear(num_ftrs, 256), 
                              nn.ReLU(inplace=True), 
                              nn.Dropout(0.4),
                              nn.Linear(256, num_cls)
                            )
    def forward(self, x):
        x = self.net(x)
        return x

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))


class Vgg16(nn.Module):
    def __init__(self, num_cls=3, last_layer=False, pretrain=True, head="1"):
        super(Vgg16, self).__init__()
        self.net = models.vgg16(pretrained=pretrain)
        num_ftrs = self.net.classifier[6].in_features
        if last_layer:
            for param in self.net.parameters():
                param.requires_grad = False
        if head=="1":
            self.net.classifier[6].out_features = num_cls #= nn.Linear(num_ftrs, num_cls)
        else:   
            self.net.classifier[6] = nn.Sequential(
                              nn.Linear(num_ftrs, 256), 
                              nn.ReLU(inplace=True), 
                              nn.Dropout(0.4),
                              nn.Linear(256, num_cls)
                            )

    def forward(self, x):
        x = self.net(x)
        return x

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))


class Densenet121(nn.Module):
    def __init__(self, num_cls=3, last_layer=False, pretrain=True):
        super(Densenet121, self).__init__()
        self.net = models.densenet121(pretrained=pretrain)
        num_ftrs = self.net.classifier.in_features
        if last_layer:
            for param in self.net.parameters():
                param.requires_grad = False
        
        self.net.classifier = nn.Sequential(
                              nn.Linear(num_ftrs, 256), 
                              nn.ReLU(inplace=True), 
                              nn.Dropout(0.4),
                              nn.Linear(256, num_cls)
                            )

    def forward(self, x):
        x = self.net(x)
        return x

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))


class Res2Net(nn.Module):
    def __init__(self, num_cls=3, last_layer=False, pretrain=True):
        super(Res2Net, self).__init__()
        self.net = res2net50(pretrained=pretrain)
        num_ftrs = self.net.fc.in_features
        if last_layer:
            for param in self.net.parameters():
                param.requires_grad = False
        
        #self.net.fc = nn.Linear(num_ftrs, num_cls)
        self.net.fc = nn.Sequential(
                              nn.Linear(num_ftrs, 256), 
                              nn.ReLU(inplace=True), 
                              nn.Dropout(0.4),
                              nn.Linear(256, num_cls)
                            )
    def forward(self, x):
        x = self.net(x)
        return x

    def load_model(self, weight_file):
        self.net.load_state_dict(torch.load(weight_file))



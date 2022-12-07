from __future__ import print_function, division
import argparse
import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transforms as T
import utils
from torch.optim.lr_scheduler import StepLR
from skimage import io, transform
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from dataloader import GbDataset, GbRawDataset, GbCropDataset
from models import GbcNet 
# import neptune logger
import neptune.new as neptune
# Set plot style
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/imgs")
    parser.add_argument('--set_dir', dest="set_dir", default="data")
    parser.add_argument('--train_set_name', dest="train_set_name", default="train.txt")
    parser.add_argument('--test_set_name', dest="test_set_name", default="test.txt")
    parser.add_argument('--meta_file', dest="meta_file", default="data/roi_pred.json")
    parser.add_argument('--epochs', dest="epochs", default=100, type=int)
    parser.add_argument('--lr', dest="lr", default=5e-3, type=float)
    parser.add_argument('--height', dest="height", default=224, type=int)
    parser.add_argument('--width', dest="width", default=224, type=int)
    parser.add_argument('--no_roi', action='store_true')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_path', dest="load_path", default="gbcnet")
    parser.add_argument('--save_dir', dest="save_dir", default="outputs")
    parser.add_argument('--save_name', dest="save_name", default="gbcnet")
    parser.add_argument('--optimizer', dest="optimizer", default="sgd")
    parser.add_argument('--batch_size', dest="batch_size", default=16, type=int)
    parser.add_argument('--att_mode', dest="att_mode", default="1")
    parser.add_argument('--va', action="store_true")

    args = parser.parse_args()
    return args


def main(args):
    
    transforms = []
    transforms.append(T.Resize((args.width, args.height)))
    #transforms.append(T.RandomHorizontalFlip(0.25))
    transforms.append(T.ToTensor())
    img_transforms = T.Compose(transforms)
    
    val_transforms = T.Compose([T.Resize((args.width, args.height)),\
                                T.ToTensor()])

    with open(args.meta_file, "r") as f:
        df = json.load(f)

    train_labels = []
    t_fname = os.path.join(args.set_dir, args.train_set_name)
    with open(t_fname, "r") as f:
        for line in f.readlines():
            train_labels.append(line.strip())
    val_labels = []
    v_fname = os.path.join(args.set_dir, args.test_set_name)
    with open(v_fname, "r") as f:
        for line in f.readlines():
            val_labels.append(line.strip())
    if args.no_roi:
        train_dataset = GbRawDataset(args.img_dir, df, train_labels, img_transforms=img_transforms)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=5)
        val_dataset = GbRawDataset(args.img_dir, df, val_labels, img_transforms=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=5)
    else:
        val_dataset = GbCropDataset(args.img_dir, df, val_labels, to_blur=False, img_transforms=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=5)

    net = GbcNet(num_cls=3, pretrain=args.pretrain, att_mode=args.att_mode) 

    if args.load_model:
        net.load_state_dict(torch.load(args.load_path))
    net.net = net.net.float().cuda()

    params = [p for p in net.parameters() if p.requires_grad]
   
    total_params = sum(p.numel() for p in net.parameters())
    print("Total Param: ", total_params)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        optimizer = optim.Adam(params, lr=args.lr)
    lr_sched = StepLR(optimizer, step_size=5, gamma=0.8)
    
    os.makedirs(args.save_dir, exist_ok=True)

    train_loss = []

    for epoch in range(args.epochs):
        if not args.no_roi:
            if args.va:
                if epoch <10:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, blur_kernel_size=(65,65), sigma=16, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)#, collate_fn=utils.collate_fn)
                elif epoch >=10 and epoch <15:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, blur_kernel_size=(33,33), sigma=8, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
                elif epoch >=15 and epoch <20:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, blur_kernel_size=(17,17), sigma=4, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
                elif epoch >=20 and epoch <25:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, blur_kernel_size=(9,9), sigma=2, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
                elif epoch >=25 and epoch <30:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, blur_kernel_size=(5,5), sigma=1, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
                else:
                    train_dataset = GbDataset(args.img_dir, df, train_labels, to_blur=False, img_transforms=img_transforms)
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
            else:
                train_dataset = GbDataset(args.img_dir, df, train_labels, to_blur=False, img_transforms=img_transforms)
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)#, collate_fn=utils.collate_fn)
        
        running_loss = 0.0
        total_step = len(train_loader)
        for images, targets, fnames in train_loader:
            images, targets = images.float().cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs.cpu(), targets.cpu())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss.append(running_loss/total_step)
       
        y_true, y_pred = [], []
        with torch.no_grad():
            net.eval()
            for images, targets, fname in val_loader:
                images, targets = images.float().cuda(), targets.cuda()
                if not args.no_roi:
                    images = images.squeeze(0)
                    outputs = net(images)
                    _, pred = torch.max(outputs, dim=1)
                    pred_label = torch.max(pred)
                    pred_idx = pred_label.item()
                    pred_label = pred_label.unsqueeze(0)
                    y_true.append(targets.tolist()[0][0])
                    y_pred.append(pred_label.item())
                else:
                    outputs = net(images)
                    _, pred = torch.max(outputs, dim=1)
                    pred_idx = pred.item()
                    y_true.append(targets.tolist()[0])
                    y_pred.append(pred.item())
            acc = accuracy_score(y_true, y_pred)
            cfm = confusion_matrix(y_true, y_pred)
            spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
            sens = cfm[2][2]/np.sum(cfm[2])
            print('Epoch: [{}/{}] Train-Loss: {:.4f} Val-Acc: {:.4f} Val-Spec: {:.4f} Val-Sens: {:.4f}'\
                    .format(epoch+1, args.epochs, train_loss[-1], acc, spec, sens))

            _name = "%s_epoch_%s.pth"%(args.save_name, epoch)
            save_path = os.path.join(args.save_dir, _name)
            torch.save(net.state_dict(), save_path)

        net.train()
        #lr_sched.step()
        

if __name__ == "__main__":
    args = parse()
    main(args)


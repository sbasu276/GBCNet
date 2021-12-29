from __future__ import print_function, division
import argparse
import os
import cv2
import json
import copy
import pickle
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
# Set plot style
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="data/gb_imgs")
    parser.add_argument('--set_dir', dest="set_dir", default="data/cls_split")
    parser.add_argument('--set_num', dest="set_num", default=0, type=int)
    parser.add_argument('--test_set_name', dest="test_set_name", default="val_set.txt")
    parser.add_argument('--meta_file', dest="meta_file", default="data/res.json")
    parser.add_argument('--height', dest="height", default=224, type=int)
    parser.add_argument('--width', dest="width", default=224, type=int)
    parser.add_argument('--no_roi', action='store_true')
    parser.add_argument('--load_path', dest="load_path", default="gbcnet")
    parser.add_argument('--save_dir', dest="save_dir", default="outputs/temp")
    parser.add_argument('--network', dest="network", default="gbcnet")
    parser.add_argument('--att_mode', dest="att_mode", default="1")
    parser.add_argument('--head', dest="head", default="2")
    parser.add_argument('--sigma', dest="sigma", default=0, type=int) 
    parser.add_argument('--patch', dest="patch", default=0.15, type=float) 
    parser.add_argument('--to_blur', action='store_true') 
    parser.add_argument('--score_name', dest="score_name", default="gbcnet")

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

    val_labels = []
    v_fname = os.path.join(args.set_dir, args.test_set_name)
    with open(v_fname, "r") as f:
        for line in f.readlines():
            val_labels.append(line.strip())
    
    if args.no_roi:
        val_dataset = GbRawDataset(args.img_dir, df, val_labels, img_transforms=val_transforms)
    else:
        val_dataset = GbCropDataset(args.img_dir, df, val_labels, to_blur=args.to_blur, \
                                    sigma=args.sigma, p=args.patch, img_transforms=val_transforms)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    
    net = GbcNet(num_cls=3, pretrain=False, att_mode=args.att_mode, head=args.head) 
    net.load_state_dict(torch.load(args.load_path))

    net.net = net.net.float().cuda()

    y_true, y_pred = [], []
    score_dmp = []
    
    net.eval()
    for images, targets, filenames in val_loader:
        images, targets = images.float().cuda(), targets.cuda()
        cam_img_name = filenames[0]
        if not args.no_roi:
            images = images.squeeze(0)
            outputs = net(images)
            _, pred = torch.max(outputs, dim=1)
            pred_label = torch.max(pred)
            pred_idx = pred_label.item()
            pred_label = pred_label.unsqueeze(0)
            idx = torch.argmax(pred)
            
            y_true.append(targets.tolist()[0][0])
            y_pred.append(pred_label.item())
            score_dmp.append([y_true[-1], outputs[idx.item()].tolist()])
        else:
            outputs = net(images)
            _, pred = torch.max(outputs, dim=1)
            pred_idx = pred.item()
            y_true.append(targets.tolist()[0])
            y_pred.append(pred.item())
            score_dmp.append([y_true[-1], outputs[0].tolist()])
    
    acc = accuracy_score(y_true, y_pred)
    cfm = confusion_matrix(y_true, y_pred)
    spec = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1])/(np.sum(cfm[0]) + np.sum(cfm[1]))
    sens = cfm[2][2]/np.sum(cfm[2]) 
    acc_bin = (cfm[0][0] + cfm[0][1] + cfm[1][0] + cfm[1][1] + cfm[2][2])/np.sum(cfm)
    print("Acc: %.4f 2-Class Acc: %.4f Specificity: %.4f Sensitivity: %.4f"%(acc, acc_bin, spec, sens))
    out_dir = "roc"
    os.makedirs(out_dir, exist_ok=True)
    score_fname = os.path.join(out_dir, args.score_name)
    with open(score_fname, "wb") as f:
        pickle.dump(score_dmp, f)
    print("score_saved!")

if __name__ == "__main__":
    args = parse()
    main(args)


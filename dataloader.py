from __future__ import print_function, division
import cv2
import os
import torch
import json
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LABEL_ENUM = {0: "nrml", 1: "benign", 2: "malg"} 


class BusiDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, to_blur=True, blur_kernel_size=(1,1), sigma=0, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d
        self.sigma = sigma
        self.to_blur = to_blur
        self.blur_kernel_size = blur_kernel_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        if self.transforms:
            img = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        #cv2.imwrite(filename, image)
        print
        return img, label, filename


class GbRawDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.transforms:
            img = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        #cv2.imwrite(filename, image)
        print
        return img, label, filename


def crop_image(image, box, p):
    x1, y1, x2, y2 = box
    cropped_image = image[int((1-p)*y1):int((1+p)*y2), \
                            int((1-p)*x1):int((1+p)*x2)]
    return cropped_image


class GbDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, is_train=True, to_blur=True, blur_kernel_size=(65,65), sigma=0, p=0.15, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        self.to_blur = to_blur
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma
        self.is_train = is_train
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d
        self.p = p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        image = crop_image(image, self.df[idx]["Gold"], self.p)
        if self.transforms:
            image = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        return image, label, filename
        """
        # Get the roi bbox
        num_objs = len(self.df[idx]["Boxes"])
        crps = [orig]
        labels = [label]
        for i in range(num_objs):
            bbs = self.df[idx]["Boxes"][i]
            crp_img = crop_image(image, bbs, 0.1)
            #stack the predicted rois as different samples
            if self.transforms:
                crp_img = self.transforms(crp_img)
            crps.append(crp_img)
            labels.append(label)
        if num_objs == 0:
            #use the original img if no bbox predicted
            #orig = self.transforms(image)
            orig = orig.unsqueeze(0)
            label = label.unsqueeze(0)
        else:
            orig = torch.stack(crps, 0)
            label = torch.stack(labels, 0)
        """


class GbCropDataset(Dataset):
    """ GB classification dataset. """
    def __init__(self, img_dir, df, labels, to_blur=True, blur_kernel_size=(65,65), sigma=16, p=0.15, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        self.to_blur = to_blur
        self.blur_kernel_size = (4*sigma+1, 4*sigma+1)#blur_kernel_size
        self.sigma = sigma
        self.p = p
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["filename"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        filename = self.df[idx]["filename"]
        img_name = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        orig = crop_image(image, self.df[idx]["Gold"], self.p)
        if self.transforms:
            orig = self.transforms(orig)
        # Get the roi bbox
        num_objs = len(self.df[idx]["Boxes"])
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        crps = []
        labels = []
        for i in range(num_objs):
            bbs = self.df[idx]["Boxes"][i]
            crp_img = crop_image(image, bbs, self.p)
            #stack the predicted rois as different samples
            if self.transforms:
                crp_img = self.transforms(crp_img)
            crps.append(crp_img)
            labels.append(label)
        if num_objs == 0:
            #use the original img if no bbox predicted
            #orig = self.transforms(image)
            orig = orig.unsqueeze(0)
            label = label.unsqueeze(0)
        else:
            orig = torch.stack(crps, 0)
            label = torch.stack(labels, 0)
        return orig, label, filename


if __name__ == "__main__":
    VAL_IMG_DIR = "data/gb_imgs"
    VAL_JSON = "data/res.json"
    labels = []
    with open("data/cls_split/val_0.txt", "r") as f:
        for e in f.readlines():
            labels.append(e.strip())
    with open(VAL_JSON, "r") as f:
        df = json.load(f)
    img_transforms = T.Compose([T.Resize((224,224)), T.ToTensor()])
    dataset = GbCropDataset(VAL_IMG_DIR, df, labels, img_transforms = img_transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn)
    images, labels, filename = next(iter(loader))
    print(labels)
    print(images[0].size())
    print(filename)


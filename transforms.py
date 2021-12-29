import cv2
import random
import torch
import numpy as np
from skimage import transform
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            #image = image.flip(-1)
            img = np.flip(image,axis=1).copy()
        else:
            img = image
        return img


class Resize(object):
    def __init__(self, size):
        self.output_size = size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), anti_aliasing=True)
        return img


class Blur(object):
    def __init__(self, sigmas, probs):
        self.sigmas = sigmas
        self.probs = probs

    def __call__(self, image):
        p = random.random()
        psums = []
        sum_ = 0
        for e in self.probs:
            sum_ += e
            psums.append(sum_)
        for i, prob in enumerate(psums):
            if p <= prob:
                sigma = self.sigmas[i]
                break
        ksize = (4*sigma+1, 4*sigma+1)
        img = cv2.GaussianBlur(image, ksize, sigma)
        return img


class ToTensor(object):
    def __call__(self, image):
        #image = image.transpose((2, 0, 1))
        image = F.to_tensor(image)
        return image

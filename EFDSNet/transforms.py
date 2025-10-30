import numpy as np
import random
import cv2

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from typing import List, Union
from PIL import Image

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        aux_data = None
        for t in self.transforms:
            if isinstance(t, GenerateLaplacian):
                image, aux_data, target = t(image, target)
            elif isinstance(t, (ToTensor, Normalize)):
                image, aux_data, target = t(image, aux_data, target)
            else:
                image, target = t(image, target)
        
        # Nếu không có GenerateLaplacian (ví dụ: trong tập validation), aux_data sẽ là None.
        if aux_data is None:
            # Trả về ảnh gốc 2 lần để giữ cấu trúc tuple
            return image, image, target
        
        return image, aux_data, target

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, self.size)
        target = F.resize(target, self.size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class GenerateLaplacian(object):
    def __call__(self, image, target):
        image_np_rgb = np.array(image)
        image_np_gray = np.array(image.convert('L'))

        gaussian_down1 = cv2.pyrDown(image_np_gray)
        gaussian_up1 = cv2.pyrUp(gaussian_down1, dstsize=(image_np_gray.shape[1], image_np_gray.shape[0]))
        gaussian_down2 = cv2.pyrDown(gaussian_down1)
        gaussian_up2 = cv2.pyrUp(gaussian_down2, dstsize=(gaussian_down1.shape[1], gaussian_down1.shape[0]))

        laplacian1 = cv2.subtract(image_np_gray, gaussian_up1)
        laplacian2 = cv2.subtract(gaussian_down1, gaussian_up2)
        laplacian2_resized = cv2.resize(laplacian2, (laplacian1.shape[1], laplacian1.shape[0]))

        # Ghép 6 kênh: R, G, B, Laplacian1, Laplacian2, Gray
        laplacian_image_np = np.stack([
            image_np_rgb[:,:,0], image_np_rgb[:,:,1], image_np_rgb[:,:,2],
            laplacian1, laplacian2_resized, image_np_gray
        ], axis=-1).astype(np.float32)

        return image, laplacian_image_np, target

class ToTensor(object):
    def __call__(self, image, aux_data, target):
        image = F.to_tensor(image)
        if aux_data is not None:
             aux_data = torch.from_numpy(aux_data.transpose((2, 0, 1)))
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, aux_data, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, aux_data, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, aux_data, target
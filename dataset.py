import os
import cv2
import torch
import random
import numpy as np
import albumentations as A
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image 
from torch.utils.data import Dataset


class PokemonArtwork(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

        self.augs = A.Compose([
            A.Resize(220, 220, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0.1, rotate_limit=0.1, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0,0), p=0.25)])

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)
        img = self.augs(image=img)['image']
        img = self.transform(img)

        return img


class PokemonSprites(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

        self.augs = A.Compose([
            A.Resize(42, 56, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0, rotate_limit=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0,0), p=0.25)])

        self.transform = T.ToTensor()

    def shift(self, img): 
        shift = [torch.full((1, 56, 56), (random.uniform(0, 1)-0.5)*2*0.1)] * 3
        zeros = torch.zeros(1, 56, 56)
        full = torch.cat((shift[0], shift[1], shift[2], zeros), 0)

        return img + full

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGBA')
        img = np.array(img)
        img = self.augs(image=img)['image']
        img = self.transform(img)
        img = F.pad(img, (0, 0, 7, 7), "constant", 0)

        if random.random() < 0.25:
            img = self.shift(img)

        return img
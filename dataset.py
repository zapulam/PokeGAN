import os
import cv2
import numpy as np
import albumentations as A
import torchvision.transforms as T

from PIL import Image 
from torch.utils.data import Dataset


class PokemonSprites(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.CenterCrop(42, 42, p=1.0),
            A.ShiftScaleRotate (shift_limit=0.1, scale_limit=0, rotate_limit=0, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0,0), p=0.25)])

        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGBA')
        img = np.array(img)
        img = self.augs(image=img)['image']
        img = self.transform(img)

        return img
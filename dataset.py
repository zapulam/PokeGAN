import os
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class PokemonSprites(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        transform = ToTensor()
        img = transform(img)

        return img
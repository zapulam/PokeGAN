import os
import torchvision.transforms as transforms

from PIL import Image 
from torch.utils.data import Dataset


class PokemonSprites(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        
        transform = transforms.Compose([
            transforms.CenterCrop(42), 
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()])

        img = transform(img)

        return img
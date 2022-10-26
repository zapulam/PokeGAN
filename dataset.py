import os
import torchvision
import torchvision.transforms as T

from PIL import Image 
from torch.utils.data import Dataset


class PokemonSprites(Dataset):
    def __init__(self, path):
        self.imgs = [os.path.join(path, img) for img in os.listdir(path)]

        self.transform = T.Compose([
            T.CenterCrop(42), 
            T.RandomHorizontalFlip(0.0),
            T.Resize(size=(128,128), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST),
            T.ToTensor()])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGBA')
        img = self.transform(img)

        return img
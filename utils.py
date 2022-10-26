import os

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan import GAN
from dataset import PokemonSprites


def create_batch_img(path):
    dataset = PokemonSprites(path)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    save_image(batch, os.path.join('batch.png'))
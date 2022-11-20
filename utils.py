import os
import argparse

from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan import GAN
from dataset import PokemonSprites, PokemonArtwork


def create_batch_img(path, req):
    if req == 'sprites':
        dataset = PokemonSprites(path)
    elif req == 'artwork':
        dataset = PokemonArtwork(path)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    save_image(batch, os.path.join('batch.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--util', type=str, choices=['create_batch_img'], help='util of choice')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.util == 'create_batch_img':
        req = input("\nCreate batch for Sprites or Artwork? (sprites/artwork): ")
        path = input("\nProvide the path to folder: ")
        if req == 'sprites':
            create_batch_img(path, req)
        elif req == 'artwork':
            create_batch_img(path, req)


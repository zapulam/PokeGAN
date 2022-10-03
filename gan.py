import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, channels, gfmaps, latent):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent, gfmaps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gfmaps * 8),
            nn.ReLU(True),
            # state size. (gfmaps*8) x 4 x 4
            nn.ConvTranspose2d(gfmaps * 8, gfmaps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps * 4),
            nn.ReLU(True),
            # state size. (gfmaps*4) x 8 x 8
            nn.ConvTranspose2d( gfmaps * 4, gfmaps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps * 2),
            nn.ReLU(True),
            # state size. (gfmaps*2) x 16 x 16
            nn.ConvTranspose2d( gfmaps * 2, gfmaps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps),
            nn.ReLU(True),
            # state size. (gfmaps) x 32 x 32
            nn.ConvTranspose2d( gfmaps, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, channels, dfmaps):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (channels) x 64 x 64
            nn.Conv2d(channels, dfmaps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dfmaps) x 32 x 32
            nn.Conv2d(dfmaps, dfmaps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfmaps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dfmaps*2) x 16 x 16
            nn.Conv2d(dfmaps * 2, dfmaps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfmaps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dfmaps*4) x 8 x 8
            nn.Conv2d(dfmaps * 4, dfmaps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfmaps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (dfmaps*8) x 4 x 4
            nn.Conv2d(dfmaps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN():
    def __init__(self, device, channels, gfmaps, dfmaps, latent):
        if torch.cuda.is_available() and device != 'cpu':
            self.generator = Generator(channels, gfmaps, latent).to(device)
            self.discriminator = Discriminator(channels, dfmaps).to(device)
        else:
            self.generator = Generator(channels, gfmaps, latent)
            self.discriminator = Discriminator(channels, dfmaps)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
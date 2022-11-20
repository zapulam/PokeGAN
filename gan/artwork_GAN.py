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
    def __init__(self, gfmaps, latent):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(latent, gfmaps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps, gfmaps, 4, 4, 2, bias=False),
            nn.BatchNorm2d(gfmaps),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps, gfmaps*2, 4, 4, 1, bias=False),
            nn.BatchNorm2d(gfmaps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps*2, gfmaps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps*2, gfmaps*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps*2, gfmaps*2, 4, 2, 2, bias=False),
            nn.BatchNorm2d(gfmaps*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps*2, gfmaps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfmaps),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps, gfmaps, 1, 1, 0, bias=False),
            nn.BatchNorm2d(gfmaps),
            nn.ReLU(True),

            nn.ConvTranspose2d(gfmaps, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
   

class Discriminator(nn.Module):
    def __init__(self, dfmaps):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(

            nn.Conv2d(3, dfmaps*2, 4, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dfmaps*2, dfmaps*2, 4, 1, bias=False),
            nn.BatchNorm2d(dfmaps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dfmaps*2, dfmaps*2, 2, 2, bias=False),
            nn.BatchNorm2d(dfmaps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dfmaps*2, dfmaps*4, 2, 1, bias=False),
            nn.BatchNorm2d(dfmaps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dfmaps*4, dfmaps*4, 2, bias=False),
            nn.BatchNorm2d(dfmaps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(dfmaps*4, dfmaps*4, 2, bias=False),
            nn.BatchNorm2d(dfmaps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(dfmaps*10000, 256),
            nn.Linear(256, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GAN3():
    def __init__(self, device, gfmaps, dfmaps, latent):
        if torch.cuda.is_available() and device != 'cpu':
            self.generator = Generator(gfmaps, latent).to(device)
            self.discriminator = Discriminator(dfmaps).to(device)
        else:
            self.generator = Generator(gfmaps, latent)
            self.discriminator = Discriminator(dfmaps)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
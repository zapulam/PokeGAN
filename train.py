import os
import torch
import argparse
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as T

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from gan import GAN
from dataset import PokemonSprites


def train(args):
    # Parameters
    data_path, save_path, epochs, batch_size, lr, betas, device, workers, gfmaps, dfmaps, latent =\
        args.data_path, args.save_path, args.epochs, args.batch_size, args.lr, args.betas, args.device, args.workers, args.gfmaps, args.dfmaps, args.latent

    # Create weights directory
    if not os.path.isdir(save_path):
        os.mkdir(os.path.join(save_path))

    # Create generator test directory
    if not os.path.isdir('.\\generated'):
        os.mkdir(os.path.join('.\\generated'))

    # Create datasets
    dataset = PokemonSprites(data_path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    # Create model (Generator and Discriminator)
    gan = GAN(device, gfmaps, dfmaps, latent)

    # Define loss function
    criterion = nn.BCELoss()

    # Create latent vector batch
    fixed_noise = torch.randn(64, latent, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both Generator and Discriminator
    optimizerD = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=betas)
    optimizerG = optim.Adam(gan.generator.parameters(), lr=lr, betas=betas)

    # Logging
    G_losses, D_losses, iters, best = [], [], 0, 1000

    # Begin training
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")

        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc='Training...', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):

            """ (1) Update Discriminator network with real images """

            # Zero Discriminator gradients
            gan.discriminator.zero_grad()

            # Format batch
            batch = data.to(device)
            label = torch.full((batch.size(0),), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through Discriminator
            output = gan.discriminator(batch).view(-1)

            # Calculate loss on all-real batch
            lossD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            lossD_real.backward()

            # Calculate number correct
            D_y = output.mean().item()


            """ (2) Update Discriminator network with fake images produced by Generator """

            # Create incoherent images
            noise = torch.randn(batch.size(0), latent, 1, 1, device=device)

            # Generate fake image batch with Generator
            fake = gan.generator(noise)
            label.fill_(fake_label)

            # Classify all fake batch with Discriminator
            output = gan.discriminator(fake.detach()).view(-1)

            # Calculate Discriminator's loss on the all-fake batch
            lossD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            lossD_fake.backward()
            D_G_y1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            lossD = lossD_real + lossD_fake

            # Update D
            optimizerD.step()


            """ (3) Update Generator network based on Discriminator's new results  """

            gan.generator.zero_grad()
            label.fill_(real_label)

            # Since we just updated Discriminator, perform another forward pass of all-fake batch through D
            output = gan.discriminator(fake).view(-1)

            # Calculate Generator's loss based on this output
            lossG = criterion(output, label)

            # Calculate gradients for G
            lossG.backward()
            D_G_y2 = output.mean().item()

            # Update G
            optimizerG.step()

            """             # Output training stats
            if i % 2 == 0:
                print('\n[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(dataloader),
                        lossD.item(), lossG.item(), D_y, D_G_y1, D_G_y2)) """

            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            # Save latest model
            torch.save(gan.generator.state_dict(), os.path.join(save_path, "last.pth"))

            # Save best model
            if lossG < best:
                best = lossG
                torch.save(gan.generator.state_dict(), os.path.join(save_path, "best.pth"))

            iters += 1

        with torch.no_grad():
            generated = gan.generator(fixed_noise).detach().cpu()

            transform = T.Resize(size=(42, 42), interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST)
            generated = transform(generated)

            imgs = vutils.make_grid(generated, padding=2, normalize=True)
            save_image(imgs, os.path.join('.\\generated', 'epoch'+str(epoch+1)+'.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='path to images folder')
    parser.add_argument('--epochs', type=int, help='number of training epochs')

    parser.add_argument('--save-path', type=str, help='path to save weights', default='./weights')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--betas', type=tuple, default=(0.005, 0.05), help='beta1 hyperparam for Adam optimizers')
    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:0 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--gfmaps', type=int, default=64, help='number of generator feature maps')
    parser.add_argument('--dfmaps', type=int, default=64, help='number of discriminator feature maps')
    parser.add_argument('--latent', type=str, default=200, help='size of latent vector')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
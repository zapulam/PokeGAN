import torch
import argparse
import torchvision.utils as vutils

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader

from gan import GAN
from dataset import PokemonSprites

# Training Loop
def train(args):
    path, epochs, batch, lr, beta1, device, workers, channels, gfmaps, dfmaps, latent =\
        args.path, args.epochs, args.batch, args.beta1, args.lr,  args.device, args.workers, args.channels, args.gfmaps, args.dfmaps, args.latent

    # Create datasets
    dataset = PokemonSprites(path)
    dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, num_workers=workers)

    # Create model
    gan = GAN(device, channels, gfmaps, dfmaps, latent)

    # Define BCELoss
    criterion = nn.BCELoss()

    # Create latent vector batch
    fixed_noise = torch.randn(64, latent, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(gan.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(gan.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Logging
    img_list, G_losses, D_losses, iters = [], [], [], 0

    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} of {epochs}")

        # For each batch in the dataloader
        for i, data in enumerate(tqdm(dataloader, desc='Training...', ascii=True, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')):


            """ (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) """

            ### Train with all-real batch
            gan.discriminator.zero_grad()

            # Format batch
            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = gan.discriminator(real).view(-1)

            # Calculate loss on all-real batch
            errD_real = criterion(output, label)

            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()


            ### Train with all-fake batch

            # Generate batch of latent vectors
            noise = torch.randn(b_size, latent, 1, 1, device=device)

            # Generate fake image batch with G
            fake = gan.generator(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = gan.discriminator(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)

            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake

            # Update D
            optimizerD.step()


            """ (2) Update G network: maximize log(D(G(z))) """

            gan.generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost

            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = gan.discriminator(fake).view(-1)

            # Calculate G's loss based on this output
            errG = criterion(output, label)

            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()

            # Update G
            optimizerG.step()


            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 50 == 0) or ((epoch == epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = gan.generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to images folder')
    parser.add_argument('--epochs', type=int, help='number of training epochs')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 hyperparam for Adam optimizers')
    parser.add_argument('--device', type=str, default='cuda:0', help='device; cuda:0 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='number of workers')
    parser.add_argument('--channels', type=int, default=3, help='number of color channels')
    parser.add_argument('--gfmaps', type=int, default=64, help='number of generator feature maps')
    parser.add_argument('--dfmaps', type=int, default=64, help='number of discriminator feature maps')
    parser.add_argument('--latent', type=str, default=100, help='size of latent vector')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
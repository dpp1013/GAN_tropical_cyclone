import argparse
import os
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch

from torch.utils.data import Dataset, ConcatDataset
import numpy as np

from torchsummary import summary


def crop_center(img, cropx, cropy):
    # print(img.shape)
    # print(np.typename(img))
    img = img.astype(np.float32)
    img = np.reshape(img, (301, 301))
    y, x = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return np.reshape(img[starty:starty + cropy, startx:startx + cropx], (1, cropx, cropy)).astype("float")


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file, lat_file, lon_file, l):
        self.l = l
        self.file = file
        self.lat_file = lat_file
        self.lon_file = lon_file
        self.data = None
        self.lat_data = None
        self.lon_data = None

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.load(self.file)
        if self.lat_data is None:
            self.lat_data = np.load(self.lat_file)
        if self.lon_data is None:
            self.lon_data = np.load(self.lon_file)
        # print("data_shape", self.data[index:index + 1].shape)
        # print("lat_data_shape", self.lat_data[index:index + 1].shape)
        return np.vstack([
            crop_center(self.data[index:index + 1], opt.img_size, opt.img_size),
            crop_center(self.lat_data[index:index + 1], opt.img_size, opt.img_size),
            crop_center(self.lon_data[index:index + 1], opt.img_size, opt.img_size)
        ]).astype(np.float32), np.vstack([
            crop_center(self.data[index + 1:index + 2], opt.img_size, opt.img_size),
            crop_center(self.lat_data[index + 1:index + 2], opt.img_size, opt.img_size),
            crop_center(self.lon_data[index + 1:index + 2], opt.img_size, opt.img_size),
        ]).astype(np.float32)

    def __len__(self):
        return self.l - 1


lat_lon_data_dir = "/Volumes/董萍萍 18655631746/my_data_set/new_data_lat_lon"


def gen_data_set(data_dir):
    result = []
    for file_name in os.listdir(data_dir):
        seq = file_name.split('_')
        # 3_1979167N13071__lat.npy
        lat_chanel = os.path.join(lat_lon_data_dir, "%s_%s__lat.npy" % (seq[0], seq[1]))
        lon_chanel = os.path.join(lat_lon_data_dir, "%s_%s__lon.npy" % (seq[0], seq[1]))

        result.append(DealDataset(os.path.join(data_dir, file_name), lat_chanel, lon_chanel, int(seq[0])))
    return ConcatDataset(result)


os.makedirs("img", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(32 * 32 * opt.channels, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        # out.shape torch.Size([64, 8192])
        # print("out.shape", out.shape)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
# summary(generator, (1, 100))

discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)

dataloader = torch.utils.data.DataLoader(gen_data_set('./data/new_data_norm'), batch_size=opt.batch_size,
                                         shuffle=True, )
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, next_imgs) in enumerate(dataloader):
        # print(imgs.shape)
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        # Configure input
        # print(next_imgs.dtype)
        # print(next_imgs.shape)
        t = Tensor(next_imgs)
        real_imgs = Variable(t)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # print('imgs.shape', imgs.shape)
        z = Variable(Tensor(np.reshape(imgs, (imgs.shape[0], opt.channels * 32 * 32))))
        # print('z shape:', z.shape)

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     for img in gen_imgs.data:
        #         save_image(gen_imgs.data, "img/%d_gen.png" % batches_done, nrow=1, normalize=True)
        #         save_image(real_imgs, "img/%d_real.png" % batches_done, nrow=1, normalize=True)
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data, "img/%d_gen.png" % batches_done, nrow=8, normalize=True)
            save_image(real_imgs, "img/%d_real.png" % batches_done, nrow=8, normalize=True)

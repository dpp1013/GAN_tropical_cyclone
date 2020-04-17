import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from src.ConvLSTM import ConvLSTM
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.DoubleTensor

channels = 3
image_size = 32
in_step = 3  # 监督化，输入步长
out_step = 1  # 监督化，输出输入步长后的第几个

epochs = 10
batch_size = 64
lr = 0.0001

b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
sample_interval = 1000

origin_data_dir = r"D:\my_data_set\tran_data"
save_data_dir = r"D:\my_data_set\imgs"


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvLSTM(
                input_size=(image_size, image_size),
                input_dim=channels,
                out_dim=channels,
                hidden_dim=[32, 16, 3],
                kernel_size=(3, 3),
                num_layers=3,
                batch_first=True,
                bias=True,
                return_all_layers=False
            ),
        )

    def forward(self, z):
        img = self.model(z)
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
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = image_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, file):
        self.data = None
        self.file = file

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.load(self.file)
        mask = np.zeros(self.data.shape)
        mask[:, :, 0, :, :] = 1
        mask[:, :, 1, 15:16, 16:17] = 1
        mask[:, :, 2, 15:16, 16:17] = 1
        self.data = self.data * mask
        return self.data[index][:-1].reshape((in_step, channels, image_size, image_size)), self.data[index][-1].reshape(
            (1, channels, image_size, image_size))

    def __len__(self):
        return int(os.path.basename(self.file).split("_")[0])


def gen_data_set():
    result = []
    for file_name in os.listdir(origin_data_dir):
        result.append(DealDataset(os.path.join(origin_data_dir, file_name)))
    return ConcatDataset(result)


if __name__ == "__main__":
    # Data Loader
    data_loader = DataLoader(gen_data_set(), batch_size=batch_size, shuffle=True, )
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # models
    generator = Generator()
    discriminator = Discriminator()
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    for epoch in range(epochs):
        for i, (x, y) in enumerate(data_loader):
            x = x.float()
            # x.view(1, in_step, channels, width, height)
            y = y.float()
            # Adversarial ground truths
            valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)
            optimizer_G.zero_grad()
            out = generator(x)
            out = out.view((x.shape[0], channels, image_size, image_size))
            d_out = discriminator(out)
            g_loss = adversarial_loss(d_out.double(), valid)
            g_loss.backward()
            optimizer_G.step()
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(
                discriminator(y.float().view((y.shape[0], channels, image_size, image_size))), valid.float())
            fake_loss = adversarial_loss(
                discriminator(out.detach()),
                fake.float())
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, epochs, i, len(data_loader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(data_loader) + i
            if batches_done % sample_interval == 0:
                save_image(out.data[:, 0:1, ::], os.path.join(save_data_dir, "%d_gen.png" % batches_done), nrow=8,
                           normalize=True)
                save_image(y.view((y.shape[0], channels, image_size, image_size))[:, 0:1, ::],
                           os.path.join(save_data_dir, "%d_real.png" % batches_done),
                           nrow=8, normalize=True)
                np.save(os.path.join(save_data_dir, "%d_gen.npy" % batches_done), out.data)
                np.save(os.path.join(save_data_dir, "%d_real.npy" % batches_done), y.data)

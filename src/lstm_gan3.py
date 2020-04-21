import torch.nn as nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from ConvLSTM import ConvLSTM
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import math

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.DoubleTensor

channels = 3
image_size = 32
in_step = 3  # 监督化，输入步长
out_step = 1  # 监督化，输出输入步长后的第几个

epochs = 5
batch_size = 64
lr = 0.0001

b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.999  # adam: decay of first order momentum of gradient
sample_interval = 100

origin_data_dir = r"D:\my_data_set\tran_data"
save_data_dir = r"D:\my_data_set\imgs\new gen image"


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            ConvLSTM(
                input_size=(image_size, image_size),
                input_dim=channels,
                out_dim=channels,
                hidden_dim=[32, 32, 3],
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


def gen_tran_data_set():
    result = []
    for file_name in os.listdir(origin_data_dir):
        tf_id = int(file_name.split('_')[2][:4])
        if tf_id < 1980:
            result.append(DealDataset(os.path.join(origin_data_dir, file_name)))
            # break
    return ConcatDataset(result)


def gen_test_data_set():
    result = []
    for file_name in os.listdir(origin_data_dir):
        tf_id = int(file_name.split('_')[2][:4])
        if tf_id >= 1992:
            result.append(DealDataset(os.path.join(origin_data_dir, file_name)))
    return ConcatDataset(result)


def evaluate_trajectory(p_y, y):
    batch = p_y.shape[0]
    p_y_lat_lon = p_y[:, 1:3, 15:16, 15:16]
    y_lat_lon = y.view((y.shape[0], channels, image_size, image_size))[:, 1:3, 15:16, 15:16]
    # 计算 lat_mae lon_mae
    lat_lon = abs(p_y_lat_lon - y_lat_lon)
    lat = lat_lon[:, 0:1, :, :]
    lon = lat_lon[:, 1:2, :, :]
    lat_mae = sum(lat) / batch
    lon_mae = sum(lon) / batch

    # 计算 lat_mse lon_mse
    lat_lon = lat_lon * lat_lon
    lat = lat_lon[:, 0:1, :, :]
    lon = lat_lon[:, 1:2, :, :]
    lat_mse = sum(lat) / batch
    lon_mse = sum(lon) / batch

    # 计算 lat_rmse lon_rmse
    lat_rmse = math.sqrt(lat_mse)
    lon_rmse = math.sqrt(lon_mse)
    print('lat_mae:', lat_mae.item(), 'lat_mse:', lat_mse.item(), 'lat_rmse:', lat_rmse, 'lon_mae:',
          lon_mae.item(), 'lon_mse:', lon_mse.item(),
          'lon_rmse:', lon_rmse)
    return lat_mae, lat_mse, lat_rmse, lon_mae, lon_mse, lon_rmse


if __name__ == "__main__":

    batches_done_list = []

    d_loss_list = []
    g_loss_list = []

    ame_list = []
    mse_list = []
    rmse_list = []

    lat_mae_list = []
    lat_mse_list = []
    lat_rmse_list = []
    lon_mae_list = []
    lon_mse_list = []
    lon_rmse_list = []

    loss_plt = plt.subplot(2, 2, 1)
    loss_plt.set_title('loss')  # 添加子标题

    lat_evaluate_trajectory_plt = plt.subplot(2, 2, 2)
    lat_evaluate_trajectory_plt.set_title('lat trajectory evaluate')  # 添加子标题

    lon_evaluate_trajectory_plt = plt.subplot(2, 2, 3)
    lon_evaluate_trajectory_plt.set_title('lon trajectory evaluate')  # 添加子标题
    plt.ion()
    plt.figure(1, [12.8, 9])
    # Data Loader
    data_loader = DataLoader(gen_tran_data_set(), batch_size=batch_size, shuffle=True, )
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # models
    generator = Generator()
    discriminator = Discriminator()
    # generator = torch.load('./generator.pk')
    # discriminator = torch.load('./discriminator.pk')
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
    for epoch in range(epochs):
        batches_done_list.clear()
        d_loss_list.clear()
        g_loss_list.clear()
        ame_list.clear()
        mse_list.clear()
        rmse_list.clear()
        lat_mae_list.clear()
        lat_mse_list.clear()
        lat_rmse_list.clear()
        lon_mae_list.clear()
        lon_mse_list.clear()
        lon_rmse_list.clear()
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
            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())
            batches_done_list.append(batches_done)

            loss_plt.plot(batches_done_list, d_loss_list, c='r', ls='-', marker='+', mec='b', mfc='w')
            loss_plt.plot(batches_done_list, g_loss_list, c='g', ls='-', marker='+', mec='b', mfc='w')
            loss_plt.legend(['Discriminator', 'Generator'])

            lat_mae, lat_mse, lat_rmse, lon_mae, lon_mse, lon_rmse = evaluate_trajectory(out, y)
            lat_mae_list.append(lat_mae)
            lat_mse_list.append(lat_mse)
            lat_rmse_list.append(lat_rmse)
            lon_mae_list.append(lon_mae)
            lon_mse_list.append(lon_mse)
            lon_rmse_list.append(lon_rmse)

            lat_evaluate_trajectory_plt.plot(batches_done_list, lat_mae_list, c='r', ls='-', marker='+', mec='r',
                                             mfc='w')
            lat_evaluate_trajectory_plt.plot(batches_done_list, lat_mse_list, c='g', ls='-', marker='+', mec='g',
                                             mfc='w')
            lat_evaluate_trajectory_plt.plot(batches_done_list, lat_rmse_list, c='b', ls='-', marker='+', mec='b',
                                             mfc='w')
            lat_evaluate_trajectory_plt.legend(['mae', 'mse', 'rmse'])
            lon_evaluate_trajectory_plt.plot(batches_done_list, lon_mae_list, c='r', ls='-', marker='+', mec='r',
                                             mfc='w')
            lon_evaluate_trajectory_plt.plot(batches_done_list, lon_mse_list, c='g', ls='-', marker='+', mec='g',
                                             mfc='w')
            lon_evaluate_trajectory_plt.plot(batches_done_list, lon_rmse_list, c='b', ls='-', marker='+', mec='b',
                                             mfc='w')
            lon_evaluate_trajectory_plt.legend(['mae', 'mse', 'rmse'])

            # out.shape(64,3,32,32)
            # y.shape(64,3,32,32)
            #
            # out_lat_lon = out[:, 1:3, 15:16, 15:16]
            # y_lat_lon = y[:, 1:3, 15:16, 15:16]

            plt.pause(0.1)

            if batches_done % sample_interval == 0:
                save_image(out.data[:, 0:1, ::], os.path.join(save_data_dir, "%d_gen.png" % batches_done), nrow=8,
                           normalize=True)
                save_image(y.view((y.shape[0], channels, image_size, image_size))[:, 0:1, ::],
                           os.path.join(save_data_dir, "%d_real.png" % batches_done),
                           nrow=8, normalize=True)
                np.save(os.path.join(save_data_dir, "%d_gen.npy" % batches_done), out.data)
                np.save(os.path.join(save_data_dir, "%d_real.npy" % batches_done), y.data)
        torch.save(generator, 'generator_%d.pk' % epoch)
        torch.save(discriminator, 'discriminator_%d.pk' % epoch)
        plt.savefig('process_%d.png' % epoch, dpi=300)
    plt.show()

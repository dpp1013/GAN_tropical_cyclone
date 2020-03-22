'''
time:20200321
url:https://blog.csdn.net/jizhidexiaoming/article/details/96485095
功能：MNIST生成图片
'''
import torch.nn as nn
import os
import torch.autograd
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image
from torch.autograd import Variable

# 文件不存在就创建文件
if not os.path.exists('./img'):
    os.mkdir('./img')


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)  # clamp将随机数限制在[min,max]这个区间
    out = out.view(-1, 1, 28, 28)
    return out


# 定义判别器 Discrimination 使用多层网络作为判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(

            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()

        )

    def forward(self,x):
        x = self.dis(x)
        return x


# 定义生成器
# 输出一个100维的0-1之间的高斯分布，然后通过第一层线性变换将其变换到256维，然后通过LeakyReLU
# 激活函数，接着进行一个线性变换，在经过一个LeakyReLU激活函数，然后经过线性变换将其变成784维，最后
# 经过Tanh激活函数是希望生成的假的图片数据分布能过够在-1-1之间
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()  # 使得数据在[-1,1]之间，因为输入的真实数据经过transform之后就是这个分布
        )

    def forward(self,x):
        x = self.gen(x)
        return x


if __name__ == '__main__':
    batch_size = 128
    num_epoch = 100
    z_dimension = 784
    # 图像预处理
    img_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )
    mnist = datasets.MNIST(root='./data', train=True, transform=img_transform, download=True)
    # data loader 数据载入
    dataloader = torch.utils.data.DataLoader(
        dataset=mnist, batch_size=batch_size, shuffle=True
    )
    D = discriminator()
    G = generator()
    if torch.cuda.is_available():
        D = D.cuda()
        G = G.cuda()
    criterion = nn.BCELoss()  # 单目标二分类交叉熵函数
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

    # 判别器的判断过程
    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(dataloader):
            # 训练判别器
            num_img = img.size(0)
            img = img.view(num_img, -1)  # 将图片展开为784
            real_img = Variable(img)
            real_label = Variable(torch.ones(num_img))
            fake_label = Variable(torch.zeros(num_img))
            # 判别器训练：分为两部分1. 真的图像判别为真 2. 假的部分判定为假
            # 计算
            real_out = D(real_img)
            d_loss_real = criterion(real_out, real_label)
            real_scores = real_out
            z = Variable(torch.randn(num_img, z_dimension))  # 随机产生一点噪声
            fake_img = G(z).detach()  # 随机噪声放入生成网络
            fake_out = D(fake_img)
            d_loss_fake = criterion(fake_out, fake_label)
            fake_scores = fake_out
            d_loss = d_loss_fake + d_loss_real
            d_optimizer.zero_grad()  # 在传播之前，先将梯度归0
            d_loss.backward()  # 将误差反向传播
            d_optimizer.step()  # 更新参数
            # ==================训练生成器============================
            # ###############################生成网络的训练###############################
            # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
            # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
            # 反向传播更新的参数是生成网络里面的参数，
            # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
            # 这样就达到了对抗的目的
            z = Variable(torch.randn(num_img, z_dimension))
            fake_img = G(z)  # 随机噪声输入到生成器中，得到一个假图片
            output = D(fake_img)
            g_loss = criterion(output, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
                      'D real: {:.6f},D fake: {:.6f}'.format(
                    epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
                    real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
                ))
            if epoch == 0:
                real_images = to_img(real_img.cpu().data)
                save_image(real_images, './img/real_images.png')
        fake_images = to_img(fake_img.cpu().data)
        save_image(fake_images, './img/fake_images-{}.png'.format(epoch + 1))

    # 保存模型
    torch.save(G.state_dict(), './generator.pth')
    torch.save(D.state_dict(), './discriminator.pth')
"""
1. 从一个压缩包中逐个读取压缩文件,查看图片
"""

import netCDF4 as nc
import os
import tarfile
import numpy as np
from torch import Tensor
from torchvision.utils import save_image
from PIL import Image, ImageDraw

data_dir = "/Volumes/董萍萍 18655631746/my_data_set/data"
size = 64
channel = 'IRWIN'


def gen(file_name):
    full_path = os.path.join(data_dir, file_name)
    tar = tarfile.open(full_path)
    all = None
    for nc_file in tar.getmembers():
        file = tar.extractfile(nc_file)
        dataset = nc.Dataset('', memory=file.read())
        if channel not in dataset.variables:
            continue
        data = dataset.variables[channel][:]
        img = Image.fromarray(data.reshape((data.shape[1], data.shape[2])), mode='F')
        draw = ImageDraw.Draw(img)
        draw.rectangle(((301 - size) // 2, (301 - size) // 2, (301 - size) // 2 + size, (301 - size) // 2 + size))
        data = np.asarray(img).reshape(data.shape)
        if all is None:
            all = np.array(data)
        else:
            all = np.concatenate([all, np.array(data)])
    if all is not None:
        save_image(Tensor(all).view((all.shape[0], 1, all.shape[1], all.shape[2])), 'img/%s.png' % file_name, nrow=8,
                   normalize=True)


if __name__ == "__main__":
    # file_name = "HURSAT_b1_v06_1978168N11242_BUD_c20170721.tar.gz"
    file_name_list = os.listdir(data_dir)
    i = 0
    for file_name in file_name_list:
        if file_name.endswith('.tar.gz'):
            gen(file_name)
        print('进度:[%d/%d]' % (i, len(file_name_list)))
        i += 1

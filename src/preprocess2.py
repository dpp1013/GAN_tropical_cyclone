"""
提取nc文件中的
"""'''
对原始数据进行归一化操作，并变成1-1的训练数据形式
'''
import os
import numpy as np
import tarfile
import stat
import netCDF4 as nc

origin_data_path = "D:\my_data_set\data"
save_data_path = r"D:\my_data_set\single_data"


def Image_normalizeration(Image):
    x_norm = None
    for i in range(len(Image)):
        i = Image[i, :, :]
        _range = np.max(i) - np.min(i)
        image_norm = (i - np.min(i)) / _range
        image_norm = image_norm[None, :, :]
        if x_norm is None:
            x_norm = image_norm
        else:
            x_norm = np.concatenate((x_norm, image_norm))
    return x_norm


def targz_file(path):
    '''
    :param path: 要解压的文件
    :return: 解压后的文件夹
    '''
    target_path = os.path.join(os.path.dirname(path), ''.join(path.split('_')[5:6]))
    if os.path.exists(target_path) is False:
        os.mkdir(target_path)
    tar = tarfile.open(path)
    for file_name in tar.getnames():
        tar.extract(file_name, target_path)
        os.chmod(os.path.join(target_path, file_name), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    tar.close()
    return target_path


# 读取nc文件并保存
def read_nc_single(path):
    '''
    :param path: 压缩后要读取的文件地址
    :return:
    '''
    data_lat = None
    data_lon = None
    irwin = None
    result = None
    Lat_lon = np.ones((301, 301))
    length = len(os.listdir(path))
    newFileName = os.path.join(save_data_path, str(length) + '_' + os.path.basename(path) + '.npy')
    for fileName in os.listdir(path):
        fileName = os.path.join(path, fileName)
        f = nc.Dataset(fileName)
        Lat = f.variables['lat'][:]  # 横向
        Lon = f.variables['lon'][:]
        win = f.variables['IRWIN'][:]
        Lat = Lat * Lat_lon
        Lon = np.transpose(Lon * Lat_lon)
        Lat = Lat.reshape((-1, 301, 301))
        Lon = Lon.reshape((-1, 301, 301))
        single = np.vstack((win, Lat, Lon))
        single = single.reshape((-1, 3, 301, 301))
        if result is None:
            result = single
        else:
            result = np.concatenate((result, single))
        f.close()
    print("result", result.shape)
    print("save as:", newFileName)
    np.save(newFileName, np.array(result))
    # np.save("%s_%s" % (newFileName, '_lat.npy'), np.array(data_lat))
    # np.save("%s_%s" % (newFileName, '_lon.npy'), np.array(data_lon))


# 压缩处理完文件后，删除解压后的文件
def remove(path):
    file_list = os.listdir(path)
    for f in file_list:
        os.remove(os.path.join(path, f))
    os.rmdir(path)


def readFile(path):
    '''
    :param path: 初始文件夹path
    :return:
    '''
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        fileName = dir_list[i]
        filePath = os.path.join(path, fileName)
        target_path = targz_file(filePath)  # 解压文件
        read_nc_single(target_path)  # 读取压缩后的文件
        remove(target_path)  # 删除处理后的文件夹
        print("进度:", i, '/', len(dir_list))


if __name__ == '__main__':
    readFile(origin_data_path)

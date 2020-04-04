'''
对原始数据进行归一化操作，并变成1-1的训练数据形式
'''
import os
import numpy as np
import tarfile
import stat
import netCDF4 as nc


def Image_normalizeration(Image):
    print(Image.shape)
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
    print(x_norm.shape)
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
    Lat_lon = np.ones((301, 301))
    for fileName in os.listdir(path):
        length = len(os.listdir(path))
        newFileName = str(length) + '_' + fileName.split('.')[0]
        newFileName = os.path.join(r'D:\my_data_set\new_data_lat_lon', newFileName)
        fileName = os.path.join(path, fileName)
        f = nc.Dataset(fileName)
        Lat = f.variables['lat'][:]  # 横向
        Lon = f.variables['lon'][:]
        Lat = Lat * Lat_lon
        Lon = np.transpose(Lon * Lat_lon)
        if data_lat is None:
            data_lat = Lat
        else:
            data_lat = np.concatenate((data_lat, Lat))
        if data_lon is None:
            data_lon = Lon
        else:
            data_lon = np.concatenate((data_lon, Lon))
        f.close()
    data_lon = data_lon.reshape((-1, 301, 301))
    data_lat = data_lat.reshape((-1, 301, 301))
    print(data_lat.shape, data_lon.shape)
    data_lat = Image_normalizeration(data_lat)
    data_lon = Image_normalizeration(data_lon)
    np.save("%s_%s" % (newFileName, '_lat.npy'), np.array(data_lat))
    np.save("%s_%s" % (newFileName, '_lon.npy'), np.array(data_lon))


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
    i = 1
    for fileName in os.listdir(path):
        i += 1
        if i > 100:
            break
        filePath = os.path.join(path, fileName)
        target_path = targz_file(filePath)  # 压缩文件
        read_nc_single(target_path)  # 读取压缩后的文件
        remove(target_path)  # 删除处理后的文件夹


if __name__ == '__main__':
    path = r'D:\my_data_set\data'
    readFile(path)

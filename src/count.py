"""
统计lat和lon的范围
"""

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
tmp_data_path = r"D:\my_data_set\tmp"


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
    target_path = os.path.join(tmp_data_path, ''.join(path.split('_')[5:6]))
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
    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None
    for fileName in os.listdir(path):
        fileName = os.path.join(path, fileName)
        f = nc.Dataset(fileName)
        lat = f.variables['lat']
        lon = f.variables['lon']
        _min_lat, _max_lat = lat.actual_range
        _min_lon, _max_lon = lon.actual_range
        min_lat = _min_lat if min_lat is None or _min_lat < min_lat else min_lat
        min_lon = _min_lon if min_lon is None or _min_lon < min_lon else min_lon
        max_lat = _max_lat if max_lat is None or _max_lat > max_lat else max_lat
        max_lon = _max_lon if max_lon is None or _max_lon > max_lon else max_lon
        f.close()
    return min_lat, max_lat, min_lon, max_lon


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
    min_lat = None
    max_lat = None
    min_lon = None
    max_lon = None
    dir_list = os.listdir(path)
    for i in range(len(dir_list)):
        fileName = dir_list[i]
        if fileName.endswith("tar.gz"):
            filePath = os.path.join(path, fileName)
            target_path = targz_file(filePath)  # 解压文件
            _min_lat, _max_lat, _min_lon, _max_lon = read_nc_single(target_path)  # 读取压缩后的文件
            remove(target_path)  # 删除处理后的文件夹
            min_lat = _min_lat if min_lat is None or _min_lat < min_lat else min_lat
            min_lon = _min_lon if min_lon is None or _min_lon < min_lon else min_lon
            max_lat = _max_lat if max_lat is None or _max_lat > max_lat else max_lat
            max_lon = _max_lon if max_lon is None or _max_lon > max_lon else max_lon
            print("进度:", i, '/', len(dir_list), "当前结果", min_lat, max_lat, min_lon, max_lon)
    return min_lat, max_lat, min_lon, max_lon


if __name__ == '__main__':
    min_lat, max_lat, min_lon, max_lon = readFile(origin_data_path)
    print("min_lat, max_lat, min_lon, max_lon", min_lat, max_lat, min_lon, max_lon)

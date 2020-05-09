# 1.解压文件 2. 取得解压后的文件名字获取开始时间：str类型1980.01.01.00，结束时间str。
# 3. 读取文件的lat_range和Lon_range
import os
import tarfile
import shutil
from netCDF4 import Dataset
import stat
import pandas as pd
import numpy as np

main_path = 'D:\my_data_set\data'
# 以列表的形式存储，第一个值是开始时间
map_TC = []
label = []


# 解压文件
def un_tar(file):
    '''
    :param file: 需要解压的文件
    :param dir: 解压后存放的路径
    :return:
    '''
    un_file = file.replace('.tar.gz', '')
    if os.path.exists(un_file):
        return un_file
    tar = tarfile.open(file)
    names = tar.getnames()
    # print(names)
    if os.path.isdir(un_file):
        pass
    else:
        os.mkdir(un_file)
    for name in names:
        tar.extract(name, un_file)
        os.chmod(os.path.join(un_file, name), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    tar.close()
    return un_file


# 获取一些信息
def getMessage(un_file):
    '''
    :param un_file: 解压后的文件夹
    '''
    y_lat = []
    y_lon = []
    global map_TC
    min_lat = None
    min_lon = None
    max_lat = None
    max_lon = None
    files = os.listdir(un_file)
    year = ''.join(files[0].split('.')[2:3])
    start_time = ''.join(files[0].split('.')[2:6])
    end_time = ''.join(files[-1].split('.')[2:6])
    # 找出最大范围的range
    for file in files:
        f = Dataset(os.path.join(un_file, file))
        lat = f.variables['lat']
        lon = f.variables['lon']
        CentLat = float(f.variables['CentLat'][:])
        CentLon = float(f.variables['CentLon'][:])
        # print(list(CentLon))
        _min_lat, _max_lat = lat.actual_range
        _min_lon, _max_lon = lon.actual_range
        min_lat = _min_lat if min_lat is None or _min_lat < min_lat else min_lat
        min_lon = _min_lon if min_lon is None or _min_lon < min_lon else min_lon
        max_lat = _max_lat if max_lat is None or _max_lat > max_lat else max_lat
        max_lon = _max_lon if max_lon is None or _max_lon > max_lon else max_lon
        y_lat.append(CentLat)
        y_lon.append(CentLon)
        f.close()
    min_y_lat = min(y_lat)
    min_y_lon = min(y_lon)
    max_y_lat = max(y_lat)
    max_y_lon = max(y_lon)
    label.append([start_time, y_lat, y_lon])
    map_TC.append(
        [year, start_time, end_time, min_lat, max_lat, min_lon, max_lon, min_y_lat, max_y_lat, min_y_lon, max_y_lon])

    # 删除文件夹


def remove(file):
    '''
    :param file: 要删除的文件夹
    '''
    file_list = os.listdir(file)
    # print(file_list)
    for f in file_list:
        os.remove(os.path.join(file, f))
    os.rmdir(file)


if __name__ == '__main__':
    col_name = ['year', 'start_time', 'end_start', 'min_lat', 'max_lat', 'min_lon', 'max_lon', 'min_y_lat', 'max_y_lat',
                'min_y_lon', 'max_y_lon']
    i = 0

    for file in os.listdir(main_path):
        # i += 1
        # if i == 10:
        #     break
        try:
            zip_file = os.path.join(main_path, file)
            print(zip_file)
            un_file = un_tar(zip_file)  # 解压文件，并获得文件夹的地址
            getMessage(un_file)
            remove(un_file)
            map_tc = pd.DataFrame(columns=col_name, data=map_TC)
        except Exception as err:
            print(err)
            continue
    # pass
    # print('--------------map_tc---------------')
    # print(map_TC)
    # print('---------------label---------------')
    # print(label)
    np.save('D:\my_data_set\label.npy', label)
    map_tc.to_csv('D:\my_data_set\map_TC_y.csv', index=0, encoding='gbk')

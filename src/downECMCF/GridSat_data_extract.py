from netCDF4 import Dataset
import pandas as pd
import os
import numpy as np

np.set_printoptions(threshold=20)
'''
2D
irwin_cdr:long_name = "NOAA FCDR of Brightness Temperature near 11 microns (Nadir-most observations)"; 有值
irwin_vza_adj  :long_name = "Adjustment made to all IRWIN channels for view zenith angle effects";
irwvp   :long_name = "Brightness Temperature near 6.7 microns (Nadir-most observations)"; None
satid_ir   :long_name = "IRWIN_CDR Satellite Index (packed)";负值且一样的值比较多
satid_vs    :long_name = "VSCHN Satellite Index (packed)";负值且一样的值比较多
satid_wv    :long_name = "IRWVP Satellite Index (packed)";
vschn   :long_name = "Visible reflectance near 0.6 microns (Nadir-most observations)";没啥值
'''


def segImage(maptc, lat, lon, irwin_cdr):
    pass


def readNC(file):
    '''
    :param file: 文件名字
    :return:
    '''
    data = []
    nc_obj = Dataset(file)
    data = nc_obj.variables.keys()
    lat = nc_obj.variables['lat'][:]
    lon = nc_obj.variables['lon'][:]
    irwin_cdr = nc_obj.variables['irwin_cdr'][:]


if __name__ == '__main__':
    main_file = 'D:\GridSat'
    maptc_csv = 'D:\my_data_set\map_TC.csv'
    filesList = os.listdir(main_file)
    filesListDate = [int(''.join(file.split('.')[1:5])) for file in filesList]
    maptc = pd.read_csv(maptc_csv)
    count = 1
    for index, row in maptc.iterrows():
        year = row['year']
        start_time = row['start_time']
        end_time = row['end_start']
        min_lat = row['min_lat']
        max_lat = row['max_lat']
        min_lon = row['min_lon']
        max_lon = row['max_lon']
        print('start_time:%d, end_time:%d, min_lat:%d, max_lat:%d, min_lon:%d, max_lon:%d' % (
            start_time, end_time, min_lat, max_lat, min_lon, max_lon))
        start_index = filesListDate.index(start_time) if start_time in filesListDate else None
        end_index = filesListDate.index(end_time) if end_time in filesListDate else None
        count += 1
        if start_index == None or end_index == None:
            break
        print('start_index:%s,end_index:%s' % (start_index, end_index))
        print(filesListDate[start_index: end_index + 1])
        for fileNum in filesListDate[start_index: end_index + 1]:
            fileNum = list(str(fileNum))
            fileNum.insert(4, '.')
            fileNum.insert(7, '.')
            fileNum.insert(10, '.')
            fileNum = ''.join(fileNum)
            file_path = os.path.join(main_file, 'GRIDSAT-B1.' + fileNum + '.v02r01.nc')
            # 读取nc文件，并提取值
            # readNC(file_path)
        # print(file_path)
    print(count)

    # print(count)
    # break

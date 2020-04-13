import numpy as np
import os


def getFile(path):
    data = np.load(path)
    data = data.reshape((8, 3, 32, 32))
    data = data[:, 1:3, 15:16, 15:16]
    return data


def Mae(real, gen):
    '''
    :return:
    '''
    result = abs(real - gen)
    return result


if __name__ == '__main__':
    count, allLat, allLon = 0, 0, 0
    path = r'D:\my_data_set\imgs'
    files = os.listdir(r'D:\my_data_set\imgs')
    files = [file for file in files if file.split('.')[1] != 'png']
    real_files = [file for file in files if file.split('_')[1] == 'real.npy']
    gen_files = [file for file in files if file.split('_')[1] == 'gen.npy']
    for i in range(len(real_files)):
        real_path, gen_path = os.path.join(path, real_files[i]), os.path.join(path, gen_files[i])
        real_data = getFile(real_path)
        gen_data = getFile(gen_path)
        result = Mae(real_data, gen_data)
        count += 1
        Lat = result[:, 0:1, :, :]
        Lon = result[:, 1:2, :, :, ]
        # print(sum(Lat) / 8, sum(Lon) / 8)
        allLat += sum(Lat) / 8
        allLon += sum(Lon) / 8

    avg_Lat = allLat / count
    avg_Lon = allLon / count
    print(avg_Lat, avg_Lon)

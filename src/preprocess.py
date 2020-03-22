'''
对原始数据进行归一化操作，并变成1-1的训练数据形式
'''
import os
import numpy as np


def Normalization(result):
    '''

    :param result:
    :return:归一化之后的标签
    '''
    _range = np.max(result, axis=0) - np.min(result, axis=0)
    return (result - np.min(result, axis=0)) / _range


def renormalization(Norm):
    pass


def Image_normalizeration(Image):
    x_norm = None
    for i in Image:
        _range = np.max(i) - np.min(i)
        image_norm = (i - np.min(i)) / _range
        if x_norm is None:
            x_norm = image_norm
        else:
            x_norm = np.vstack((x_norm, image_norm))
    return x_norm


def getData(path):
    '''
    :param path: 原始目录
    :return: x,y x.shape(n,301,301) y.shape(n,2)
    '''
    y = None
    x = None
    for file_name in os.listdir(path):
        if file_name.split('_')[1] == 'CentLat':
            Loc_file_path = os.path.join(path, file_name)
            file = np.load(Loc_file_path)
            if y is None:
                y = file
            else:
                y = np.vstack((y, file))
        else:
            images_file_path = os.path.join(path, file_name)
            file = np.load(images_file_path)
            file = Image_normalizeration(file)
            if x is None:
                x = file
            else:
                x = np.vstack((x, file))

    return x, Normalization(y)
    # result = np.array(result)
    # print(result.shape)
    # print(file_name.split('_')[1])


if __name__ == '__main__':
    path = r'D:\my_data_set\new_data'
    x, y = getData(path)
    print(x.shape)
    print(y.shape)
    np.savez(r'D:\my_data_set\new_data_norm', x=x, y=y)


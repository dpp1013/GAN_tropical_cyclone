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
    print('_range:{}'.format(_range))
    print('_range:{}'.format(np.min(result, axis=0)))
    return (result - np.min(result, axis=0)) / _range


def renormalization(Norm):
    pass


def Image_normalizeration(Image):
    x_norm = None
    # if x_norm_shape is None:
    #     x_norm_shape = x_norm.shape
    # else:
    #     x_norm_shape = np.vstack((x_norm_shape, x_norm.shape))
    # print(x_norm_shape)
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
    return x_norm, x_norm.shape


def getData(path):
    '''
    :param path: 原始目录
    :return: x,y x.shape(n,301,301) y.shape(n,2)
    '''
    y = None
    x = None

    i = 0
    for file_name in os.listdir(path):
        # i += 1
        # if i == 100:
        #     break
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


# 保存归一化后的文件
def savaData(path, _range, min):
    i = 0
    for file_name in os.listdir(path):
        i += 1
        if i > 100:
            break
        if file_name.split('_')[1] == 'CentLat':
            new_file = os.path.join(r'D:\my_data_set\new_data_norm', file_name)
            Loc_file_path = os.path.join(path, file_name)
            file = np.load(Loc_file_path)
            y = (file - min) / _range
            # np.save(new_file, y)
        else:
            Loc_file_path = os.path.join(path, file_name)
            file = np.load(Loc_file_path)
            print(file.shape)
            x_norm, x_norm_shape = Image_normalizeration(file)
            file_name = str(x_norm_shape[0]) + '_' + file_name
            new_file = os.path.join(r'D:\my_data_set\new_data_norm', file_name)
            np.save(new_file, x_norm)


if __name__ == '__main__':
    path = r'D:\my_data_set\new_data'
    _range = np.array([138.2425, 359.99396])
    min = np.array([-68.5, -180.])
    # getData(path)
    savaData(path, _range, min)

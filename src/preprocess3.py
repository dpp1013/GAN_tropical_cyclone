"""
中心裁剪，归一化，监督化
"""
import numpy as np
import os
import pickle

crop_size = 32  # 中心裁剪的尺寸，(crop_size,crop_size)
in_step = 3  # 监督化，输入步长
out_step = 1  # 监督化，输出输入步长后的第几个

min_lat, max_lat, min_lon, max_lon = -63.5, 75.975, -180.0, 180.0

origin_data_path = r"D:\my_data_set\single_data"
save_data_path = r"D:\my_data_set\tran_data"


def crop_center(img):
    img = img.astype(np.float32)
    img = np.reshape(img, (301, 301))
    y, x = img.shape
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    return np.reshape(img[starty:starty + crop_size, startx:startx + crop_size], (1, crop_size, crop_size)).astype(
        "float")


def crop(data):
    result = None
    for chanels in data:
        img = None
        for chanel in chanels:
            if img is None:
                img = crop_center(chanel)
            else:
                img = np.concatenate((img, crop_center(chanel)))
        if result is None:
            result = img.reshape(tuple([1]) + img.shape)
        else:
            result = np.concatenate((result, img.reshape(tuple([1]) + img.shape)))
    return result


def superversion(data, in_step, out_step):
    count = data.shape[0]
    window_size = in_step + out_step
    if window_size > count:
        print("数据量:%d,window_size:%d,跳过." % (count, window_size))
        return None
    result = None
    for i in range(count - window_size):
        x = data[i:i + in_step].reshape(tuple([in_step]) + data.shape[1:])
        y = data[i + in_step:i + in_step + out_step].reshape(tuple([1]) + data.shape[1:])
        row = np.concatenate((x, y)).reshape(tuple([1, in_step + 1]) + data.shape[1:])
        if result is None:
            result = row
        else:
            result = np.concatenate((result, row))
    return result


def normalization_2d(data, min_value, max_value):
    return (data - min_value) / (max_value - min_value)


def normalization(data):
    result = None
    for chanels in data:
        win, lat, lon = chanels
        win, lat, lon = normalization_2d(win, np.min(win.reshape(-1)), np.max(win.reshape(-1))), \
                        normalization_2d(lat, min_lat, max_lat), \
                        normalization_2d(lon, min_lon, max_lon)
        img = np.array((win, lat, lon))
        if result is None:
            result = img.reshape(tuple([1]) + img.shape)
        else:
            result = np.concatenate((result, img.reshape(tuple([1]) + img.shape)))
    return result


def read_file():
    dir_list = os.listdir(origin_data_path)
    for i in range(len(dir_list)):
        if i < 483:
            continue
        filename = dir_list[i]

        data = np.load(os.path.join(origin_data_path, filename))
        data0 = normalization(data)
        data1 = crop(data0)
        data2 = superversion(data1, in_step, out_step)
        if data2 is not None:
            length, size = data2.shape[0], data2.shape[1]
            new_path = os.path.join(save_data_path, "%d_%d_%s" % (length, size, filename.split("_")[1]))
            np.save(new_path, data2)
            print("result", data2.shape)
            print("save as:", new_path)
            print("进度:", i, "/", len(dir_list))


if __name__ == "__main__":
    read_file()

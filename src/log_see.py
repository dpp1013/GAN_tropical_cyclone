import numpy as np
import re


def sum(str1, str2):
    return int(str1.split('_')[0]) + int(str2.split('_')[0])


if __name__ == "__main__":
    size = 0.01
    batch_size = 64
    fp = open("log.txt", 'r')
    file_list = fp.readlines()[1]
    file_list = [x.strip(' ').strip('\'') for x in file_list[1:-2].split(',')]

    for i in range(len(file_list) - 1):
        file_list[i + 1] = str(sum(file_list[i + 1], file_list[i])) + '_' + file_list[i + 1]
    print(file_list)
    fp.close()
    fp = open("log.txt", 'r')

    lines = [line.strip() for line in fp.readlines() if line.startswith("[E") or line.startswith('lat_mae')]
    blocks = np.array_split(lines, len(lines) / 2)
    count = 0
    ignore_list = []
    for block in blocks:
        row = block[1]
        result = re.findall(r'\d+\.\d+', row, 0)
        lat_mae = float(result[0])
        lon_mae = float(result[3])
        row = block[0]
        result = re.findall(r'\d+', row, 0)
        batch = int(result[2])
        if lat_mae > size and lon_mae > size:
            print(batch, batch * batch_size, lat_mae, lon_mae)
            count += 1
            ignore_list.append(batch)
        else:
            print(batch, batch * batch_size, lat_mae, lon_mae, 'found!!!')
    print("%d/%d %f%%" % (count, len(blocks), count / len(blocks) * 100))
    print(ignore_list) # [0, 1, 2, 3, 4, 5, 6, 23, 24, 25, 27, 28, 29, 30, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 117, 118, 119, 120]

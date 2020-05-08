import pandas as pd

if __name__ == '__main__':
    file = 'D:\my_data_set\map_TC.csv'
    data = pd.read_csv(file, header=None)
    print(data[:])

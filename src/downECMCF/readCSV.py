import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
if __name__ == '__main__':
    file = 'D:\my_data_set\map_TC_y.csv'
    # file_y=
    data = pd.read_csv(file)
    print(data[:])

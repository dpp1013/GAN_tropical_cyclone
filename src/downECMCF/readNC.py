from netCDF4 import Dataset
import netCDF4 as nc


def readNC(path):
    print(path)
    nc_obj = Dataset(path)
    # print(nc_obj)
    print(nc_obj.variables.keys())
    satid_wv = nc_obj.variables['satid_wv']
    satid_ir = nc_obj.variables['satid_ir']
    lat = nc_obj.variables['lat']
    irwin_cdr = nc_obj.variables['irwin_cdr']
    irwin_vza_adj = nc_obj.variables['irwin_vza_adj']

    print('-----------------')


if __name__ == '__main__':
    path = r'D:\ECWMF\GRIDSAT-B1.1980.01.01.00.v02r01.nc'
    readNC(path)

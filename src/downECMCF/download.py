from ecmwfapi import ECMWFDataServer

if __name__ == '__main__':
    server = ECMWFDataServer()
    server.retrieve({
        'target': r'D:\ECWMF\1.nc',
        'dataset': 'interim',
        "class": "EI",
        "type": "AN",
        "stream": "OPER",
        "expver": "0001",
        "repres": "SH",
        "levtype": "SFC",
        "param": "34.128/134.128/165.128/166.128",  # 变量
        "time": "0000/0600/1200/1800",
        "step": "0",
        "domain": "G",
        "resol": "AUTO",
        "area": "2/-120/23/-99",  # 范围
        "grid": "0.25/0.25",  # 精度
        "padding": "0",
        "format":"netcdf",
        "expect": "ANY",
        "date": "19790101/19790102/19790103/19790104/19790105/19790106/19790107/19790108/19790109/19790110/19790111/19790112/19790113/19790114/19790115/19790116/19790117/19790118/19790119/19790120/19790121/19790122/19790123/19790124/19790125/19790126/19790127/19790128/19790129/19790130/19790131"
    })

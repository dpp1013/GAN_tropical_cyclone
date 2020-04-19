from src.lstm_gan3 import *


def avg(x):
    return sum(x) / len(x)


if __name__ == "__main__":
    generator = torch.load('./generator_0.pk')
    data_loader = DataLoader(gen_test_data_set(), batch_size=batch_size, shuffle=True)
    lat_mae_list = []
    lat_mse_list = []
    lat_rmse_list = []
    lon_mae_list = []
    lon_mse_list = []
    lon_rmse_list = []
    for i, (x, y) in enumerate(data_loader):
        print('[%d/%d]' % (i, len(data_loader)))
        x = x.float()
        y = y.float()
        out = generator(x)
        out = out.view((x.shape[0], channels, image_size, image_size))
        lat_mae, lat_mse, lat_rmse, lon_mae, lon_mse, lon_rmse = evaluate_trajectory(out, y)
        lat_mae_list.append(lat_mae)
        lat_mse_list.append(lat_mse)
        lat_rmse_list.append(lat_rmse)
        lon_mae_list.append(lon_mae)
        lon_mse_list.append(lon_mse)
        lon_rmse_list.append(lon_rmse)
    print('all finish')
    print('lat_mae:', avg(lat_mae_list), 'lat_mse:', avg(lat_mae_list), 'lat_rmse:', avg(lat_rmse_list), 'lon_mae:',
          avg(lon_mae_list), 'lon_mse:', avg(lon_mse_list),
          'lon_rmse:', avg(lon_rmse_list))

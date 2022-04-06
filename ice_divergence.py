import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plot as pl
from datetime import date, datetime
import case_information as ci


def dt_from_path(path):
    date1, date2 = path[36:48], path[49:-3]
    datetime1 = datetime(int(date1[:4]), int(date1[4:6]), int(date1[6:8]), int(date1[8:10]), int(date1[10:12]))
    datetime2 = datetime(int(date2[:4]), int(date2[4:6]), int(date2[6:8]), int(date2[8:10]), int(date2[10:12]))

    td = datetime2 - datetime1
    return td.total_seconds()


def del_matrix_neighbour(matrix):
    shape = matrix.shape
    matrix = matrix.flatten()
    matrix_p1 = np.delete(matrix, -1)
    matrix_p1 = np.insert(matrix_p1, 0, np.nan)
    return np.reshape(matrix - matrix_p1, shape)


class IceDivergence:
    def __init__(self):
        self.dir = './data/ice drift/CMEMS'
        self.path_list = os.listdir(self.dir)

        self.div = None

    def ice_drift_correlation(self):
        self.dir = './data/ice drift/CMEMS'
        for path in self.path_list:
            ds = nc.Dataset(self.dir + '/' + path)
            fig, ax = pl.setup_plot(None)
            dY = ds['dY'][:]
            dY[dY == -998.] = 0
            dX = ds['dX'][:]
            dX[dX == -998.] = 0
            im = ax.pcolormesh(ds['lon'][:], ds['lat'][:], ds['data_status'], transform=ccrs.PlateCarree(),
                               cmap='Accent')
            cbar = fig.colorbar(im)
            # cbar.set_ticks([0, 1, 2, 4, 5])
            # cbar.set_ticklabels(['valid', 'correlation less than min', 'drift speed larger than max', 'invalid',
            # 'invalid (filter)'])
            ax.set_title(f'dY-start:{ds.start_date} stop:{ds.stop_date}', fontsize=20)
            plt.savefig(f'{ds.start_date}_to_{ds.stop_date}.png', bbox_inches='tight')
            plt.close(fig)
            print(ds.start_date, ds.stop_date)

    def ice_div(self):
        self.dir = './data/ice drift/CMEMS'
        for path in self.path_list:
            # calculate time difference
            dt = dt_from_path(path)
            # load dataset
            ds = nc.Dataset(self.dir + '/' + path)

            # data validity
            validity = ds['data_status'][:]

            # load displacement values (x and y direction) from ds and replace no data cells with nan
            dY = ds['dY'][:]
            dY[dY == -998.] = 0
            dX = ds['dX'][:]
            dX[dX == -998.] = 0
            # x, y = ds['xc'][:], ds['yc'][:]
            # x, y = np.tile(x, (np.size(y), 1)), np.transpose(np.tile(y, (np.size(x), 1)))

            # calculate drift speed in the x and y direction in m/s
            u, v = dX / (1000*dt), dY / (100*dt)

            # calculate divergence values
            du, dv = del_matrix_neighbour(u), del_matrix_neighbour(v)
            # dx, dy = del_matrix_neighbour(x), del_matrix_neighbour(y)
            div = (du + dv)/20000
            div[validity == 4] = np.nan

            # plot divergence
            # fig, ax = pl.setup_plot(None)
            fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)},
                                           constrained_layout=True)
            fig.set_size_inches(32, 18)
            cap = 1.e-11
            im = ax1.pcolormesh(ds['lon'][:], ds['lat'][:], div, transform=ccrs.PlateCarree(), vmax=cap, vmin=-cap)
            ax1.coastlines(resolution='50m')
            ax1.set_extent(ci.barent_extent, crs=ccrs.PlateCarree())
            cbar1 = fig.colorbar(im, ax=ax1)
            cbar1.ax.tick_params(axis='both', labelsize=25)
            ax1.set_title(f'ice divergence in 1/s \n start:{ds.start_date} stop:{ds.stop_date}', fontsize=25)

            im2 = ax2.pcolormesh(ds['lon'][:], ds['lat'][:], ds['data_status'][:], transform=ccrs.PlateCarree(),
                                 cmap='Accent')
            cbar2 = fig.colorbar(im2, ax=ax2)
            cbar2.set_ticks([0, 1, 2, 4, 5])
            cbar2.set_ticklabels(['valid', 'correlation less than min', 'drift speed larger than max', 'invalid',
                                  'invalid (filter)'])
            cbar2.ax.tick_params(labelsize=25)
            ax2.coastlines(resolution='50m')
            ax2.set_extent(ci.barent_extent, crs=ccrs.PlateCarree())
            ax2.set_title('data status, green means good', fontsize=25)
            plt.savefig(f'./plots/ice divergence/divergence-{ds.start_date}_to_{ds.stop_date}.png', bbox_inches='tight')
            plt.close(fig)
            print(ds.start_date, ds.stop_date)


if __name__ == '__main__':
    IceDivergence().ice_div()

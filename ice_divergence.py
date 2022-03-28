import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plot as pl
from datetime import date, datetime


def dt_from_path(path):
    date1, date2 = path[36:48], path[49:-3]
    datetime1 = datetime(int(date1[:4]), int(date1[4:6]), int(date1[6:8]), int(date1[8:10]), int(date1[10:12]))
    datetime2 = datetime(int(date2[:4]), int(date2[4:6]), int(date2[6:8]), int(date2[8:10]), int(date2[10:12]))

    td = datetime2 - datetime1
    return td.total_seconds() / 3600


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
            im = ax.pcolormesh(ds['lon'][:], ds['lat'][:], dY, transform=ccrs.PlateCarree(), cmap='bwr')
            cbar = fig.colorbar(im)
            #cbar.set_ticks([0, 1, 2, 4, 5])
            #cbar.set_ticklabels(['valid', 'correlation less than min', 'drift speed larger than max', 'invalid',
                                 #'invalid (filter)'])
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

            # load displacement values (x and y direction) from ds and replace no data cells with nan
            dY = ds['dY'][:]
            dY[dY == -998.] = 0
            dX = ds['dX'][:]
            dX[dX == -998.] = 0

            # calculate drift speed in the x and y direction
            dX, dY = dX / dt, dY / dt

            # calculate divergence values
            del_dX = del_matrix_neighbour(dX)
            del_dY = del_matrix_neighbour(dY)

            length = (dX**2 + dY**2)**.5
            div = (del_dX**2 + del_dY**2)**.5 / length

            # plot divergence
            print(div[0][10:20])
            print(dY[0][10:20])
            fig, ax = pl.setup_plot(None)
            im = ax.pcolormesh(ds['lon'][:], ds['lat'][:], div, transform=ccrs.PlateCarree())
            fig.colorbar(im)
            ax.set_title(f'start:{ds.start_date} stop:{ds.stop_date}', fontsize=20)
            plt.savefig(f'divergence-{ds.start_date}_to_{ds.stop_date}.png', bbox_inches='tight')
            plt.close(fig)
            print(ds.start_date, ds.stop_date)



if __name__ == '__main__':
    IceDivergence().ice_drift_correlation()



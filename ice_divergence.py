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


class IceDivergence:
    def __init__(self):
        self.dir = './data/ice drift/CMEMS'
        self.path_list = os.listdir(self.dir)

    def ice_drift_correlation(self):
        self.dir = './data/ice drift/CMEMS'
        for path in self.path_list:
            ds = nc.Dataset(self.dir + '/' + path)
            fig, ax = pl.setup_plot(None)
            im = ax.pcolormesh(ds['lon'][:], ds['lat'][:], ds['data_status'][:], transform=ccrs.PlateCarree(), cmap='Accent')
            cbar = fig.colorbar(im)
            cbar.set_ticks([0, 1, 2, 4, 5])
            cbar.set_ticklabels(['valid', 'correlation less than min', 'drift speed larger than max', 'invalid',
                                 'invalid (filter)'])
            ax.set_title(f'start:{ds.start_date} stop:{ds.stop_date}', fontsize=20)
            plt.savefig(f'{ds.start_date}_to_{ds.stop_date}.png', bbox_inches='tight')
            plt.close(fig)
            print(ds.start_date, ds.stop_date)

    def drift_speed(self):
        self.dir = './data/ice drift/CMEMS'
        for path in self.path_list:
            print(dt_from_path(path))


if __name__ == '__main__':
    IceDivergence().drift_speed()
'''
dY = ds['dY'][:]
dY[dY == -998.] = np.nan

dX = ds['dX'][:]
dX[dX == -998.] = np.nan
'''

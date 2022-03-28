import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plot as pl
from datetime import date, datetime


def ice_drift_correlation():
    dir = './data/ice drift/CMEMS'
    for path in os.listdir(dir):
        ds = nc.Dataset(dir + '/' + path)
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


def time_to_datetime(dates):
    if not isinstance(dates, list):
        print(dates)
        print(int(dates[:4]), int(dates[4:6]), int(dates[6:8]), int(dates[8:10]), int(dates[10:12]))
        return datetime(int(dates[:4]), int(dates[4:6]), int(dates[6:8]), int(dates[8:10]), int(dates[10:12]))
    else:
        return [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in dates]


def ice_drift_speed():
    dir = './data/ice drift/CMEMS'
    for path in os.listdir(dir):
        datetime1, datetime2 = path[36:48], path[49:-3]
        print(time_to_datetime(datetime1))


if __name__ == '__main__':
    ice_drift_speed()
'''
dY = ds['dY'][:]
dY[dY == -998.] = np.nan

dX = ds['dX'][:]
dX[dX == -998.] = np.nan
'''

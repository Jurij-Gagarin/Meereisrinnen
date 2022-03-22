import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plot as pl
path1 = f'./data/ice drift/CMEMS/ice_drift_nh_polstere-200_avhrr-ch4_202001310549_202002010528.nc'

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

'''
dY = ds['dY'][:]
dY[dY == -998.] = np.nan

dX = ds['dX'][:]
dX[dX == -998.] = np.nan
'''

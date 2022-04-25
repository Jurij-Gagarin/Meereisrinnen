import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import plot
import plot as pl
import datetime
import case_information as ci
import data_science as dscience


def dt_from_path(path):
    date1, date2 = path[36:48], path[49:-3]
    datetime1 = datetime.datetime(int(date1[:4]), int(date1[4:6]), int(date1[6:8]), int(date1[8:10]), int(date1[10:12]))
    datetime2 = datetime.datetime(int(date2[:4]), int(date2[4:6]), int(date2[6:8]), int(date2[8:10]), int(date2[10:12]))

    td = datetime2 - datetime1
    return td.total_seconds()


def del_matrix_neighbour(matrix):
    shape = matrix.shape
    matrix = matrix.flatten()
    matrix_p1 = np.delete(matrix, -1)
    matrix_p1 = np.insert(matrix_p1, 0, np.nan)
    return np.reshape(matrix - matrix_p1, shape)


def lonlat_mask(extent, lon, lat):
    extent = list(extent)
    lon_mask1 = lon >= min(extent[:2])
    lon_mask2 = lon <= max(extent[:2])
    lat_mask1 = lat >= min(extent[2:])
    lat_mask2 = lat <= max(extent[2:])
    print(lon_mask1 & lon_mask2 & lat_mask1 & lat_mask2)

    return lon_mask1 & lon_mask2 & lat_mask1 & lat_mask2


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
            # edit this comment
            # x, y = ds['xc'][:], ds['yc'][:]
            # x, y = np.tile(x, (np.size(y), 1)), np.transpose(np.tile(y, (np.size(x), 1)))

            # calculate drift speed in the x and y direction in m/s
            u, v = 1000 * dX / dt, 1000 * dY / dt

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
            cap = 2.e-06
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


class Eumetsat:
    def __init__(self, extent=ci.barent_extent):
        self.dir = './data/ice drift/Eumetsat'
        path = 'ice_drift_nh_polstere-625_multi-oi_202002171200-202002191200.nc'
        self.path_list = os.listdir(self.dir)
        self.data_sets = {}
        self.extent = extent

        for path in self.path_list:
            ds = nc.Dataset(self.dir + '/' + path)
            date = (datetime.datetime(1978, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=int(ds['time'][0]))).date()
            self.data_sets[date] = ds

        dummy = list(self.data_sets.values())[0]
        self.lon, self.lat = dummy['lon'][:], dummy['lat'][:]
        self.lonlat_mask = ~lonlat_mask(self.extent, self.lon, self.lat)

    def ice_div(self, date):
        # choose the right data set corresponding to date
        ds = self.data_sets[dscience.string_time_to_datetime(date)]

        # Get Variables for ice displacement in km
        # set masked values to NaN
        dY = ds['dY'][0, :]
        dY[dY.mask] = np.nan
        dY[self.lonlat_mask] = np.nan
        dX = ds['dX'][0, :]
        dX[dX.mask] = np.nan
        dX[self.lonlat_mask] = np.nan

        # observation time (48h) in seconds
        dt = 172800

        # calculate drift speed in the x and y direction in km/s
        u, v = dX / dt, dY / dt

        # calculate divergence values in 1/s
        # distance between two cells is always 62.5 km (both x,y direction)
        du, dv = del_matrix_neighbour(u), del_matrix_neighbour(v)
        return (du + dv)/62.5

    def plot_div(self, dates):
        divs = []
        cap = 0
        for date in dates:
            div = self.ice_div(date)
            cap = max([cap, abs(div.min()), abs(div.max())])
            divs.append(div)

        for date, div in zip(dates, divs):
            fig, ax = plot.setup_plot(self.extent)
            im = ax.pcolormesh(self.lon, self.lat, div, transform=ccrs.PlateCarree(), vmax=cap, vmin=-cap, cmap='bwr')
            ax.set_title(f'Ice divergence in 1/s \n {dscience.string_time_to_datetime(date)}', fontsize=25)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(axis='both', labelsize=25)
            plot.show_plot(fig, f'./plots/ice divergence/divergence-{dscience.string_time_to_datetime(date)}.png',
                           False)


if __name__ == '__main__':
    Eumetsat().plot_div(dscience.time_delta('20200210', '20200229'))

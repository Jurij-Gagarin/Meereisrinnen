import netCDF4 as nc
import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import plot
import plot as pl
import datetime
import case_information as ci
import data_science as dscience
import cftime



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


def matrix_neighbour_diff(x_matrix, y_matrix):
    div_shape = x_matrix.shape
    hy = np.empty((div_shape[0], 1))
    vx = np.empty((1, div_shape[1]))
    hy[:], vx[:] = np.nan, np.nan
    up1 = np.hstack((np.hstack((x_matrix, hy)), hy))
    vp1 = np.vstack((vx, np.vstack((vx, y_matrix))))
    up1, vp1 = np.delete(up1, 0, 1), np.delete(vp1, -1, 0)
    up1, vp1 = np.delete(up1, 0, 1), np.delete(vp1, -1, 0)
    du, dv = up1 - x_matrix, vp1 - y_matrix

    return du, dv


def lonlat_mask(extent, lon, lat):
    extent = list(extent)
    lon_mask1 = lon >= min(extent[:2])
    lon_mask2 = lon <= max(extent[:2])
    lat_mask1 = lat >= min(extent[2:])
    lat_mask2 = lat <= max(extent[2:])

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
    def __init__(self, extent):
        self.drift_width = {ci.barent_extent: .008, ci.arctic_extent: None}
        self.drift_scale = {ci.barent_extent: 5, ci.arctic_extent: 10}
        cols = {ci.barent_extent: 6, ci.arctic_extent: 4}
        file_dict = {ci.barent_extent: 'Barent Sea', ci.arctic_extent: 'Arctic'}

        self.dir = './data/ice drift/Eumetsat/2010-2022/'
        self.path_list = os.listdir(self.dir)
        self.data_sets = {}
        self.extent = extent
        self.nrows = 2
        self.ncols = cols[self.extent]
        self.prod = self.ncols
        self.file = file_dict[self.extent]

        self.time = None
        self.skip = 2
        self.ds_drift = nc.Dataset('./data/drift_combined.nc')

        for path in self.path_list:
            if path == '.DS_Store':
                break

            ds = nc.Dataset(self.dir + path)
            date = (datetime.datetime(1978, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=int(ds['time'][0]))).date()
            self.data_sets[date] = ds

        dummy = list(self.data_sets.values())[0]
        self.lon, self.lat = dummy['lon'][:], dummy['lat'][:]
        self.xc, self.yc = 1000*dummy['xc'][:], 1000*dummy['yc'][:]
        self.lonlat_mask = ~lonlat_mask(self.extent, self.lon, self.lat)

    def get_disp(self, date):
        # choose the right data set corresponding to date
        ds = self.data_sets[dscience.string_time_to_datetime(date) + datetime.timedelta(days=1)]

        # Get Variables for ice displacement in km
        # set masked values to NaN
        dY = ds['dY'][0, :]
        dY[dY.mask] = np.nan
        dY[self.lonlat_mask] = np.nan
        dX = ds['dX'][0, :]
        dX[dX.mask] = np.nan
        dX[self.lonlat_mask] = np.nan

        # correct dY for different axis
        if ds.product_version < 1.4:
            dY = -dY
        return dX, dY

    def get_drift(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 12, 0, 0, 0) - datetime.timedelta(days=1)
        self.time = self.ds_drift['time']
        t1 = cftime.date2index(d1, self.time)

        # return drift speed in m/s
        return [1000 * self.ds_drift['dX'][t1] / 172800, 1000 * self.ds_drift['dY'][t1] / 172800]
        pass

    def ice_div(self, date):
        # choose the right data set corresponding to date
        dX, dY = self.get_disp(date)
        # observation time (48h) in seconds
        dt = 172800

        # calculate drift speed in the x and y direction in km/s
        u, v = dX / dt, dY / dt

        # calculate divergence values in 1/s
        # distance between two cells is always 62.5 km (both x,y direction)
        #du, dv = matrix_neighbour_diff(u, v)

        du, dv = matrix_neighbour_diff(u, v)

        return (du + dv)/125

    def ice_shear(self, date):
        # choose the right data set corresponding to date
        dX, dY = self.get_disp(date)
        # observation time (48h) in seconds
        dt = 172800

        # calculate drift speed in the x and y direction in km/s
        u, v = dX / dt, dY / dt

        # calculate divergence values in 1/s
        # distance between two cells is always 62.5 km (both x,y direction)
        # du, dv = matrix_neighbour_diff(u, v)

        hy = np.empty((dY.shape[0], 1))
        vx = np.empty((1, dX.shape[1]))
        hy[:], vx[:] = np.nan, np.nan
        up1 = np.hstack((np.hstack((u, hy)), hy))
        vp1 = np.vstack((vx, np.vstack((vx, v))))
        up1, vp1 = np.delete(up1, 0, 1), np.delete(vp1, -1, 0)
        up1, vp1 = np.delete(up1, 0, 1), np.delete(vp1, -1, 0)
        du, dv = up1 - u, vp1 - v

        return (du - dv) / 125

    def plot_div(self, dates):
        divs = []
        cap = 0
        for date in dates:
            div = self.ice_div(date)
            cap = max([cap, abs(div.min()), abs(div.max())])
            divs.append(div)

        for date, div in zip(dates, divs):
            print(date)
            fig, ax = plot.setup_plot(self.extent)
            im = ax.pcolormesh(self.lon, self.lat, div, transform=ccrs.PlateCarree(), vmax=cap, vmin=-cap, cmap='bwr')
            ax.set_title(f'Ice divergence in 1/s \n {dscience.string_time_to_datetime(date)}', fontsize=25)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(axis='both', labelsize=25)
            plot.show_plot(fig, f'./plots/ice divergence/divergence-{dscience.string_time_to_datetime(date)}.png',
                           False)

    def plot_drift(self, dates, save=True):
        quivs = []
        lengths = []
        cap = 0
        factor = 1000 / 172800
        images = []
        for date in dates:
            quiv = self.get_disp(date)
            quivs.append(quiv)
            lengths.append((quiv[0]**2 + quiv[1]**2)**.5)
            cap = max([cap, lengths[-1].max()])

        for date, quiv, length in zip(dates, quivs, lengths):
            print(date)
            print(self.xc.shape, self.yc.shape)
            fig, ax = plot.setup_plot(self.extent)
            im = ax.quiver(self.xc, self.yc, quiv[0] * factor, quiv[1] * factor, length * factor, scale=10, clim=(None, cap * factor),
                           transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm')
            ax.set_title(f'Ice drift in m/s \n {dscience.string_time_to_datetime(date)}', fontsize=25)
            cbar = fig.colorbar(im)
            cbar.ax.tick_params(axis='both', labelsize=25)
            plt.show()
            plot.show_plot(fig, f'./plots/ice divergence/displacement-{dscience.string_time_to_datetime(date)}.png',
                           save)

    def setup_plot(self):
        # create figure and base map
        fig, ax = plt.subplots(self.nrows, self.ncols,
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.coastlines(resolution='50m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def plot_drift_wind(self, dates):
        quivs = []
        wind_quivs = []
        w_lenghts = []
        lengths = []
        q_cap = 0
        w_cap = 0
        factor = 1000 / 172800
        im = None
        lon, lat = None, None
        dim = (50, 50)
        for date in dates:
            quiv = self.get_disp(date)
            quivs.append(quiv)
            lengths.append((quiv[0] ** 2 + quiv[1] ** 2) ** .5)
            q_cap = max([q_cap, lengths[-1].max()])

            Wind, resize = plot.ds_from_var('wind_quiver', date)
            lat_dir, lon_dir = Wind.get_quiver(date)
            lon, lat = resize(Wind.lon, dim), resize(Wind.lat, dim)
            wind_quiv = resize(lon_dir, dim), resize(lat_dir, dim)
            wind_quivs.append(wind_quiv)
            w_lenghts.append((wind_quiv[0] ** 2 + wind_quiv[1] ** 2) ** .5)
            w_cap = max([w_cap, w_lenghts[-1].max()])

        for i in range(int(len(dates) / 6)):
            fig, axs = self.setup_plot()
            print(i)
            for j, (ax, quiv, length) in enumerate(zip(axs[0], quivs[i * 6:i * 6 + 7], lengths[i * 6:i * 6 + 7])):
                im = ax.quiver(self.xc, self.yc, quiv[0] * factor, quiv[1] * factor, length * factor, scale=5,
                               clim=(0, q_cap * factor), transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm', width=.008)
                ax.set_title(f'{dscience.string_time_to_datetime(dates[6 * i + j])}', fontsize=20)
                print(dates[6 * i + j])
            cbar = fig.colorbar(im, ax=axs[0])
            cbar.set_label('ice drift in m/s', size=18)
            cbar.ax.tick_params(labelsize=15)

            for j, (ax, w_quiv, w_length) in enumerate(zip(axs[1], wind_quivs[i * 6:i * 6 + 7], w_lenghts[i * 6:i * 6 + 7])):
                v10, u10 = w_quiv
                im = ax.quiver(lon, lat, v10, u10, w_length, clim=(0, w_cap), transform=ccrs.PlateCarree(),
                               cmap='coolwarm', scale=150, width=.008)
            cbar = fig.colorbar(im, ax=axs[1])
            cbar.set_label('wind in m/s', size=18)
            cbar.ax.tick_params(labelsize=15)

            plot.show_plot(fig, f'./plots/ice divergence/{self.file}/drift_wind{dates[i * 6]}_{dates[i * 6 + 5]}.png',
                           False)

    def plot_drift_div(self, dates):
        quivs = []
        divs = []
        lengths = []
        q_cap = 0
        d_cap = 0
        factor = 1000 / 172800
        im = None
        for date in dates:
            quiv = self.get_disp(date)
            div = self.ice_div(date)
            quivs.append(quiv)
            lengths.append((quiv[0] ** 2 + quiv[1] ** 2) ** .5)
            q_cap = max([q_cap, lengths[-1].max()])
            d_cap = max([d_cap, abs(div.min()), abs(div.max())])
            divs.append(div)

        for i in range(int(len(dates)/self.ncols)):
            fig, axs = self.setup_plot()
            begin, end = i*self.prod, i*self.prod+self.prod + 1
            for j, (ax, quiv, length) in enumerate(zip(axs[0], quivs[begin:end], lengths[begin:end])):
                im = ax.quiver(self.xc, self.yc, quiv[0] * factor, quiv[1] * factor, length * factor,
                               scale=self.drift_scale[self.extent], width=self.drift_width[self.extent],
                               clim=(0, q_cap * factor), transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm')
                ax.set_title(f'{dscience.string_time_to_datetime(dates[self.prod * i + j])}', fontsize=20)
                print(dates[self.prod * i + j])
            cbar = fig.colorbar(im, ax=axs[0])
            cbar.set_label('ice drift in m/s', size=18)
            cbar.ax.tick_params(labelsize=15)

            for j, (ax, div) in enumerate(zip(axs[1], divs[begin:end])):
                im = ax.pcolormesh(self.lon, self.lat, div, transform=ccrs.PlateCarree(), vmax=d_cap, vmin=-d_cap,
                                   cmap='bwr')
            cbar = fig.colorbar(im, ax=axs[1])
            cbar.set_label(r'ice divergence (pos)/convergence (neg) in $10^{-6}/s$', size=18)
            cbar.ax.tick_params(labelsize=15)
            plot.show_plot(fig, f'./plots/ice divergence/{self.file}/divergence_drift_'
                                f'{dates[begin]}_{dates[begin + self.prod - 1]}.png', False)

    def plot_drift_vort(self, dates):
        quivs = []
        vorts = []
        lengths = []
        q_cap = 0
        d_cap = 0
        factor = 1000 / 172800
        im = None
        for date in dates:
            quiv = self.get_disp(date)
            vort = self.ice_shear(date)
            quivs.append(quiv)
            lengths.append((quiv[0] ** 2 + quiv[1] ** 2) ** .5)
            q_cap = max([q_cap, lengths[-1].max()])
            d_cap = max([d_cap, abs(vort.min()), abs(vort.max())])
            vorts.append(vort)

        for i in range(int(len(dates)/self.ncols)):
            fig, axs = self.setup_plot()
            begin, end = i*self.prod, i*self.prod+self.prod + 1
            for j, (ax, quiv, length) in enumerate(zip(axs[0], quivs[begin:end], lengths[begin:end])):
                im = ax.quiver(self.xc, self.yc, quiv[0] * factor, quiv[1] * factor, length * factor,
                               scale=self.drift_scale[self.extent], width=self.drift_width[self.extent],
                               clim=(0, q_cap * factor), transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm')
                ax.set_title(f'{dscience.string_time_to_datetime(dates[self.prod * i + j])}', fontsize=20)
                print(dates[self.prod * i + j])
            cbar = fig.colorbar(im, ax=axs[0])
            cbar.set_label('ice drift in m/s', size=18)
            cbar.ax.tick_params(labelsize=15)

            for j, (ax, vort) in enumerate(zip(axs[1], vorts[begin:end])):
                im = ax.pcolormesh(self.lon, self.lat, vort, transform=ccrs.PlateCarree(), vmax=d_cap, vmin=-d_cap,
                                   cmap='bwr')
            cbar = fig.colorbar(im, ax=axs[1])
            cbar.set_label(r'ice vorticity', size=18)
            cbar.ax.tick_params(labelsize=15)
            plot.show_plot(fig, f'./plots/ice divergence/{self.file}/vort_drift_'
                                f'{dates[begin]}_{dates[begin + self.prod - 1]}.png', False)

    def plot_div_leads(self, dates, new=False):
        divs = []
        lead = []
        msls = []
        d_cap = 0
        im = None
        lon, lat = leads.CoordinateGrid().lon, leads.CoordinateGrid().lat
        m_lon, m_lat = leads.Era5('msl').lon, leads.Era5('msl').lat
        path = None

        for date in dates:
            msl = leads.Era5('msl').get_variable(date)
            div = self.ice_div(date)
            d_cap = max([d_cap, abs(div.min()), abs(div.max())])
            divs.append(div)
            msls.append(msl)

            if new:
                lead.append(leads.Lead(date).new_leads())
                path = 'newlead'
            else:
                lead.append(leads.Lead(date).lead_data * 100)
                path = 'lead'

        for i in range(int(len(dates)/self.ncols)):
            fig, axs = self.setup_plot()
            begin, end = i * self.prod, i * self.prod + self.prod + 1

            for j, (ax, l, msl) in enumerate(zip(axs[0], lead[begin: end], msls[begin: end])):
                print(dates[self.prod * i + j])
                cim = ax.contour(m_lon, m_lat, msl, transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                ax.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                im = ax.pcolormesh(lon, lat, l, transform=ccrs.PlateCarree(), cmap='cool', vmin=0, vmax=100)
                ax.set_title(f'{dscience.string_time_to_datetime(dates[self.prod * i + j])}', fontsize=20)
            cbar = fig.colorbar(im, ax=axs[0])
            cbar.set_label(f'{path} in %', size=18)
            cbar.ax.tick_params(labelsize=15)
            for j, (ax, div) in enumerate(zip(axs[1], divs[begin:end])):
                im = ax.pcolormesh(self.lon, self.lat, div, transform=ccrs.PlateCarree(), vmax=d_cap, vmin=-d_cap,
                                   cmap='bwr')
            cbar = fig.colorbar(im, ax=axs[1])
            cbar.set_label(r'ice divergence (pos)/convergence (neg) in $10^{-6}/s$', size=18)
            cbar.ax.tick_params(labelsize=15)
            plot.show_plot(fig, f'./plots/ice divergence/{self.file}/divergence_{path}_'
                                f'{dates[begin]}_{dates[begin + self.prod - 1]}.png', False)

    def plot_drift_leads(self, dates, new=False):
        quivs = []
        lead = []
        msls = []
        lengths = []
        q_cap = 0
        im = None
        lon, lat = leads.CoordinateGrid().lon, leads.CoordinateGrid().lat
        m_lon, m_lat = leads.Era5('msl').lon, leads.Era5('msl').lat
        factor = 100
        path = None
        print(self.xc.size)
        self.xc, self.yc = self.xc[::self.skip], self.yc[::self.skip]

        for date in dates:
            msl = leads.Era5('msl').get_variable_drift(date)
            quiv = self.get_drift(date)
            msls.append(msl)
            quivs.append(quiv)
            lengths.append((quiv[0] ** 2 + quiv[1] ** 2) ** .5)
            q_cap = max([q_cap, lengths[-1].max()])

            if new:
                lead.append(leads.Lead(date).new_leads())
                path = 'newlead'
            else:
                lead.append(leads.Lead(date).lead_data*100)
                path = 'lead'

        count = 0
        for i in range(int(len(dates) / self.ncols)):
            fig, axs = self.setup_plot()
            begin, end = i * self.prod, i * self.prod + self.prod + 1

            for j, (ax1, ax2) in enumerate(zip(axs[0], axs[1])):
                print(dates[count])
                cim = ax1.contour(m_lon, m_lat, msls[count], transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                lim = ax1.pcolormesh(lon, lat, lead[count], transform=ccrs.PlateCarree(), cmap='cool', vmin=0, vmax=100)
                ax1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                ax1.set_title(f'{dscience.string_time_to_datetime(dates[count])}', fontsize=20)

                qim = ax2.quiver(self.xc, self.yc, quivs[count][0][::self.skip, ::self.skip] * factor,
                                quivs[count][1][::self.skip, ::self.skip] * factor,
                                lengths[count][::self.skip, ::self.skip] * factor, scale=500,
                                clim=(0, q_cap * factor), transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm',
                                )  # width=.008, scale=5
                count += 1

            cbar = fig.colorbar(lim, ax=axs[0])
            cbar.set_label(f'{path} fraction in %', size=18)
            cbar.ax.tick_params(labelsize=15)

            cbar = fig.colorbar(qim, ax=axs[1])
            cbar.set_label(r'ice drift in cm/s', size=18)
            cbar.ax.tick_params(labelsize=15)

            plot.show_plot(fig, f'./plots/ice divergence/{self.file}/drift lead/drift_{path}_'
                                f'{dates[begin]}_{dates[begin + self.prod - 1]}.png', False)


class GeneralEumetsat:
    def __init__(self, date, extent=ci.arctic_extent):
        self.extent = extent
        self.nrows, self.ncols = 1, 2
        # load data set for given day
        dt_date = dscience.string_time_to_datetime(date)
        dt_date_p2 = dt_date + datetime.timedelta(days=2)
        date_p2 = dscience.datetime_to_string(dt_date_p2)
        self.path = './data/ice drift/Eumetsat/2010-2022/'
        self.path = self.path + f'ice_drift_nh_polstere-625_multi-oi_{date}1200-{date_p2}1200.nc'
        ds = nc.Dataset(self.path)

        # get displacement
        self.dX = ds['dX'][0, :] * 1000
        self.dY = ds['dY'][0, :] * 1000
        self.dY[self.dY.mask] = np.nan
        self.dX[self.dX.mask] = np.nan
        if float(ds.product_version) < 1.4:
            self.dY = -self.dY

        # calculate drift speed from displacement
        dt = 172800
        self.u = self.dX / dt
        self.v = self.dY / dt

        # get coords
        self.xc = ds['xc'][:] * 1000
        self.yc = ds['yc'][:] * 1000
        self.xx, self.yy = np.meshgrid(self.xc, self.yc, indexing='xy')
        print(self.xx)
        print()
        print(self.yy)

    def setup_plot(self):
        # create figure and base map
        fig, ax = plt.subplots(self.nrows, self.ncols,
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.coastlines(resolution='50m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def calculate_divergence(self):
        points = [self.xc, self.yc]
        # calculate spatial difference in the grid. This should always be 62.5km
        sp = [np.diff(p)[0] for p in points]
        return divergence(np.array([self.u, self.v]), sp)

    def plot_divergence(self):
        fig, (ax1, ax2) = self.setup_plot()
        n_skip = None
        skip = (slice(None, None, n_skip), slice(None, None, n_skip))

        im2 = ax2.pcolormesh(self.xx, self.yy, self.calculate_divergence(), cmap='coolwarm', vmin=-2.e-6, vmax=2.e-6,
                             transform=ccrs.NorthPolarStereo(-45))
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        print(self.xx.shape, self.yy.shape, self.u.shape, self.v.shape)
        dist = np.sqrt(self.u ** 2 + self.v ** 2)
        im1 = ax1.quiver(self.xx[skip], self.yy[skip], self.u[skip], self.v[skip], dist[skip], cmap='coolwarm',
                         transform=ccrs.NorthPolarStereo(-45))
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        plt.show()


def divergence(f, sp):
    """ Computes divergence of vector field
    f: array -> vector field components [Fx,Fy,Fz,...]
    sp: array -> spacing between points in respecitve directions [spx, spy,spz,...]
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], sp[i], axis=i) for i in range(num_dims)])


def divergence_test():
    # Number of points (NxN)
    N = 51
    # Boundaries
    ymin = -2.
    ymax = 2.
    xmin = -2.
    xmax = 2.

    # Create Meshgrid
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    x, y = np.flip(x), np.flip(y)
    xx, yy = np.meshgrid(x, y)

    print(xx)
    print()
    print(yy)

    xx, yy = np.meshgrid(x, y, indexing='ij')
    print()
    print(xx)
    print()
    print(yy)

    Fx = np.cos(xx + 2 * yy)
    Fy = np.sin(xx - 2 * yy)
    F = np.array([Fx, Fy])

    # Analytic computation of the divergence (EXACT)
    div_analy = -np.sin(xx + 2 * yy) - 2 * np.cos(xx - 2 * yy)

    # Compute Divergence
    points = [x, y]
    sp = [np.diff(p)[0] for p in points]
    div_num = divergence(F, sp)

    # PLOT
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))
    im1 = ax1.pcolormesh(xx, yy, div_analy, cmap="jet")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set_title('analytical solution')

    im2 = ax2.pcolormesh(xx, yy, div_num, cmap="jet")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_title('numerical solution')

    n_skip = 5
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    ax2.quiver(xx[skip], yy[skip], Fx[skip], Fy[skip])
    ax1.quiver(xx[skip], yy[skip], Fx[skip], Fy[skip])
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('divergence_test.png', bbox_inches='tight', dpi=200)


if __name__ == '__main__':
    # all plots related to ice drift Barent Sea
    # all_dates = dscience.time_delta('20200102', '20200429')
    # Eumetsat(ci.barent_extent).plot_drift_leads(all_dates, True)
    # Eumetsat(ci.barent_extent).plot_drift_leads(all_dates, False)
    # Eumetsat(ci.barent_extent).plot_drift_div(all_dates)
    # Eumetsat(ci.barent_extent).plot_div_leads(all_dates, True)
    # Eumetsat(ci.barent_extent).plot_div_leads(all_dates, False)
    # Eumetsat(ci.barent_extent).plot_drift_vort(all_dates)

    # all plots related to ice drift entire Arctic
    # Eumetsat(ci.arctic_extent).plot_drift_leads(all_dates, True)
    # Eumetsat(ci.arctic_extent).plot_drift_leads(all_dates, False)
    # Eumetsat(ci.arctic_extent).plot_drift_div(all_dates)
    # Eumetsat(ci.arctic_extent).plot_div_leads(all_dates, True)
    # Eumetsat(ci.arctic_extent).plot_div_leads(all_dates, False)

    GeneralEumetsat('20100101').plot_divergence()
    GeneralEumetsat('20200228').calculate_divergence()

    #divergence_test()







    pass

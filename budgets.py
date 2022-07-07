# Calculate ice dynamic budgets in hear

import datetime
import data_science as ds
import cftime
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import case_information as ci
import plot


class IceData:
    def __init__(self, extent=ci.arctic_extent):
        self.ds_spring = nc.Dataset('./data/ERA5_METAs_remapbil_drift.nc')
        self.ds_winter = nc.Dataset('./data/ERA5_METAw_remapbil_drift.nc')
        self.ds_drift = nc.Dataset('./data/drift_combined.nc')

        self.extent = extent
        self.nrows, self.ncols = 2, 4
        self.time = None
        self.xc = 1000*self.ds_spring['xc'][:]
        self.yc = 1000*self.ds_spring['yc'][:]
        self.step_size = 62500
        self.dt = 24*60*60

    def get_variable(self, date, variable='siconc'):
        data_set = None
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        if d1.year == 2019:
            data_set = self.ds_winter
        elif d1.year == 2020:
            data_set = self.ds_spring

        self.time = data_set['time']
        var = data_set[variable][:]
        t1, t2 = cftime.date2index([d1, d2], self.time)  # cftime nows unit from ds

        mean_var = np.zeros(var[0].shape)
        for t in range(t1, t2+1):
            mean_var += var[t]

        return .25 * mean_var

    def get_drift(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 12, 0, 0, 0) - datetime.timedelta(days=1)
        self.time = self.ds_drift['time']
        t1 = cftime.date2index(d1, self.time)

        # return drift speed in m/s
        return 1000*self.ds_drift['dX'][t1]/172800, 1000*self.ds_drift['dY'][t1]/172800

    def setup_plot(self):
        # create figure and base map
        fig, ax = plt.subplots(self.nrows, self.ncols,
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.coastlines(resolution='50m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def plot_siconc(self, date1, date2):
        dates = ds.time_delta(date1, date2)
        im = None
        count = 0
        for i in range(int(np.floor(len(dates) / (self.nrows * self.ncols)))):
            fig, axs = self.setup_plot()
            for ax in axs:
                for a in ax:
                    print(dates[count])
                    im = a.pcolormesh(self.xc, self.yc, self.get_variable(dates[count]), vmin=0, vmax=1,
                                      transform=ccrs.NorthPolarStereo(-45), cmap='Blues')
                    a.set_title(str(ds.string_time_to_datetime(dates[count])), fontsize=20)
                    count += 1
            cbar = fig.colorbar(im, ax=axs)
            cbar.ax.tick_params(axis='both', labelsize=20)
            plot.show_plot(fig, f'./plots/budgets/siconc_{dates[count - self.nrows * self.ncols]}-{dates[count]}.png',
                           False)

    def plot_drift(self, date1, date2):
        # Gather all drift data in an array, find min and max values
        dates = ds.time_delta(date1, date2)
        us, vs, lengths = [], [], []
        min_cap, max_cap = 0, 0

        for date in dates:
            u, v = self.get_drift(date)
            us.append(u)
            vs.append(v)
            lengths.append((u**2 + v**2)**.5)
            min_cap, max_cap = min(min_cap, np.min(lengths[-1])), max(max_cap, np.max(lengths[-1]))

        # Make Plots
        im = None
        count = 0
        for i in range(int(np.floor(len(dates) / (self.nrows * self.ncols)))):
            fig, axs = self.setup_plot()
            for ax in axs:
                for a in ax:
                    print(dates[count])
                    im = a.quiver(self.xc, self.yc, us[count], vs[count], lengths[count], clim=(min_cap, max_cap),
                                  transform=ccrs.NorthPolarStereo(-45), cmap='coolwarm', scale=8)
                    a.set_title(str(ds.string_time_to_datetime(dates[count])), fontsize=20)
                    count += 1
            cbar = fig.colorbar(im, ax=axs)
            cbar.ax.tick_params(axis='both', labelsize=20)
            plot.show_plot(fig, f'./plots/budgets/drift/drift_{dates[count - self.nrows * self.ncols]}-{dates[count]}.png',
                           False)

    def divergence(self, ux, uy, C):
        du = (ux[:, :-2] - ux[:, 2:])[1:-1]
        dv = (uy[:-2] - uy[2:])[:, 1:-1]

        return np.multiply(-C, (du + dv)/(2*self.step_size))

    def advection(self, ux, uy, C):
        dCdx = (C[:, :-2] - C[:, 2:])[1:-1]
        dCdy = (C[:-2] - C[2:])[:, 1:-1]

        return -np.multiply(ux, dCdx) - np.multiply(uy, dCdy)

    def intensification(self, C1, C2, dt):
        return (C2 - C1) / dt

    def get_budgets(self, date1, date2):
        dates = ds.time_delta(date1, date2)
        advs, divs, ints, ress = [], [], [], []

        for date1, date2 in zip(dates[:-1], dates[1:]):
            C1, C2 = self.get_variable(date2, 'siconc'), self.get_variable(date2, 'siconc')
            ux, uy = self.get_drift(date2)

            advs.append(self.advection(ux, uy, C2))
            divs.append(self.divergence(ux, uy, C2))
            ints.append(self.intensification(C1, C2, self.dt))
            ress.append(ints[-1] - advs[-1] - divs[-1])

        return dates, advs, divs, ints, ress








if __name__ == '__main__':
    arr1 = np.random.randint(10, size=(10, 10))
    arr2 = np.random.randint(10, size=(10, 10))
    print(arr1)
    print(arr2)
    print(np.multiply(arr1, arr2))
    #print(arr)
    #print()
    #print(arr[:-2])
    #print(arr[2:])
    #print()
    #print(arr[:, :-2])
    #print(arr[:, 2:])
    #print((arr[:-2]-arr[2:])[:, 1:-1])
    # IceData().plot_drift('20191201', '20200331')
    pass


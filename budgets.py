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
from calendar import monthrange
from pandas import date_range


def max_matrix(cap, M):
    return max(cap, abs(np.nanmin(M)), np.nanmax(M))


class IceData:
    def __init__(self, extent=ci.arctic_extent):
        self.ds_spring = nc.Dataset('./data/ERA5_METAs_remapbil_drift.nc')
        self.ds_winter = nc.Dataset('./data/ERA5_METAw_remapbil_drift.nc')
        self.ds_spring_monthly = nc.Dataset('./data/ERA5_avg_METAs_remapbil_drift.nc')
        self.ds_winter_monthly = nc.Dataset('./data/ERA5_avg_METAw_remapbil_drift.nc')
        self.ds_drift = nc.Dataset('./data/drift_combined.nc')
        self.ds_drift_monthly = nc.Dataset('./data/drift_maverage.nc')

        self.extent = extent
        self.nrows, self.ncols = 2, 4
        self.time = None
        self.xc = 1000 * self.ds_spring['xc'][:]
        self.yc = 1000 * self.ds_spring['yc'][:]
        self.step_size = 62500
        self.dt = 24 * 60 * 60

        self.advs, self.divs, self.ints, self.ress = [], [], [], []
        self.adv_cap, self.div_cap, self.int_cap, self.res_cap = 0, 0, 0, 0

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
        for t in range(t1, t2 + 1):
            mean_var += var[t]

        mean_var[mean_var < 0.0] = np.nan
        return .25 * mean_var

    def get_drift(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 12, 0, 0, 0) - datetime.timedelta(days=1)
        self.time = self.ds_drift['time']
        t1 = cftime.date2index(d1, self.time)

        # return drift speed in m/s
        return 1000 * self.ds_drift['dX'][t1] / 172800, 1000 * self.ds_drift['dY'][t1] / 172800

    def get_monthly(self, month, year):
        ds = None
        _, day = monthrange(year, month)
        t = cftime.date2index(datetime.datetime(year, month, day, 12, 0, 0), self.ds_drift_monthly['time'])

        if year == 2020:
            ds = self.ds_spring_monthly
        elif year == 2019:
            ds = self.ds_winter_monthly

        t_sic = cftime.date2index(datetime.datetime(year, month, day, 18, 0, 0), ds['time'])
        siconc = ds['siconc'][t_sic]

        siconc[siconc == -32767] = np.nan
        self.ds_drift_monthly['dX'][t][self.ds_drift_monthly['dX'][t] == -10000000000.0] = np.nan
        self.ds_drift_monthly['dY'][t][self.ds_drift_monthly['dY'][t] == -10000000000.0] = np.nan

        return 1000 * self.ds_drift_monthly['dX'][t] / 172800, 1000 * self.ds_drift_monthly['dY'][t] / 172800, siconc



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
                                      transform=ccrs.NorthPolarStereo(-45), cmap='viridis')
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
            lengths.append((u ** 2 + v ** 2) ** .5)
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
            plot.show_plot(fig,
                           f'./plots/budgets/drift/drift_{dates[count - self.nrows * self.ncols]}-{dates[count]}.png',
                           False)

    def divergence(self, ux, uy, C):
        du = (ux[:, 2:] - ux[:, :-2])[1:-1]
        dv = (uy[2:] - uy[:-2])[:, 1:-1]

        return np.multiply(-C[1:-1, 1:-1], (du + dv) / (2 * self.step_size))

    def advection(self, ux, uy, C):
        dCdx = (C[:, :-2] - C[:, 2:])[1:-1] / (2 * self.step_size)
        dCdy = (C[:-2] - C[2:])[:, 1:-1] / (2 * self.step_size)

        return -np.multiply(ux[1:-1, 1:-1], dCdx) - np.multiply(uy[1:-1, 1:-1], dCdy)

    def intensification(self, C1, C2):
        return (C2 - C1)[1:-1, 1:-1] / self.dt

    def get_budgets(self, date1, date2):
        dates = ds.time_delta(date1, date2)

        for date1, date2 in zip(dates[:-1], dates[1:]):
            C1, C2 = self.get_variable(date1, 'siconc'), self.get_variable(date2, 'siconc')
            ux, uy = self.get_drift(date2)

            self.advs.append(self.advection(ux, uy, C2))
            self.adv_cap = max_matrix(self.adv_cap, self.advs[-1])
            self.divs.append(self.divergence(ux, uy, C2))
            self.div_cap = max_matrix(self.div_cap, self.divs[-1])
            self.ints.append(self.intensification(C1, C2))
            self.int_cap = max_matrix(self.int_cap, self.ints[-1])
            self.ress.append(self.ints[-1] - self.advs[-1] - self.divs[-1])
            self.res_cap = max_matrix(self.res_cap, self.ress[-1])

        return dates[1:]

    def monthly_average(self, dates):
        vars = [self.advs, self.divs, self.ints, self.ress]
        caps = [self.adv_cap, self.div_cap, self.int_cap, self.res_cap]
        months = []
        once = True

        for var, cap in zip(vars, caps):
            month = dates[0][4:6]
            monthly_avg = []
            buffer = []

            for i, date in enumerate(dates):
                c_month = date[4:6]
                if c_month == month:
                    buffer.append(var[i])
                else:
                    if once:
                        months.append(int(month))
                    month = c_month
                    monthly_avg.append(np.nanmean(np.array(buffer), axis=0))
                    buffer = []
                    buffer.append(var[i])

            var = monthly_avg
            cap = np.max(np.absolute(monthly_avg))
            once = False

        return months

    def plot_budgets(self, date1, date2, monthly=False, average=False):
        dates = self.get_budgets(date1, date2)

        if monthly:
            dates = self.monthly_average(dates)

        for i in range(int(np.floor(len(dates) / 2))):
            fig, axs = self.setup_plot()
            print(f'working on {dates[2 * i]} to {dates[2 * i + 1]}')
            once = True

            for a1, a2, var1, var2, cap, title in zip(axs[0], axs[1],
                                                      [self.advs[2 * i], self.divs[2 * i], self.ints[2 * i],
                                                       self.ress[2 * i]],
                                                      [self.advs[2 * i + 1], self.divs[2 * i + 1], self.ints[2 * i + 1],
                                                       self.ress[2 * i + 1]],
                                                      [self.adv_cap, self.div_cap, self.int_cap, self.res_cap],
                                                      ['advection', 'divergence', 'intensification', 'residual']):
                a1.pcolormesh(self.xc[1:-1], self.yc[1:-1], var1, transform=ccrs.NorthPolarStereo(-45), vmin=-cap,
                              vmax=cap,
                              cmap='coolwarm')
                a1.set_title(title, fontsize=20)
                if once:
                    if monthly:
                        d1, d2 = dates[2 * i], dates[2 * i + 1]
                    else:
                        d1, d2 = ds.string_time_to_datetime(dates[2 * i]), ds.string_time_to_datetime(dates[2 * i + 1])

                    a1.text(-0.01, 0.55, d1, va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor', transform=a1.transAxes, fontsize=20)
                    a2.text(-0.01, 0.55, d2, va='bottom', ha='center',
                            rotation='vertical', rotation_mode='anchor', transform=a2.transAxes, fontsize=20)
                    once = False

                im = a2.pcolormesh(self.xc[1:-1], self.yc[1:-1], var2, transform=ccrs.NorthPolarStereo(-45), vmin=-cap,
                                   vmax=cap,
                                   cmap='coolwarm')
                fig.colorbar(im, orientation='horizontal', ax=a2)

            # axs[0,-1].set_ylabel(ds.string_time_to_datetime(dates[2*i+1]), fontsize=20)

            plot.show_plot(fig, f'./plots/budgets/budgets/budgets_{dates[2 * i]}_{dates[2 * i + 1]}.png', False)

    def plot_average_budget(self, date1, date2):
        months = date_range(ds.string_time_to_datetime(date1), ds.string_time_to_datetime(date2),
                           freq='MS').month.tolist()
        years = date_range(ds.string_time_to_datetime(date1), ds.string_time_to_datetime(date2),
                           freq='MS').year.tolist()

        advs, divs, ints, ress = [], [], [], []
        for month, year in zip(months, years):
            u, v, siconc = self.get_monthly(month, year)
            try:
                _, _, siconcp1 = self.get_monthly(month+1, year)
                ints.append(self.intensification(siconc, siconcp1))
                advs.append(self.advection(u, v, siconc))
                divs.append(self.divergence(u, v, siconc))
                ress.append(ints[-1] - advs[-1] - divs[-1])
            except ValueError:
                print('Could not find month: ', month+1)

            try:
                _, _, siconcp1 = self.get_monthly(1, year+1)
                ints.append(self.intensification(siconc, siconcp1))
                advs.append(self.advection(u, v, siconc))
                divs.append(self.divergence(u, v, siconc))
                ress.append(ints[-1] - advs[-1] - divs[-1])
            except ValueError:
                print('Could not find J in year: ', year+1)

        a_cap = np.max(np.absolute(np.array(advs)))
        i_cap = np.max(np.absolute(np.array(ints)))
        d_cap = np.max(np.absolute(np.array(divs)))
        r_cap = np.max(np.absolute(np.array(ress)))

        print(a_cap, i_cap, d_cap, r_cap)

        self.nrows = 1
        for adv, div, int, res, month, year in zip(advs, divs, ints, ress, months[:-1], years[:-1]):
            fig, axs = self.setup_plot()
            axs[0].text(-0.01, 0.55, str(month), va='bottom', ha='center',
                        rotation='vertical', rotation_mode='anchor', transform=axs[0].transAxes, fontsize=20)
            for var, cap, title, ax in zip([adv, div, int, res],
                                           [a_cap, i_cap, d_cap, r_cap],
                                           ['advection', 'divergence', 'intensification', 'residual'], axs):
                im = ax.pcolormesh(self.xc[1:-1], self.yc[1:-1], var, transform=ccrs.NorthPolarStereo(-45),
                                   cmap='coolwarm', vmin=-cap, vmax=cap)
                ax.set_title(title, fontsize=20)
                fig.colorbar(im, orientation='horizontal', ax=ax)

            plot.show_plot(fig, f'./plots/budgets/budgets/budget_avg_{year}-{month}.png', False)


if __name__ == '__main__':
    # IceData().plot_siconc('20191201', '20200331')
    # IceData().plot_siconc('20191201', '20191210')

    IceData().plot_budgets('20191101', '20200430', True)
    # IceData().plot_budgets('20200101', '20200430', True)

    # IceData().plot_average_budget('20200201', '20200229')
    # IceData().plot_average_budget('20191101', '20200430')
    # arr = np.array([np.array([[1,2,3], [4,5,np.nan]]), np.array([[1,2,3], [4,5,8]])])
    # print(np.nanmean(arr, axis=0))

    pass

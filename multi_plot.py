# Module for the creation of multiple Plots in one Figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import case_information as ci
import data_science as ds
import plot
import leads
import numpy as np


class Multiplot:
    def __init__(self, date1, date2, extent=ci.barent_extent):
        self.fig_shape = (2, 6)
        self.dates = ds.time_delta(date1, date2)
        self.extent = extent
        self.lon, self.lat = leads.CoordinateGrid().lon, leads.CoordinateGrid().lat
        self.regr_lon, self.regr_lat = leads.Era5('msl').lon, leads.Era5('msl').lat

    def setup_plot(self):
        # create figure and base map
        fig, ax = plt.subplots(self.fig_shape[0], self.fig_shape[1],
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.coastlines(resolution='50m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def plot_leads_wind_div(self):
        im1, im2 = None, None
        divs = []
        cap = 0

        for date in self.dates:
            div = leads.Era5('wind_quiver').get_div(date)
            divs.append(div)
            plt.imshow(div)
            cap = max(cap, np.nanmax(div), abs(np.nanmin(div)))

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            div_span = divs[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]

            for a1, a2, date, div in zip(ax[0], ax[1], date_span, div_span):
                print(date)
                im1 = a1.pcolormesh(self.lon, self.lat, 100*leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, div, vmin=-cap, vmax=cap,
                                    transform=ccrs.PlateCarree(), cmap='bwr')

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar1.set_label('lead fraction in %', size=15)

            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            cbar2.set_label('wind divergence (pos) and convergence (neg) in 1/s', size=15)

            plot.show_plot(fig, f'./plots/leads_winddiv_{date_span[0]}_{date_span[-1]}.png', False)

    def plot_leads_windspeed(self):
        im1, im2 = None, None
        w_speeds = []
        cap = 0

        for date in self.dates:
            w_speed = leads.Era5('wind').get_variable(date)
            w_speeds.append(w_speed)
            plt.imshow(w_speed)
            cap = max(cap, np.nanmax(w_speed))

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            speed_span = w_speeds[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]

            for a1, a2, date, w_speed in zip(ax[0], ax[1], date_span, speed_span):
                print(date)
                # plot lead fraction
                im1 = a1.pcolormesh(self.lon, self.lat, 100 * leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                # plot second variable
                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, w_speed, vmin=0, vmax=cap,
                                    transform=ccrs.PlateCarree(), cmap='cividis')

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar1.set_label('lead fraction in %', size=15)

            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            cbar2.set_label('wind speed in m/s', size=15)

            plot.show_plot(fig, f'./plots/leads_windspeed_{date_span[0]}_{date_span[-1]}.png', False)

    def plot_leads_t2m(self):
        im1, im2 = None, None
        t2ms = []
        cap = 0
        mcap = 0

        for date in self.dates:
            t2m = leads.Era5('t2m').get_variable(date)
            t2ms.append(t2m)
            plt.imshow(t2m)
            cap = max(cap, np.nanmax(t2m))
            mcap = min(mcap, np.nanmin(t2m))

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            t2m_spann = t2ms[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]

            for a1, a2, date, t2m in zip(ax[0], ax[1], date_span, t2m_spann):
                print(date)
                # plot lead fraction
                im1 = a1.pcolormesh(self.lon, self.lat, 100 * leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                # plot second variable
                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, t2m, vmin=mcap, vmax=cap,
                                    transform=ccrs.PlateCarree(), cmap='bwr')

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar1.set_label('lead fraction in %', size=15)

            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            cbar2.set_label('temperature 2m above ground in °K', size=15)

            plot.show_plot(fig, f'./plots/leads_t2m_{date_span[0]}_{date_span[-1]}.png', False)

    def plot_leads_cyclonoccurence(self):
        im1, im2 = None, None
        cycs = []

        for date in self.dates:
            cyc = leads.Era5('cyclone_occurence').get_variable(date)
            cycs.append(cyc)
            plt.imshow(cyc)

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            cyc_spann = cycs[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]

            for a1, a2, date, cyc in zip(ax[0], ax[1], date_span, cyc_spann):
                print(date)
                # plot lead fraction
                im1 = a1.pcolormesh(self.lon, self.lat, 100 * leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                # plot second variable
                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, cyc, vmin=0, vmax=100,
                                    transform=ccrs.PlateCarree(), cmap='Greys', alpha=.4)
                cim = a2.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a2.clabel(cim, inline=True, fontsize=15, inline_spacing=10)

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar1.set_label('lead fraction in %', size=15)

            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            cbar2.set_label('cyclone occurence in %', size=15)

            plot.show_plot(fig, f'./plots/leads_cycloneocc_{date_span[0]}_{date_span[-1]}.png', False)

    def plot_leads_sic(self):
        im1, im2 = None, None
        sics = []

        for date in self.dates:
            sic = leads.Era5('siconc').get_variable(date)
            sics.append(sic)
            plt.imshow(sic)

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            sic_spann = sics[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]

            for a1, a2, date, sic in zip(ax[0], ax[1], date_span, sic_spann):
                print(date)
                # plot lead fraction
                im1 = a1.pcolormesh(self.lon, self.lat, 100 * leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                # plot second variable
                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, sic, vmin=0, vmax=100,
                                    transform=ccrs.PlateCarree(), cmap='Blues_r')

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar1.set_label('lead fraction in %', size=15)

            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            cbar2.set_label('sea ice concentration in %', size=15)

            plot.show_plot(fig, f'./plots/leads_siconc_{date_span[0]}_{date_span[-1]}.png', False)


if __name__ == '__main__':
    Multiplot('20200201', '20200229').plot_leads_sic()

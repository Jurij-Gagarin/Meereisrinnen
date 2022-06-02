# Module for the creation of multiple Plots in one Figure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import case_information as ci
import data_science as ds
import plot
import leads


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
        im1 = None

        for i in range(len(self.dates) % self.fig_shape[1] + 1):
            fig, ax = self.setup_plot()
            date_span = self.dates[i * self.fig_shape[1]: i * self.fig_shape[1] + self.fig_shape[1]]
            divs = []
            cap = 0

            for date in date_span:
                div = leads.Era5('wind_quiver').get_div(date)
                divs.append(div)
                cap = max(cap, div.max())

            for a1, a2, date, div in zip(ax[0], ax[1], date_span, divs):
                im1 = a1.pcolormesh(self.lon, self.lat, 100*leads.Lead(date).lead_data, transform=ccrs.PlateCarree(),
                                    cmap='cool')
                cim = a1.contour(self.regr_lon, self.regr_lat, leads.Era5('msl').get_variable(date),
                                 transform=ccrs.PlateCarree(), cmap='Oranges_r', levels=10)
                a1.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
                a1.set_title(ds.string_time_to_datetime(date), fontsize=15)

                im2 = a2.pcolormesh(self.regr_lon, self.regr_lat, div, vmin=0, vmax=cap,
                                    transform=ccrs.PlateCarree(), cmap='bwr')

            cbar1 = fig.colorbar(im1, ax=ax[0])
            cbar1.ax.tick_params(labelsize=15)
            cbar2 = fig.colorbar(im2, ax=ax[1])
            cbar2.ax.tick_params(labelsize=15)
            plt.show()


if __name__ == '__main__':
    Multiplot('20200215', '20200225').plot_leads_wind_div()

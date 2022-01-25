import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import data_science as ds
import numpy as np


def setup_plot(extent):
    # create figure and base map
    fig, ax = plt.subplots(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(-45))
    ax.gridlines()
    ax.set_global()
    ax.coastlines(resolution='50m')
    extent = extent if extent else (-180.0, 180.0, 68.5, 90)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def show_plot(fig, file_name, show):
    if show:
        plt.show()
    plt.savefig(file_name)
    plt.close(fig)


def regional_lead_plot(date, extent=None, file_name=None, show=False, msl=True):
    if not file_name:
        file_name = f'./plots/{date}.png'

    # setup data
    lead = leads.Lead(date)
    grid = leads.CoordinateGrid(lead)

    # setup plot
    fig, ax = setup_plot(extent)
    ax.set_title(f'Sea ice leads {lead.date[6:]}-{lead.date[4:6]}-{lead.date[:4]} in % per lattice-cell',
                 size=17)

    # plot lead data with color-bar
    im = ax.pcolormesh(grid.lon, grid.lat, 100*lead.lead_data, cmap='cool', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.land, cmap='twilight', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.cloud, cmap='Blues', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)

    # plot msl data with colorbar
    if msl:
        msl_plot(date, fig, ax, 'Oranges_r')

    # Show/Save the figure
    show_plot(fig, file_name, show)


def two_lead_diff_plot(date1, date2, extent=None, file_name=None, show=False, msl=True):
    if not file_name:
        file_name = f'./plots/diff{date1}-{date2}.png'

    # setup data
    lead1 = leads.Lead(date1)
    lead2 = leads.Lead(date2)
    grid = leads.CoordinateGrid(lead1)

    # setup plot
    fig, ax = setup_plot(extent)
    ax.set_title(f'Sea ice leads difference {lead1.date[6:]}-{lead1.date[4:6]}-{lead1.date[:4]}/'
                 f'{lead2.date[6:]}-{lead2.date[4:6]}-{lead2.date[:4]} in % per lattice- cell',
                 size=17)

    # plot data
    ds.two_lead_diff(lead1, lead2)
    im = ax.pcolormesh(grid.lon, grid.lat, 100*lead2.lead_data, cmap='bwr', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.land, cmap='Set2_r', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.cloud, cmap='twilight', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)

    # plot msl
    msl_plot(date2, fig, ax, 'summer')

    # Show/Save the figure
    show_plot(fig, file_name, show)


def msl_plot(date, fig, ax, cmap):
    data_set = leads.AirPressure()
    contours = ax.contour(data_set.lon, data_set.lat, data_set.get_msl(date), cmap=cmap,
                          transform=ccrs.PlateCarree())#, levels=np.linspace(90000, 110000, 30000))
    ax.clabel(contours, inline=True, fontsize=15, inline_spacing=10)
    #cbar = fig.colorbar(contours, ax=ax)
    #cbar.ax.tick_params(labelsize=17)


def regional_plot_month(extent=None):
    days = list(range(1, 29))
    dates = [f'202002{str(day).zfill(2)}' for day in days]
    for date in dates:
        regional_lead_plot(date, extent)
        print(date)


def lead_diff_month(extent=None):
    days = list(range(1, 28))
    dates = [f'202002{str(day).zfill(2)}' for day in days]

    for i in range(0, 28):
        two_lead_diff_plot(dates[i], dates[i+1], extent)
        print(i)


if __name__ == '__main__':
    #regional_lead_plot('20200217', show=True)
    #regional_plot_month()
    pass

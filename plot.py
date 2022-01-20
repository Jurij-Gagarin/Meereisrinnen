import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads


def regional_lead_plot(date, extent=None, file_name=None, show=False):
    if not file_name:
        file_name = f'./plots/{date}.png'

    # setup data
    grid = leads.CoordinateGrid()
    lead = leads.Lead(date)
    lead.clear_matrix()
    grid.clear_grid(lead.del_row, lead.del_col)

    # create figure and base map
    fig, ax = plt.subplots(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(-45))
    ax.gridlines()
    ax.set_global()
    ax.coastlines(resolution='50m')
    extent = extent if extent else (-180.0, 180.0, 61, 90)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # plot data with color-bar
    im = ax.pcolormesh(grid.lon, grid.lat, lead.lead_frac, cmap='cool', transform=ccrs.PlateCarree())
    im.cmap.set_over('#dddddd')
    im.cmap.set_under('#00394d')
    im.set_clim(0, 1)
    fig.colorbar(im, ax=ax)

    if show:
        plt.show()
    plt.savefig(file_name)
    plt.close(fig)


if __name__ == '__main__':
    pass

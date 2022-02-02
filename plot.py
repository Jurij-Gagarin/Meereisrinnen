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


def regional_lead_plot(date, extent=None, show=False, variable=None, plot_leads=True):
    file_name = date

    # setup data
    lead = leads.Lead(date)
    grid = leads.CoordinateGrid(lead)

    # setup plot
    fig, ax = setup_plot(extent)
    title = f'{lead.date[6:]}-{lead.date[4:6]}-{lead.date[:4]}'

    # plot lead data with color-bar
    if plot_leads:
        lead_plot(grid, lead, fig, ax)

    # plot variable data with colorbar
    if variable:
        if isinstance(variable, list):
            for v in variable:
                variable_plot(date, fig, ax, v)
                file_name = v + '_' + file_name
                title += ', ' + v
        else:
            variable_plot(date, fig, ax, variable)
            file_name = variable + '_' + file_name
            title += ', ' + variable

    # Show/Save the figure
    ax.set_title(title, size=17)
    file_name += '.png'
    file_name = './plots/' + file_name
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
    im = ax.pcolormesh(grid.lon, grid.lat, 100 * lead2.lead_data, cmap='bwr', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.land, cmap='Set2_r', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead2.cloud, cmap='twilight', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)

    # plot variable
    if msl:
        variable_plot(date2, fig, ax, 'summer')

    # Show/Save the figure
    show_plot(fig, file_name, show)


def lead_plot(grid, lead, fig, ax):
    im = ax.pcolormesh(grid.lon, grid.lat, 100 * lead.lead_data, cmap='cool', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.land, cmap='twilight', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.cloud, cmap='Blues', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)


def variable_plot(date, fig, ax, variable):
    # Plots contour lines of mean sea level air pressure.
    contour_plot = {'msl': True, 'u10': False, 't2m': False, 'cyclone_occurence': False}
    cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'Greys_r', 'u10': 'twilight_shifted', 't2m': 'coolwarm'}
    alpha_dict = {'msl': 1, 'cyclone_occurence': .1, 'u10': 1, 't2m': 1}
    data_set = leads.Era5(variable)
    # data_set = leads.Era5Regrid(leads.Lead(date), variable)

    if contour_plot[variable]:
        contours = ax.contour(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=cmap_dict[variable],
                              alpha=alpha_dict[variable], transform=ccrs.PlateCarree(), levels=10)
        ax.clabel(contours, inline=True, fontsize=15, inline_spacing=10)
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=cmap_dict[variable],
                           alpha=alpha_dict[variable], transform=ccrs.PlateCarree())
        # im.set_clim(-25, 25)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=17)


'''
def era5_plot(date, fig, ax, cmap, extent):
    # Era5 Dataset is typically not used for plotting. This function might be removed in the near future
    data_set = leads.Era5Regrid(leads.Lead(date), 'variable')
    contours = ax.contour(data_set.lon, data_set.lat, data_set.get_msl(date), cmap=cmap,
                          transform=ccrs.PlateCarree(), levels=15)
    ax.clabel(contours, inline=True, fontsize=15, inline_spacing=10)
'''


def plots_for_case(case, extent=None, var=None, plot_lead=True, diff=False):
    for i, date in enumerate(case):
        print(f'Working on plots for date:{date}')
        regional_lead_plot(date, extent=extent, variable=var, plot_leads=plot_lead)

        if diff:
            try:
                two_lead_diff_plot(case[i], case[i + 1], extent=extent)
            except IndexError:
                pass


if __name__ == '__main__':
    case1 = ['20200216', '20200217', '20200218', '20200219', '20200220', '20200221', '20200222']
    extent1 = [-70, 100, 65, 90]
    case2 = ['20200114', '20200115', '20200116', '20200117', '20200118', '20200119', '20200120']
    extent2 = None
    case3 = ['20200128', '20200129', '20200130', '20200131', '20200201', '20200202', '20200203']
    extent3 = None
    case4 = ['20200308', '20200309', '20200310', '20200311', '20200312', '20200313', '20200314', '20200315', '20200316']
    extent4 = None
    path = './plots/case1'

    # plots_for_case(case1, extent1, ['cyclone_occurence', 'msl'], plot_lead=True)
    plots_for_case(case2, extent2, ['cyclone_occurence', 'msl'], plot_lead=True)
    plots_for_case(case3, extent3, ['cyclone_occurence', 'msl'], plot_lead=True)
    plots_for_case(case4, extent4, ['cyclone_occurence', 'msl'], plot_lead=True)
    #regional_lead_plot('20200221', show=False, variable=['cyclone_occurence', 'msl'], plot_leads=True)

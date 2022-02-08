import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import data_science as ds
import numpy as np
import scipy.optimize as opt


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
    grid = leads.CoordinateGrid()

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
    # lon, lat, area = ds.select_area(grid, lead, np.ones(grid.lat.shape))
    # ax.pcolormesh(lon, lat, area, transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)


def variable_plot(date, fig, ax, variable):
    # Plots contour lines of mean sea level air pressure.
    contour_plot = {'msl': True, 'wind': False, 't2m': False, 'cyclone_occurence': False}
    cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'Greys_r', 'wind': 'cividis', 't2m': 'coolwarm'}
    alpha_dict = {'msl': 1, 'cyclone_occurence': .25, 'wind': 1, 't2m': 1}
    data_set = leads.Era5(variable)
    # data_set = leads.Era5Regrid(leads.Lead(date), variable)

    if contour_plot[variable]:
        contours = ax.contour(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=cmap_dict[variable],
                              alpha=alpha_dict[variable], transform=ccrs.PlateCarree(), levels=10)
        ax.clabel(contours, inline=True, fontsize=15, inline_spacing=10)
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=cmap_dict[variable],
                           alpha=alpha_dict[variable], transform=ccrs.PlateCarree())
        # im.set_clim(0, 25)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=17)


def matrix_plot(matrix, extent=None):
    fig, ax = setup_plot(extent)
    grid = leads.CoordinateGrid()
    # im = ax.pcolormesh(grid.lon, grid.lat, matrix, cmap='RdYlGn', transform=ccrs.PlateCarree())
    im = ax.pcolormesh(grid.lon, grid.lat, matrix, cmap='bwr', transform=ccrs.PlateCarree())
    im.set_clim(-.6, .6)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)
    plt.show()


def quantify(lead_diff, cyc_diff):
    lead_diff.flatten()
    mask = np.isnan(lead_diff)
    cyc_diff.flatten()
    lead_diff = lead_diff[~mask]
    cyc_diff = cyc_diff[~mask]
    plt.scatter(cyc_diff, lead_diff, s=1)
    fit = lambda x, a: a * x
    pars, cov = opt.curve_fit(fit, cyc_diff, lead_diff)
    plt.plot(cyc_diff, [fit(x, pars[0]) for x in cyc_diff])
    plt.show()


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
    extent1 = (80, -20, 90, 70)
    case2 = ['20200114', '20200115', '20200116', '20200117', '20200118', '20200119', '20200120']
    extent2 = [80, -20, 90, 70]
    case3 = ['20200128', '20200129', '20200130', '20200131', '20200201', '20200202', '20200203']
    extent3 = [70, -10, 90, 70]
    case4 = ['20200308', '20200309', '20200310', '20200311', '20200312', '20200313', '20200314', '20200315', '20200316']
    extent4 = [70, -10, 90, 70]
    path = './plots/case1'

    # regional_lead_plot('20200221', show=True, variable=None, plot_leads=True)

    extent = [-10, 70, 70, 90]
    # matrix_plot(ds.cyclone_trace('20200308', '20200316'), extent)
    # matrix_plot(ds.cyclone_trace('20200227', '20200307'), extent)
    # plots_for_case(case1, extent1, ['cyclone_occurence', 'msl'], True)

    # lead_diff = ds.lead_average('20200114', '20200117') - ds.lead_average('20200110', '20200113') #case2
    # cyc_diff = ds.cyclone_trace('20200114', '20200117') - ds.cyclone_trace('20200110', '20200113') #case2
    # lead_diff = ds.lead_average('20200218', '20200220') - ds.lead_average('20200212', '20200216') #case1
    # cyc_diff = ds.cyclone_trace('20200218', '20200220', extent) - ds.cyclone_trace('20200212', '20200216', extent) #case1
    # lead_diff = ds.lead_average('20200131', '20200203') - ds.lead_average('20200126', '20200130') #case3
    # cyc_diff = ds.cyclone_trace('20200131', '20200203') - ds.cyclone_trace('20200126', '20200130') #case3
    # lead_diff = ds.lead_average('20200308', '20200316') - ds.lead_average('20200227', '20200307') #case4
    # cyc_diff = ds.cyclone_trace('20200308', '20200316') - ds.cyclone_trace('20200317', '20200325') #case4

    lead_test = ds.lead_average('20200217', '20200221', extent1) - .5*(ds.lead_average('20200201', '20200216', extent1) + ds.lead_average('20200222', '20200229', extent1))
    cyc_test = ds.cyclone_trace('20200217', '20200221') - .5*(ds.cyclone_trace('20200201', '20200216') + ds.cyclone_trace('20200222', '20200229'))

    matrix_plot(lead_test, None)
    quantify(lead_test, cyc_test)

    # matrix_plot(ds.lead_average('20200210', '20200215'), extent)
    # matrix_plot(ds.lead_average('20200217', '20200221'), extent)
    # plots_for_case(case1, extent1, ['wind', 'msl'],  plot_lead=False)
    # plots_for_case(case1, extent1, ['cyclone_occurence', 'msl'],  plot_lead=False)

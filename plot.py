import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import data_science as ds
import numpy as np
import calendar
from dateutil.rrule import rrule, MONTHLY
from scipy.interpolate import CubicSpline


class VarOptions:
    def __init__(self, var):
        self.var = var
        # Dictionaries that assign colors, cmaps, ... to certain variables
        contour_plot = {'msl': True, 'wind': False, 't2m': False, 'cyclone_occurence': False, 'leads': False,
                        'siconc': False}
        cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'Greys_r', 'wind': 'cividis', 't2m': 'coolwarm',
                     'leads': 'inferno', 'siconc': 'Blues'}
        alpha_dict = {'msl': 1, 'cyclone_occurence': .25, 'wind': 1, 't2m': 1, 'leads': 1, 'siconc': 1}
        color_dict = {'msl': 'red', 'leads': 'blue', 'wind': 'orange', 'cyclone_occurence': 'green', 't2m': 'purple',
                      'siconc': 'turquoise'}
        unit_dict = {'msl': 'hPa', 'leads': '%', 'cyclone_occurence': '%', 'wind': 'm/s', 't2m': '°K', 'siconc': '%'}
        name_dict = {'cyclone_occurence': 'cyclone frequency', 'leads': 'daily new lead fraction', 'wind': 'wind speed',
                     't2m': 'two meter temperature', 'msl': 'mean sea level pressure', 'siconc': 'sea ice concentration'
                     }

        self.contour = contour_plot[self.var]
        self.cmap = cmap_dict[self.var]
        self.alpha = alpha_dict[self.var]
        self.color = color_dict[self.var]
        self.unit = unit_dict[self.var]
        self.name = name_dict[self.var]

    def label(self, extra_label=''):
        return f'{self.name} in {self.unit}' + extra_label



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
    # Decides if the figure is shown or saved.
    if show:
        plt.show()
    plt.savefig(file_name)
    plt.close(fig)


def regional_lead_plot(date, extent=None, show=False, variable=None, plot_leads=True):
    # Creates a regional plot of your data.
    # Supported data: Leads, wind, cyclone_occurence, msl and possibly t2m in the future.
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
                title += ', ' + VarOptions(v).label()
        else:
            variable_plot(date, fig, ax, variable)
            file_name = variable + '_' + file_name
            title += ', ' + VarOptions(variable).label()

    # Show/Save the figure
    ax.set_title(title, size=17)
    file_name += '.png'
    file_name = './plots/' + file_name
    show_plot(fig, file_name, show)


def two_lead_diff_plot(date1, date2, extent=None, file_name=None, show=False, msl=True):
    # This method is used to plot the difference between day to day Leads. It's currently not supported.
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
    # Plots lead fraction
    im = ax.pcolormesh(grid.lon, grid.lat, 100 * lead.lead_data, cmap='cool', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.land, cmap='twilight', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.cloud, cmap='Blues', transform=ccrs.PlateCarree())
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)


def variable_plot(date, fig, ax, variable):
    # Plots data that is stored in a Era5 grid.
    Var = VarOptions(variable)
    # data_set = leads.Era5(variable)
    data_set = leads.Era5Regrid(leads.Lead(date), variable) # With this Era5Regrid-class is tested

    if Var.contour:
        contours = ax.contour(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=Var.cmap,
                              alpha=Var.alpha, transform=ccrs.PlateCarree(), levels=10)
        ax.clabel(contours, inline=True, fontsize=15, inline_spacing=10)
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=Var.cmap,
                           alpha=Var.alpha, transform=ccrs.PlateCarree())
        # im.set_clim(0, 25)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=17)


def matrix_plot(date1, date2, variable, cmap='RdYlGn', clim=(None, None), extent=None, show=False):
    if variable == 'leads':
        matrix = ds.lead_average(date1, date2, extent)
    else:
        matrix = ds.variable_average(date1, date2, extent, variable)

    fig, ax = setup_plot(extent)
    grid = leads.CoordinateGrid()
    im = ax.pcolormesh(grid.lon, grid.lat, matrix, cmap=cmap, transform=ccrs.PlateCarree())
    im.set_clim(clim[0], clim[1])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)
    ax.set_title(f'{variable}-{date1}-{date2}', fontsize=17)
    show_plot(fig, f'./plots/{variable}-{date1}-{date2}.png', show)


def variable_daily_avg(date1, date2, extent, variable):
    # This returns a narray that contains the daily average values of your variable data.
    dates = ds.time_delta(date1, date2)
    var_sum = np.zeros(len(dates))
    for i, date in enumerate(dates):
        print(date)
        if variable == 'leads':
            var = 100*ds.lead_average(date, date, extent)
        else:
            var = ds.variable_average(date, date, extent, variable)
        var_sum[i] = np.nanmean(var)
    return var_sum


def variable_avg_sum_daily(date1, date2, extent, variables):
    # This Plot shows you how the daily averages of two variables correlate with each other.
    # variables need's to be an iterable that contains the strings, that corresponds to your variable
    var1 = variable_daily_avg(date1, date2, extent, variables[0])
    var2 = variable_daily_avg(date1, date2, extent, variables[1])

    fig, ax = plt.subplots()
    ax.scatter(var1, var2)
    ax.set_xlabel(f'{variables[0]}')
    ax.set_ylabel(f'{variables[1]}')
    ax.set_title(f'Average {variables[0]} against {variables[1]} daily.')
    plt.show()


def variables_against_time(date1, date2, extent, var1, var2, spline=False):
    # This shows you how two variables change with respect to time.
    dates = ds.string_time_to_datetime(ds.time_delta(date1, date2))
    fig, ax = plt.subplots()
    ax_twin = ax.twinx()
    Var1, Var2 = VarOptions(var1), VarOptions(var2)
    title = f'Changes in {Var1.name} and {Var2.name} over time within {extent}.'

    for a, v, Var in zip([ax, ax_twin], [var1, var2], [Var1, Var2]):
        y = variable_daily_avg(date1, date2, extent, v)
        x = list(range(len(dates)))

        if spline:
            x = list(range(len(dates)))
            a.scatter(x, y, c=Var.color)
            f = CubicSpline(x, y, bc_type='natural')
            x_new = np.linspace(0, len(dates), 1000)
            y_new = f(x_new)
            a.plot(x_new, y_new, c=Var.color, linestyle='--')
        else:
            a.plot(dates, y, c=Var.color, linestyle='--')

        a.set_ylabel(Var.label(), fontsize=15)
        a.yaxis.label.set_color(Var.color)
        a.tick_params(axis='y', colors=Var.color, labelsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.set_title(title, fontsize=15)
    plt.show()


def plot_lead_cyclone_sum_monthly(date1, date2, extent, variable):
    # get all months between dates
    start_date = datetime.date(int(date1[:4]), int(date1[4:6]), int(date1[6:]))
    end_date = datetime.date(int(date2[:4]), int(date2[4:6]), int(date2[6:]))
    dates = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=start_date, until=end_date)]
    lead_avg, variable_avg = [], []

    for date in dates:
        month_r = calendar.monthrange(date[0], date[1])[1]
        year_month = str(date[0]) + str(date[1]).zfill(2)
        print(year_month)
        avg = ds.lead_average(year_month + '01', year_month + str(month_r), extent)
        var_avg = ds.variable_average(year_month + '01', year_month + str(month_r), extent, variable)
        avg = avg[~np.isnan(avg)]
        var_avg = var_avg[~np.isnan(var_avg)]
        lead_avg.append(np.sum(avg) / len(avg))
        variable_avg.append(np.sum(var_avg) / len(var_avg))

    dates = [str(date[0]) + '-' + str(date[1]).zfill(2) for date in dates]
    plt.title('Monthly lead fraction of the entire Arctic')
    plt.xlabel('Month')
    plt.ylabel('Normalized average lead fraction per month')
    plt.scatter(dates, lead_avg, label='leads')
    plt.scatter(dates, variable_avg, label=f'{variable}')
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
    # TODO: sinnvollen extent für die Barentsee definieren
    # TODO: Methode zum automatischen abspeichern der Plots erstellen
    # TODO: Plots erstellen für den gesammten Zeitraum, für verschiedene Extents, speziell für unsere Zyklonenevents
    case1 = ['20200216', '20200217', '20200218', '20200219', '20200220', '20200221', '20200222']
    extent1 = (60, 0, 80, 75)
    case2 = ['20200114', '20200115', '20200116', '20200117', '20200118', '20200119', '20200120']
    extent2 = [80, 0, 80, 75]
    case3 = ['20200128', '20200129', '20200130', '20200131', '20200201', '20200202', '20200203']
    extent3 = [70, -10, 90, 70]
    case4 = ['20200308', '20200309', '20200310', '20200311', '20200312', '20200313', '20200314', '20200315', '20200316']
    extent4 = [65, 0, 80, 75]

    #regional_lead_plot('20200221', show=True, variable='siconc', plot_leads=True)

    extent = [65, 0, 80, 71]
    s_extent = [180, -180, 90, 85]
    no_extent = [180, -180, 90, 60]
    #variable_avg_sum_daily('20200101', '20200331', no_extent, ('msl', 'cyclone_occurence'))
    variables_against_time('20200210', '20200229', extent, 'leads', 'cyclone_occurence')

    #matrix_plot(ds.lead_average('20200112', '20200118', no_extent), extent=no_extent)
    #matrix_plot('20200316', '20200322', 'leads', cmap='inferno', extent=no_extent, show=False)
    #plot_lead_cyclone_sum_monthly('20191101', '20200430', no_extent, 'cyclone_occurence')

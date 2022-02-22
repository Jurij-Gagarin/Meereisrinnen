import datetime
import case_information as ci
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import data_science as ds
import numpy as np
import calendar
from dateutil.rrule import rrule, MONTHLY
from scipy.interpolate import CubicSpline
import helpful_functions as hf
from skimage.transform import resize


class VarOptions:
    def __init__(self, var):
        self.var = var
        # Dictionaries that assign colors, cmaps, ... to certain variables
        contour_plot = {'msl': True, 'wind': False, 't2m': False, 'cyclone_occurence': False, 'leads': False,
                        'siconc': False, 'wind_quiver': False}
        cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'gray', 'wind': 'cividis', 't2m': 'coolwarm',
                     'leads': 'cool', 'siconc': 'Blues', 'wind_quiver': 'inferno'}
        alpha_dict = {'msl': 1, 'cyclone_occurence': .15, 'wind': 1, 't2m': 1, 'leads': 1, 'siconc': 1, 'wind_quiver': 1
                      }
        color_dict = {'msl': 'red', 'leads': 'blue', 'wind': 'orange', 'cyclone_occurence': 'green', 't2m': 'purple',
                      'siconc': 'turquoise', 'wind_quiver': None}
        unit_dict = {'msl': 'hPa', 'leads': '%', 'cyclone_occurence': '%', 'wind': 'm/s', 't2m': '°K', 'siconc': '%',
                     'wind_quiver': 'm/s'}
        name_dict = {'cyclone_occurence': 'cyclone frequency', 'leads': 'daily new lead fraction', 'wind': 'wind speed',
                     't2m': 'two meter temperature', 'msl': 'mean sea level pressure', 'siconc': 'sea ice concentration',
                     'wind_quiver': 'windspeed'}
        quiver_dict = {'msl': False, 'wind': False, 't2m': False, 'cyclone_occurence': False, 'leads': False,
                       'siconc': False, 'wind_quiver': True}

        self.contour = contour_plot[self.var]
        self.cmap = cmap_dict[self.var]
        self.alpha = alpha_dict[self.var]
        self.color = color_dict[self.var]
        self.unit = unit_dict[self.var]
        self.name = name_dict[self.var]
        self.quiver = quiver_dict[self.var]

    def label(self, extra_label=''):
        return f'{self.name} in {self.unit}' + extra_label


class RegionalPlot:
    # Note that n * 10 can bee plotted perfectly fine
    def __init__(self, date1, date2, variable, extent=ci.arctic_extent, fig_shape=(2, 5), show=False, show_cbar=True):
        self.fig_shape = fig_shape
        self.extent = extent
        self.dates = ds.time_delta(date1, date2)
        self.show = show
        self.variable = variable
        self.plot_leads = False
        self.show_cbar = show_cbar
        images = []

        for var in self.variable:
            if var == 'leads':
                self.variable.remove('leads')
                self.plot_leads = True

        # first we need to set up the plot
        self.full = int(np.ceil(len(self.dates) / (fig_shape[0] * fig_shape[1])))  # number of figures necessary

        for i in range(self.full):
            fig, ax = self.setup_plot(i)

            for j, a in enumerate(ax.flatten()):
                date = self.dates[j + self.fig_shape[0] * self.fig_shape[1] * i]
                print(date)
                images = self.regional_var_plot(fig, a, date)

            if images:
                print(images)
                for im in images:
                    print(im)
                    if im:
                        cbar = fig.colorbar(im, ax=ax.ravel().toList)
                        cbar.ax.tick_params(labelsize=20)

            show_plot(fig, f'./plots/{self.dates[self.fig_shape[0] * self.fig_shape[1] * i]}_'
                           f'{self.dates[self.fig_shape[0] * self.fig_shape[1] * (i + 1) - 1]}.png', self.show)

    def setup_plot(self, base):
        # create figure and base map
        fig, ax = plt.subplots(self.fig_shape[0], self.fig_shape[1],
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.set_title(ds.string_time_to_datetime(self.dates[i + self.fig_shape[0] * self.fig_shape[1] * base]),
                        fontsize=20)
            a.coastlines(resolution='110m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def regional_var_plot(self, fig, ax, date):
        # setup data
        lead = leads.Lead(date)
        grid = leads.CoordinateGrid()
        im = []

        # plot lead data
        if self.plot_leads:
            im.append(lead_plot(grid, lead, fig, ax, False))

        # plot variable data
        if self.variable:
            if isinstance(self.variable, list):
                for v in self.variable:
                    im.append(variable_plot(date, fig, ax, v, False))
                    variable_plot(date, fig, ax, v, False)
            else:
                im.append(variable_plot(date, fig, ax, self.variable, False))

        return im  # returns an Image for the colorbar


def setup_plot(extent):
    # create figure and base map
    fig, ax = plt.subplots()
    fig.set_size_inches(32, 18)
    ax = plt.axes(projection=ccrs.NorthPolarStereo(-45))
    ax.gridlines()
    ax.set_global()
    ax.coastlines(resolution='50m')
    extent = extent if extent else ci.arctic_extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return fig, ax


def show_plot(fig, file_name, show):
    # Decides if the figure is shown or saved.
    if show:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)


def regional_var_plot(date, extent=None, show=False, variable=None, plot_leads=True, show_cbar=False):
    # Creates a regional plot of your data.
    # Supported data: Leads, wind, cyclone_occurence, msl and t2m.
    file_name = date

    # setup data
    lead = leads.Lead(date)
    grid = leads.CoordinateGrid()

    # setup plot
    fig, ax = setup_plot(extent)
    title = f'{lead.date[6:]}-{lead.date[4:6]}-{lead.date[:4]}'

    # plot lead data with color-bar
    if plot_leads:
        lead_plot(grid, lead, fig, ax, show_cbar)

    # plot variable data with color-bar
    if variable:
        if isinstance(variable, list):
            for v in variable:
                variable_plot(date, fig, ax, v, show_cbar)
                file_name = v + '_' + file_name
                title += ', ' + VarOptions(v).label()
        else:
            variable_plot(date, fig, ax, variable, show_cbar)
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
    grid = leads.CoordinateGrid()

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


def lead_plot(grid, lead, fig, ax, show_cbar):
    # Plots lead fraction
    im = ax.pcolormesh(grid.lon, grid.lat, 100 * lead.lead_data, cmap='cool', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.water, cmap='coolwarm', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.land, cmap='twilight', transform=ccrs.PlateCarree())
    ax.pcolormesh(grid.lon, grid.lat, lead.cloud, cmap='Blues', transform=ccrs.PlateCarree())
    if show_cbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=20)
    return im


def variable_plot(date, fig, ax, variable, show_cbar):
    # Plots data that is stored in a Era5 grid.
    Var = VarOptions(variable)
    im = None
    if variable == 'wind_quiver':
        data_set = leads.Era5Regrid(leads.Lead(date), 'wind_quiver') # With this Era5Regrid-class is tested
    else:
        data_set = leads.Era5(variable)

    if Var.contour:
        im = ax.contour(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=Var.cmap,
                              alpha=Var.alpha, transform=ccrs.PlateCarree(), levels=10)
        ax.clabel(im, inline=True, fontsize=15, inline_spacing=10)
    elif Var.quiver:
        lat_dir, lon_dir = data_set.get_quiver(date)
        dim = (50, 50)
        lon, lat = resize(data_set.lon, dim), resize(data_set.lat, dim)
        v10, u10 = resize(lon_dir, dim), resize(lat_dir, dim)
        im = ax.quiver(lon, lat, v10, u10, np.sqrt(v10**2+u10**2), transform=ccrs.PlateCarree(), cmap='plasma',
                       width=.005, pivot='mid')  # scale=40, scale_units='inches'
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date),alpha=Var.alpha, cmap=Var.cmap,
                           transform=ccrs.PlateCarree())
        # im.set_clim(0, 25)
        if show_cbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=20)
    if im:
        return im


def matrix_plot(date1, date2, variable, clim=(None, None), extent=None, show=False):
    Var = VarOptions(variable)
    if variable == 'leads':
        matrix = ds.lead_average(date1, date2, extent)
    else:
        matrix = ds.variable_average(date1, date2, extent, variable)

    fig, ax = setup_plot(extent)
    grid = leads.CoordinateGrid()
    im = ax.pcolormesh(grid.lon, grid.lat, matrix, cmap=Var.cmap, transform=ccrs.PlateCarree())
    im.set_clim(clim[0], clim[1])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=17)
    ax.set_title(f'{Var.label()} avg from {ds.string_time_to_datetime(date1)} to '
                 f'{ds.string_time_to_datetime(date2)}', fontsize=17)
    show_plot(fig, f'./plots/{variable}-{date1}-{date2}.png', show)


def variable_avg_sum_daily(date1, date2, extent, variables):
    # This Plot shows you how the daily averages of two variables correlate with each other.
    # variables need's to be an iterable that contains the strings, that corresponds to your variable
    var1 = ds.variable_daily_avg(date1, date2, extent, variables[0])
    var2 = ds.variable_daily_avg(date1, date2, extent, variables[1])

    fig, ax = plt.subplots()
    ax.scatter(var1, var2)
    ax.set_xlabel(f'{variables[0]}')
    ax.set_ylabel(f'{variables[1]}')
    ax.set_title(f'average {variables[0]} against {variables[1]} daily.')
    plt.show()


def variables_against_time(date1, date2, extent, var1, var2, spline=False, show=False):
    # This shows you how two variables change with respect to time.
    dates = ds.string_time_to_datetime(ds.time_delta(date1, date2))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax_twin = ax.twinx()
    Var1, Var2 = VarOptions(var1), VarOptions(var2)
    title = f'Changes in {Var1.name} and {Var2.name} in the {ci.extent_dict[extent]}.'
    y, i = [], 0

    for a, v, Var in zip([ax, ax_twin], [var1, var2], [Var1, Var2]):
        y.append(ds.variable_daily_avg(date1, date2, extent, v))
        y[i] = y[i] / np.max(y[i])

        if spline:
            x = list(range(len(dates)))
            a.scatter(x, y[i], c=Var.color)
            f = CubicSpline(x, y[i], bc_type='natural')
            x_new = np.linspace(0, len(dates), 1000)
            y_new = f(x_new)
            a.plot(x_new, y_new, c=Var.color, linestyle='--')
        else:
            a.plot(dates, y[i], c=Var.color, linestyle='--')

        a.set_ylabel(f'{Var.name} normalized', fontsize=15)
        a.yaxis.label.set_color(Var.color)
        a.tick_params(axis='y', colors=Var.color, labelsize=15)
        i += 1

    ax.tick_params(axis='x', labelsize=15)
    title += f' R = {hf.round_sig(np.corrcoef(y[0], y[1])[0, 1])}'
    ax.set_title(title, fontsize=15)

    show_plot(fig, f'./plots/{var1}_{var2}_{ds.string_time_to_datetime(date1)}_'
                   f'{ds.string_time_to_datetime(date2)}_{ci.extent_dict[extent]}.png', show)


def plot_lead_from_vars(date1, date2, extent, var1, var2):
    par, cov, fit, v1, v2, l = ds.lead_from_vars(date1, date2, extent, var1, var2)
    dates = ds.string_time_to_datetime(ds.time_delta(date1, date2))

    plt.plot(dates, l, label='leads')
    fit_data = [fit(x, par[0], par[1]) for x in zip(v1, v2)]
    plt.plot(dates, fit_data, label='a*cyc+b*sic')
    plt.plot(dates, v1, label=var1)
    plt.plot(dates, v2, label=var2)

    print(par)
    plt.legend()
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
        regional_var_plot(date, extent=extent, variable=var, plot_leads=plot_lead)

        if diff:
            try:
                two_lead_diff_plot(case[i], case[i + 1], extent=extent)
            except IndexError:
                pass


if __name__ == '__main__':
    #regional_var_plot('20200223', show=True, variable=['msl', 'wind_quiver'], plot_leads=True,
                      #extent=ci.extent1, show_cbar=True)

    RegionalPlot('20200101', '20200111', ['leads'], extent=ci.extent1, show=True)

    # matrix_plot('20200320', '20200325', 'leads', extent=ci.s_extent, show=True)
    # plot_lead_cyclone_sum_monthly('20191101', '20200430', no_extent, 'cyclone_occurence')
    # plot_lead_from_vars('20200101', '20200131', ci.arctic_extent, 'cyclone_occurence', 'siconc')

    '''
    for d in ci.Mon:
        variables_against_time(d[0], d[1], ci.arctic_extent, 'leads', 'siconc')
    pass
    '''
    # variables_against_time('20200120', '20200210', ci.arctic_extent, 'leads', 'cyclone_occurence')
    # variables_against_time('20200120', '20200210', ci.arctic_extent, 'leads', 'siconc')

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
import dcor


class VarOptions:
    def __init__(self, var):
        self.var = var
        # Dictionaries that assign colors, cmaps, ... to certain variables
        contour_plot = {'msl': True, 'wind': False, 't2m': False, 'cyclone_occurence': False, 'leads': False,
                        'siconc': False, 'wind_quiver': False, 'siconc_diff': False, 'wind_diff': False}
        cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'gray', 'wind': 'cividis', 't2m': 'coolwarm',
                     'leads': 'cool', 'siconc': 'Blues', 'wind_quiver': 'summer', 'siconc_diff': 'bwr',
                     'wind_diff':'bwr'}
        alpha_dict = {'msl': 1, 'cyclone_occurence': .15, 'wind': 1, 't2m': 1, 'leads': 1, 'siconc': 1, 'wind_quiver': 1
                      , 'siconc_diff': 1, 'wind_diff': 1}
        color_dict = {'msl': 'red', 'leads': 'blue', 'wind': 'orange', 'cyclone_occurence': 'green', 't2m': 'purple',
                      'siconc': 'turquoise', 'wind_quiver': None, 'siconc_diff': None, 'wind_diff': None}
        unit_dict = {'msl': 'hPa', 'leads': '%', 'cyclone_occurence': '%', 'wind': 'm/s', 't2m': '°K', 'siconc': '%',
                     'wind_quiver': 'm/s', 'siconc_diff': '%', 'wind_diff': None}
        name_dict = {'cyclone_occurence': 'cyclone frequency', 'leads': 'daily new lead fraction', 'wind': 'wind speed',
                     't2m': 'two meter temperature', 'msl': 'mean sea level pressure', 'siconc': 'sea ice concentration'
                     , 'wind_quiver': 'windspeed', 'siconc_diff': 'sea ice concentration difference'
                     , 'wind_diff': 'wind difference'}
        quiver_dict = {'msl': False, 'wind': False, 't2m': False, 'cyclone_occurence': False, 'leads': False,
                       'siconc': False, 'wind_quiver': True, 'siconc_diff': False, 'wind_diff': False}

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
    def __init__(self, date1, date2, variable, extent=ci.arctic_extent, fig_shape=(2, 6), show=False, show_cbar=True):
        self.fig_shape = fig_shape
        self.extent = extent
        self.dates = ds.time_delta(date1, date2)
        self.show = show
        self.variable = variable
        self.plot_leads = False
        self.show_cbar = show_cbar
        self.split_figure = False

        for var in self.variable:
            if var == 'leads':
                self.variable.remove('leads')
                self.plot_leads = True
            if var == 'siconc' or var == 'siconc_diff' or var == 'wind' or var == 'wind_diff':
                self.variable.remove(var)
                self.split_figure = True

        # first we need to set up the plot
        self.full = int(np.ceil(len(self.dates) / (fig_shape[0] * fig_shape[1])))  # number of figures necessary

        for i in range(self.full):
            fig, ax = self.setup_plot()

            if self.split_figure:
                self.plot_split(fig, ax, i, self.show_cbar)
            else:
                self.plot_nonsplit(fig, ax, i, self.show_cbar)

            show_plot(fig, f'./plots/{self.dates[self.fig_shape[0] * self.fig_shape[1] * i]}_'
                           f'{self.dates[self.fig_shape[0] * self.fig_shape[1] * (i + 1) - 1]}.png', self.show)

    def setup_plot(self):
        # create figure and base map
        fig, ax = plt.subplots(self.fig_shape[0], self.fig_shape[1],
                               subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, constrained_layout=True)
        fig.set_size_inches(32, 18)
        for i, a in enumerate(ax.flatten()):
            a.coastlines(resolution='110m')
            a.set_extent(self.extent, crs=ccrs.PlateCarree())
        return fig, ax

    def regional_var_plot(self, fig, ax, date, variable, plot_leads):
        # setup data
        lead = leads.Lead(date)
        grid = leads.CoordinateGrid()
        im = []

        # plot lead data
        if plot_leads:
            im.append(lead_plot(grid, lead, fig, ax, False))

        # plot variable data
        if variable:
            if isinstance(variable, list):
                for v in variable:
                    if v == 'siconc_diff' or 'wind':
                        im.append(variable_plot(date, fig, ax, v, False))
                    else:
                        im.append(variable_plot(date, fig, ax, v, False))
            else:
                im.append(variable_plot(date, fig, ax, variable, False))

        return im  # returns an Image for the colorbar

    def plot_split(self, fig, ax, i, show_cbar):
        images1, images2 = [], []
        for j, (a1, a2) in enumerate(zip(ax[0], ax[1])):
            date = self.dates[j + self.fig_shape[0] * self.fig_shape[1] * i]
            a1.set_title(ds.string_time_to_datetime(date), fontsize=20)
            a1.set_title(ds.string_time_to_datetime(date), fontsize=20)
            print(date)
            images2 = self.regional_var_plot(fig, a2, date, ['wind_diff'], plot_leads=False)
            images1 = self.regional_var_plot(fig, a1, date, self.variable, plot_leads=True)

        if show_cbar and images1 and images2:
            cbar1 = fig.colorbar(images1[0], ax=ax[0])
            cbar1.ax.tick_params(labelsize=20)
            cbar2 = fig.colorbar(images2[0], ax=ax[1])
            cbar2.ax.tick_params(labelsize=20)

    def plot_nonsplit(self, fig, ax, i, show_cbar):
        image = None
        for j, a in enumerate(ax.flat):
            date = self.dates[j + self.fig_shape[0] * self.fig_shape[1] * i]
            a.set_title(ds.string_time_to_datetime(date), fontsize=20)
            print(date)
            image = self.regional_var_plot(fig, a, date, self.variable, plot_leads=self.plot_leads)

        if show_cbar and image:
            cbar1 = fig.colorbar(image[0], ax=ax[0])
            cbar2 = fig.colorbar(image[2], ax=ax[1])
            cbar1.ax.tick_params(labelsize=20)
            cbar2.ax.tick_params(labelsize=20)


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
    im = ax.pcolormesh(grid.lon, grid.lat, lead.new_leads(), cmap='cool', transform=ccrs.PlateCarree())
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
        data_set = leads.Era5Regrid(leads.Lead(date), 'wind_quiver')  # With this Era5Regrid-class is tested
    elif variable == 'siconc_diff':
        data_set = leads.Era5('siconc')
    elif variable == 'wind_diff':
        data_set = leads.Era5('wind')
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
        im = ax.quiver(lon, lat, v10, u10, np.sqrt(v10**2+u10**2), transform=ccrs.PlateCarree(), cmap=Var.cmap,
                       width=.005, pivot='mid', clim=(0.0, 25.0))  # scale=40, scale_units='inches'
    elif variable == 'siconc_diff' or variable == 'wind_diff' or variable == 'lead_diff':
        # prev_date = ds.string_time_to_datetime(date)
        # prev_date = ds.datetime_to_string(prev_date - datetime.timedelta(days=1))
        data_set.get_var_diff('20200214', '20200225')
        im = ax.pcolormesh(data_set.lon, data_set.lat, np.subtract(data_set.get_variable(date), data_set.var_avg),
                           alpha=Var.alpha, cmap=Var.cmap, transform=ccrs.PlateCarree(), vmin=-20.0, vmax=20.0)
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date), alpha=Var.alpha, cmap=Var.cmap,
                           transform=ccrs.PlateCarree())
        im.set_clim(0, 25)
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
    var3 = ds.variable_daily_avg(date1, date2, extent, variables[2])
    del_index = []

    for i, sic in enumerate(var3):
        if sic > 90:
            del_index.append(i)

    var1 = np.delete(var1, del_index)
    var2 = np.delete(var2, del_index)
    var3 = np.delete(var3, del_index)

    fig, ax = plt.subplots()
    im = ax.scatter(var1, var2, c=var3, cmap='coolwarm')
    ax.set_xlabel(f'{variables[0]}')
    ax.set_ylabel(f'{variables[1]}')
    ax.set_title(f'Daily new leads against cyclone frequency averaged over the Barent sea from '
                 f'{ds.string_time_to_datetime(date1)} to {ds.string_time_to_datetime(date2)}. Color based on SIC.')
    fig.colorbar(im, ax=ax)
    plt.show()


def variable_pixel_pixel(date1, date2, extent):
    dates = ds.time_delta(date1, date2)
    fig, ax = plt.subplots()
    im = None
    lead_dummy = leads.Lead('20200101')
    mask = ds.select_area(leads.CoordinateGrid(), lead_dummy, lead_dummy.lead_frac, extent)[3]

    for date in dates:
        print(date)
        lead = leads.Lead(date)
        cyc = leads.Era5Regrid(lead, 'cyclone_occurence').get_variable(date)[mask]
        sic = leads.Era5Regrid(lead, 'siconc').get_variable(date)[mask]
        lead = lead.new_leads()[mask]

        lead, cyc, sic = hf.filter_by_sic(lead, cyc, sic, 85)
        #im = ax.scatter(lead, sic, c=cyc, cmap='jet')
        mask25 = cyc == 25
        mask0 = cyc == 0
        ax.hist(lead[mask0], bins=30, density=True, alpha=.5)
        ax.hist(lead[mask25], bins=30, density=True, alpha=.5)

    #fig.colorbar(im, ax=ax)
    #ax.set_xlabel('new leads in %')
    #ax.set_ylabel('SIC in %')
    #ax.set_title(f'New leads against SIC pixel by pixel from {ds.string_time_to_datetime(date1)} to '
                 #f'{ds.string_time_to_datetime(date2)}')
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
    # title += f' R = {hf.round_sig(np.corrcoef(y[0], y[1])[0, 1])}'
    title += f' R = {hf.round_sig(dcor.distance_correlation(y[0], y[1]))}'
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
    #regional_var_plot('20200220', show=True, variable=['siconc_diff', 'wind_quiver'], plot_leads=False,
                      #extent=ci.extent1, show_cbar=True)

    RegionalPlot('20200217', '20200322', ['leads', 'msl', 'wind_quiver', 'wind_diff'], extent=ci.extent1, show=True)

    #matrix_plot('20200320', '20200330', 'leads', extent=ci.barent_extent, show=True)
    # plot_lead_cyclone_sum_monthly('20191101', '20200430', no_extent, 'cyclone_occurence')
    # plot_lead_from_vars('20200101', '20200131', ci.arctic_extent, 'cyclone_occurence', 'siconc')

    '''
    for d in ci.Mon:
        variables_against_time(d[0], d[1], ci.arctic_extent, 'leads', 'siconc')
    pass
    '''
    # variables_against_time('20200214', '20200224', ci.barent_extent, 'leads', 'cyclone_occurence', show=False)
    # variable_pixel_pixel('202002017', '20200328', ci.barent_extent)
    # variables_against_time('20200120', '20200210', ci.arctic_extent, 'leads', 'siconc')

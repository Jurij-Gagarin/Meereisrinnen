import case_information as ci
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import leads
import data_science as ds
import numpy as np
import helpful_functions as hf
from scipy.stats import gaussian_kde


def show_plot(fig, file_name, show):
    # Decides if the figure is shown or saved.
    if show:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)


class VarOptions:
    def __init__(self, var):
        self.var = var
        # Dictionaries that assign colors, cmaps, ... to certain variables
        cmap_dict = {'msl': 'Oranges_r', 'cyclone_occurence': 'gray', 'wind': 'cividis', 't2m': 'coolwarm',
                     'leads': 'cool', 'siconc': 'Blues', 'wind_quiver': 'gist_rainbow', 'siconc_diff': 'bwr',
                     'wind_diff': 'bwr', 'lead_diff': 'bwr'}
        alpha_dict = {'msl': 1, 'cyclone_occurence': .15, 'wind': 1, 't2m': 1, 'leads': 1, 'siconc': 1, 'wind_quiver': 1
                      , 'siconc_diff': 1, 'wind_diff': 1, 'lead_diff': 1}
        color_dict = {'msl': 'red', 'leads': 'blue', 'wind': 'orange', 'cyclone_occurence': 'green', 't2m': 'purple',
                      'siconc': 'turquoise', 'wind_quiver': None, 'siconc_diff': None, 'wind_diff': None,
                      'lead_diff': None}
        unit_dict = {'msl': 'hPa', 'leads': '%', 'cyclone_occurence': '%', 'wind': 'm/s', 't2m': '°K', 'siconc': '%',
                     'wind_quiver': 'm/s', 'siconc_diff': '%', 'wind_diff': 'm/s', 'lead_diff': '%'}
        name_dict = {'cyclone_occurence': 'cyclone frequency', 'leads': 'daily new lead fraction', 'wind': 'wind speed',
                     't2m': 'two meter temperature', 'msl': 'mean sea level pressure', 'siconc': 'sea ice concentration'
                     , 'wind_quiver': 'windspeed', 'siconc_diff': 'sea ice concentration difference',
                     'wind_diff': 'wind difference', 'lead_diff': 'lead fraction difference'}
        style_dict = {'msl': 'contour', 'wind': 'mesh', 't2m': 'mesh', 'cyclone_occurence': 'mesh', 'leads': 'mesh',
                      'siconc': 'mesh', 'wind_quiver': 'quiver', 'siconc_diff': 'mesh', 'wind_diff': 'mesh',
                      'lead_diff': 'mesh'}

        self.cmap = cmap_dict[self.var]
        self.alpha = alpha_dict[self.var]
        self.color = color_dict[self.var]
        self.unit = unit_dict[self.var]
        self.name = name_dict[self.var]
        self.style = style_dict[self.var]

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
            if var == 'siconc' or var == 'siconc_diff' or var == 'wind' or var == 'wind_diff' or var == 'lead_diff':
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
            images2 = self.regional_var_plot(fig, a2, date, ['lead_diff'], plot_leads=False)
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


def ds_from_var(variable, date):
    re = None
    if variable == 'wind_quiver':
        from skimage.transform import resize
        re = resize
        data_set = leads.Era5Regrid(leads.Lead(date), 'wind_quiver')  # With this Era5Regrid-class is tested
    elif variable == 'siconc_diff':
        data_set = leads.Era5('siconc')
    elif variable == 'wind_diff':
        data_set = leads.Era5('wind')
    elif variable == 'lead_diff':
        data_set = leads.Lead(date)
    else:
        data_set = leads.Era5(variable)

    return data_set, re


def variable_plot(date, fig, ax, variable, show_cbar):
    # Plots data that is stored in a Era5 grid.
    Var = VarOptions(variable)
    data_set, resize = ds_from_var(variable, date)
    im = None

    if Var.style == 'contour':
        cim = ax.contour(data_set.lon, data_set.lat, data_set.get_variable(date), cmap=Var.cmap,
                         alpha=Var.alpha, transform=ccrs.PlateCarree(), levels=10)
        ax.clabel(cim, inline=True, fontsize=15, inline_spacing=10)
    elif Var.style == 'quiver':
        lat_dir, lon_dir = data_set.get_quiver(date)
        dim = (50, 50)
        lon, lat = resize(data_set.lon, dim), resize(data_set.lat, dim)
        v10, u10 = resize(lon_dir, dim), resize(lat_dir, dim)
        im = ax.quiver(lon, lat, v10, u10, np.sqrt(v10**2+u10**2), transform=ccrs.PlateCarree(), cmap=Var.cmap,
                       width=.005, pivot='mid', clim=(0.0, 18.0))  # scale=40, scale_units='inches'
    elif variable == 'siconc_diff' or variable == 'wind_diff':
        # prev_date = ds.string_time_to_datetime(date)
        # prev_date = ds.datetime_to_string(prev_date - datetime.timedelta(days=1))
        data_set.get_var_diff('20200214', '20200225')
        im = ax.pcolormesh(data_set.lon, data_set.lat, np.subtract(data_set.get_variable(date), data_set.var_avg),
                           alpha=Var.alpha, cmap=Var.cmap, transform=ccrs.PlateCarree(), vmin=-20.0, vmax=20.0)
    elif variable == 'lead_diff':
        avg = leads.lead_avg('20200201', '20200229')
        lon, lat = leads.CoordinateGrid().lon, leads.CoordinateGrid().lat
        im = ax.pcolormesh(lon, lat, np.subtract(leads.Lead(date).new_leads(), avg), vmin=-80.0, vmax=80.0,
                           alpha=Var.alpha, cmap=Var.cmap, transform=ccrs.PlateCarree())
    else:
        im = ax.pcolormesh(data_set.lon, data_set.lat, data_set.get_variable(date), alpha=Var.alpha, cmap=Var.cmap,
                           transform=ccrs.PlateCarree())
        # im.set_clim(0, 25)
    if show_cbar and im:
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
        bins = 30

        im = ax.scatter(sic, lead, c=cyc, cmap='jet', s=25, alpha=.75, vmin=0.0, vmax=100.0)
    fig.colorbar(im)

        #ax.hist(lead[mask0], bins=bins, density=True, alpha=.5)
        #ax.hist(lead[mask25], bins=bins, density=True, alpha=.5)

    #fig.colorbar(im, ax=ax)
    ax.set_xlabel('SIC in %')
    ax.set_ylabel('new leads in %')
    ax.set_title(f'New leads against SIC pixel by pixel from {ds.string_time_to_datetime(date1)} to '
                 f'{ds.string_time_to_datetime(date2)}')
    plt.show()


def variable_pixel_pixel_density(date1, date2, extent):
    dates = ds.time_delta(date1, date2)
    fig, axs = plt.subplots(2, 2)
    lead_dummy = leads.Lead('20200101')
    mask = ds.select_area(leads.CoordinateGrid(), lead_dummy, lead_dummy.lead_frac, extent)[3]

    for ax, cyc_freq in zip(axs.flatten(), [0, 25, 50, 75]):
        print(cyc_freq)
        x = np.array([])
        y = np.array([])
        for date in dates:
            print(date)
            lead = leads.Lead(date)
            cyc = leads.Era5Regrid(lead, 'cyclone_occurence').get_variable(date)[mask]
            cyc_mask = cyc == cyc_freq
            cyc = cyc[cyc_mask]
            sic = leads.Era5Regrid(lead, 'siconc').get_variable(date)[mask][cyc_mask]
            lead = lead.lead_data()[mask][cyc_mask]
            lead, cyc, sic = hf.filter_by_sic(lead, cyc, sic, 90)

            x = np.hstack((x, sic))
            y = np.hstack((y, lead))

        # Calculate the point density
        print(y)
        rm_nan = ~np.isnan(y)
        x, y = x[rm_nan], y[rm_nan]
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        im = ax.scatter(x, y, c=z, s=25)
        ax.set_xlabel('SIC in %')
        ax.set_ylabel('new leads in %')
        ax.set_title(f'cyclone frequency = {cyc_freq}')
        fig.colorbar(im, ax=ax)

    plt.show()


def variables_against_time(date1, date2, extent, var1, var2, spline=False, show=False):
    from dcor import distance_correlation
    # This shows you how two variables change with respect to time.
    dates = ds.string_time_to_datetime(ds.time_delta(date1, date2))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax_twin = ax.twinx()
    Var1, Var2 = VarOptions(var1), VarOptions(var2)
    title = f'Changes in {Var1.name} and {Var2.name} in the {ci.extent_dict[extent]}.'
    y, y_err, i = [], [], 0

    for a, v, Var in zip([ax, ax_twin], [var1, var2], [Var1, Var2]):
        var = ds.variable_daily_avg(date1, date2, extent, v)
        y.append(var[0])
        y_err.append(var[1])

        if spline:
            from scipy.interpolate import CubicSpline
            x = list(range(len(dates)))
            a.scatter(x, y[i], c=Var.color)
            f = CubicSpline(x, y[i], bc_type='natural')
            x_new = np.linspace(0, len(dates), 1000)
            y_new = f(x_new)
            a.plot(x_new, y_new, c=Var.color, linestyle='--')
        else:
            a.plot(dates, y[i], c=Var.color, linestyle='--')
            a.fill_between(dates, (np.array(y[i]) - np.array(y_err[i])), (np.array(y[i]) + np.array(y_err[i])),
                           color=Var.color, alpha=.2)

        a.set_ylabel(f'{Var.name}', fontsize=15)
        a.yaxis.label.set_color(Var.color)
        a.tick_params(axis='y', colors=Var.color, labelsize=15)
        i += 1

    ax.tick_params(axis='x', labelsize=15)
    # title += f' R = {hf.round_sig(np.corrcoef(y[0], y[1])[0, 1])}'
    title += f' R = {hf.round_sig(distance_correlation(y[0], y[1]))}'
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


def plots_for_case(case, extent=None, var=None, plot_lead=True):
    for i, date in enumerate(case):
        print(f'Working on plots for date:{date}')
        regional_var_plot(date, extent=extent, variable=var, plot_leads=plot_lead)


if __name__ == '__main__':
    '''
    for date in ds.time_delta('20200213', '20200224'):
        regional_var_plot(date, show=False, variable=['msl', 'wind_quiver'], plot_leads=True,
                          extent=ci.extent1, show_cbar=True)
    '''

    #RegionalPlot('20200213', '20200322', ['leads', 'msl', 'wind_quiver'], extent=ci.extent1, show=True)

    #matrix_plot('20200320', '20200330', 'leads', extent=ci.barent_extent, show=True)
    # plot_lead_cyclone_sum_monthly('20191101', '20200430', no_extent, 'cyclone_occurence')
    # plot_lead_from_vars('20200101', '20200131', ci.arctic_extent, 'cyclone_occurence', 'siconc')

    '''
    for d in ci.Mon:
        variables_against_time(d[0], d[1], ci.arctic_extent, 'leads', 'siconc')
    pass
    '''
    #,variables_against_time('20200214', '20200224', ci.barent_extent, 'leads', 'cyclone_occurence', show=True)
    variable_pixel_pixel_density('202002017', '20200221', ci.barent_extent)
    # variables_against_time('20200120', '20200210', ci.arctic_extent, 'leads', 'siconc')
    pass

import datetime
import data_science as ds
import cftime
import netCDF4 as nc
import ice_divergence as id
import matplotlib.pyplot as plt
import numpy as np

import case_information as ci
import cartopy.crs as ccrs


class Lead:
    def __init__(self, date):
        # import lead fraction data
        self.date = date
        path = f'./data/leads/{self.date}.nc'
        ds_lead = nc.Dataset(path)
        self.lead_frac = ds_lead['Lead Fraction'][:]
        self.old_shape = self.lead_frac.shape

        # assign instances later needed
        self.del_row, self.del_col = [], []
        self.land, self.water = None, None
        self.cloud, self.lead_data = None, None

        # sort data clean up date from rows/cols without entries
        self.sort_matrix()

    def sort_matrix(self):
        # Creates lead frac matrix that contains only the data-points
        self.land, self.water = np.copy(self.lead_frac), np.copy(self.lead_frac)
        self.cloud, self.lead_data = np.copy(self.lead_frac), np.copy(self.lead_frac)

        self.land[self.land != np.float32(1.2)] = np.nan
        self.water[self.water != np.float32(-0.1)] = np.nan
        self.cloud[self.cloud != np.float32(-0.2)] = np.nan
        self.lead_data[self.lead_data > 1] = np.nan
        self.lead_data[self.lead_data < 0] = np.nan

    def new_leads(self):
        prior_date = ds.string_time_to_datetime(self.date)
        prior_date -= datetime.timedelta(days=1)
        prior_date = ds.datetime_to_string(prior_date)
        try:
            lead1 = Lead(prior_date).lead_data
            lead2 = self.lead_data
            new_lead = lead2 - lead1

            return 100 * new_lead.clip(min=0)
        except FileNotFoundError:
            print(f'No data available for date prior {ds.string_time_to_datetime(self.date)}.')
            print('I could therefor not calculate the new leads.')


class CoordinateGrid:
    def __init__(self):
        lead = Lead('20200101')
        # import corresponding coordinates
        path_grid = './data/leads/LatLonGrid.nc'
        ds_latlon = nc.Dataset(path_grid)
        self.lat = ds_latlon['Lat Grid'][:]
        self.lon = ds_latlon['Lon Grid'][:]

    def vals(self):
        # Method used to generate grid description, should not be used anymore
        np.savetxt('yvals.txt', self.lat.flatten(), delimiter=' ')
        np.savetxt('xvals.txt', self.lon.flatten(), delimiter=' ')

    def plot_grid(self, extent):
        #ccrs.NorthPolarStereo(-45)
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.Orthographic(0, 90)})
        ax.gridlines()
        ax.set_global()
        ax.coastlines(resolution='50m')
        extent = extent if extent else ci.arctic_extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        print(np.ones(self.lon.shape).shape)
        ax.scatter(self.lon, self.lat, transform=ccrs.PlateCarree(), s=1, alpha=1)
        plt.show()

    def get_weights(self):
        dim = self.lon.shape
        weights = np.empty(dim)

        for i in range(dim[0] - 1):
            for j in range(dim[1] - 1):
                #dphi = self.lat[i+1, j+1] - self.lat[i, j]
                #weights[i, j] = abs(dphi*(np.cos(np.radians(self.lon[i, j])) - np.cos(np.radians(self.lon[i+1, j+1]))))
                weights[i, j] = abs(np.sin(np.radians(self.lat[i, j])))

        return weights

    def plot_weights(self, extent=None):
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": ccrs.NearsidePerspective(-45,90)})
        ax.gridlines()
        ax.set_global()
        ax.coastlines(resolution='50m')
        extent = extent if extent else ci.arctic_extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        im = ax.pcolormesh(self.lon, self.lat, self.get_weights(), transform=ccrs.NearsidePerspective())
        fig.colorbar(im)
        plt.show()


class Era5:
    def __init__(self, variable):
        # import air pressure data
        self.var = variable
        variable_dict = {'msl': 'data/ERA5_METAs.nc', 'wind': 'data/ERA5_METAs.nc',
                         't2m': 'data/ERA5_METAs.nc', 'siconc': 'data/ERA5_METAs.nc',
                         'cyclone_occurence': 'data/ERA5_METAs.nc', 'wind_quiver': 'data/ERA5_METAs.nc'}

        path = variable_dict[self.var]
        data_set = nc.Dataset(path)
        print(data_set)

        # Assign variables
        if self.var == 'wind_quiver':
            self.u10 = data_set.variables['u10']
            self.v10 = data_set.variables['v10']
        else:
            self.variable = data_set.variables[self.var]
        self.time = data_set['time']
        self.var_avg = None

        # Build grid matrix
        self.lon = np.tile(data_set['longitude'][:], (161, 1))
        self.lat = np.transpose(np.tile(data_set['latitude'][:], (1440, 1)))

    def get_variable(self, date):
        # Get time index
        # datetime(year, month, day, hour, minute, second, microsecond)
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)
        # Calculate mean variable of the given date
        if self.var == 'wind_quiver':
            mean_u10, mean_v10 = np.zeros(self.u10[0].shape), np.zeros(self.v10[0].shape)
            for t in range(t1, t2 + 1):
                mean_u10 += self.u10[t]
                mean_v10 += self.v10[t]
            return .25 * mean_u10, .25 * mean_v10

        else:
            mean_var = np.zeros(self.variable[0].shape)
            for t in range(t1, t2 + 1):
                mean_var = np.add(mean_var, self.variable[t])
            return ds.variable_manip(self.var, .25 * mean_var)

    def get_variable_drift(self, date):
        # Get time index
        # datetime(year, month, day, hour, minute, second, microsecond)
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0) - datetime.timedelta(hours=12)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0) + datetime.timedelta(hours=12)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        # Calculate mean variable of the given date
        mean_var = np.zeros(self.variable[0].shape)
        for t in range(t1, t2 + 1):
            mean_var = np.add(mean_var, self.variable[t])
        return ds.variable_manip(self.var, 1/len(list(range(t1, t2 + 1))) * mean_var)

    def get_quiver(self, date):
        # Get time index
        # datetime(year, month, day, hour, minute, second, microsecond)
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        # Calculate mean variable of the given date
        mean_v10 = np.zeros(self.v10[0].shape)
        mean_u10 = np.zeros(self.u10[0].shape)
        for t in range(t1, t2 + 1):
            mean_v10 = np.add(mean_v10, self.v10[t])
            mean_u10 = np.add(mean_u10, self.u10[t])
        return .25 * mean_v10, .25 * mean_u10

    def get_var_diff(self, date1, date2):
        dates = ds.time_delta(date1, date2)

        avg = np.zeros(self.variable[0].shape)
        for date in dates:
            avg += self.get_variable(date)

        self.var_avg = avg / len(dates)

    def get_div(self, date):
        u10, v10 = self.get_variable(date)
        du, dv = id.matrix_neighbour_diff(u10, v10)
        return (du + dv)/60000


class Era5Regrid:
    def __init__(self, variable):
        # import air pressure data
        variable_dict = {'msl': 'data/ERA5_METAs_remapbil.nc', 'wind': 'data/ERA5_METAs_remapbil.nc',
                         't2m': 'data/ERA5_METAs_remapbil.nc', 'siconc': 'data/ERA5_METAs_remapbil.nc',
                         'cyclone_occurence': 'data/ERA5_METAs_remapbil.nc',
                         'wind_quiver': 'data/ERA5_METAs_remapbil.nc'}

        self.var = variable
        path = variable_dict[self.var]

        data_set = nc.Dataset(path)
        print(data_set)
        self.shape = Lead('20200101').old_shape
        self.time = data_set['time']
        self.lon = np.reshape(data_set.variables['lon'], self.shape)
        self.lat = np.reshape(data_set.variables['lat'], self.shape)

        if self.var == 'wind_quiver':
            self.u10 = data_set.variables['u10']
            self.v10 = data_set.variables['v10']
        else:
            self.variable = data_set.variables[self.var]

    def get_variable(self, date):
        dt_date = ds.string_time_to_datetime(date)
        print(dt_date)
        if dt_date.year == 2019:
            variable_dict = {'msl': 'data/ERA5_METAw_remapbil.nc', 'wind': 'data/ERA5_METAw_remapbil.nc',
                             't2m': 'data/ERA5_METAw_remapbil.nc', 'siconc': 'data/ERA5_METAw_remapbil.nc',
                             'cyclone_occurence': 'data/ERA5_METAw_remapbil.nc',
                             'wind_quiver': 'data/ERA5_METAw_remapbil.nc'}
            path = variable_dict[self.var]

            data_set = nc.Dataset(path)
            self.shape = Lead('20200101').old_shape
            self.time = data_set['time']
            self.lon = np.reshape(data_set.variables['lon'], self.shape)
            self.lat = np.reshape(data_set.variables['lat'], self.shape)

            if self.var == 'wind_quiver':
                self.u10 = data_set.variables['u10']
                self.v10 = data_set.variables['v10']
            else:
                self.variable = data_set.variables[self.var]

        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        new_shape = self.lon.shape
        mean_variable = np.zeros(new_shape)
        for t in range(t1, t2 + 1):
            add_msl = np.reshape(self.variable[t], self.shape)
            mean_variable = np.add(mean_variable, add_msl)
        return ds.variable_manip(self.var, .25 * mean_variable)

    def get_quiver(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        new_shape = self.lon.shape
        mean_v10 = np.zeros(new_shape)
        mean_u10 = np.zeros(new_shape)
        for t in range(t1, t2 + 1):
            add_v10 = np.reshape(self.v10[t], self.shape)
            add_u10 = np.reshape(self.u10[t], self.shape)
            mean_v10 = np.add(mean_v10, add_v10)
            mean_u10 = np.add(mean_u10, add_u10)
        return .25 * mean_v10, .25 * mean_u10


class LeadAllY:
    def __init__(self, date, path=None):
        # import lead fraction data
        self.date = date[:4] + '_' + date[4:]
        dt_date = ds.string_time_to_datetime(date)
        dt_datem1, dt_datep1 = dt_date - datetime.timedelta(days=1), dt_date + datetime.timedelta(days=1)
        datem1, datep1 = ds.datetime_to_string(dt_datem1), ds.datetime_to_string(dt_datep1)
        path_lead = f'./data/DailyArcticLeadFraction_12p5km_Rheinlaender/data/LeadFraction_12p5km_{self.date}.nc'
        path_cyc = f'./data/CO_2_remapbil.nc'
        path_sic = f'./data/ERA5_SIC_2000_2019_remapbil.nc'
        path_drift = f'./data/ice drift/Eumetsat/2010-2022-remapbil/'
        path_drift += f'remapbil_ice_drift_nh_polstere-625_multi-oi_{datem1}1200-{datep1}1200.nc'
        ds_lead = nc.Dataset(path_lead)
        ds_cyc = nc.Dataset(path_cyc)
        ds_sic = nc.Dataset(path_sic)

        if path:
            path_drift = path
        try:
            ds_drift = nc.Dataset(path_drift)
        except FileNotFoundError:
            ds_drift = None

        dt_date = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 9, 0, 0)
        d = cftime.date2index(dt_date, ds_cyc['time'])
        d_sic = cftime.date2index(dt_date, ds_sic['time'])

        self.lead_data = ds_lead['Lead Fraction'][:]
        self.lead_data[self.lead_data == 1] = np.nan
        self.cyc_data = ds_cyc['cyclone_occurence'][d]
        self.cyc_data = self.cyc_data.reshape(self.lead_data.shape)
        self.sic_data = ds_sic['siconc'][d_sic]
        self.sic_data = self.sic_data.reshape(self.lead_data.shape)

        try:
            self.u = ds_drift['dX'][:].reshape(self.lead_data.shape).T * 1000/172800
            self.v = ds_drift['dY'][:].reshape(self.lead_data.shape).T * 1000/172800
            self.u[self.u.mask] = np.nan
            self.v[self.v.mask] = np.nan
            self.ice_div = id.divergence(np.array([self.u, self.v]), [12000, -12000])
        except ValueError:
            print('could not find ice divergence data')
            pass
        except TypeError:
            print(f'failed to collect ice drift data for {self.date}')
            self.u, self.v = np.empty(self.lead_data.shape), np.empty(self.lead_data.shape)
            self.ice_div = np.empty(self.lead_data.shape)

        '''try:
            print('normal:')
            self.xc = ds_drift['xc'][:] * 1000
            self.yc = ds_drift['yc'][:] * 1000
            self.xx, self.yy = np.meshgrid(self.xc, self.yc, indexing='ij')
            self.u = ds_drift['dX'][0, :].T * 1000 / 172800
            self.v = ds_drift['dY'][0, :].T * 1000 / 172800
            self.u[self.u.mask] = np.nan
            self.v[self.v.mask] = np.nan
            print(self.xx.shape, self.u.shape)
        except IndexError:
            print('normal failed')
            pass'''


class CoordinateGridAllY:
    def __init__(self):
        # import corresponding coordinates
        path_grid = './data/DailyArcticLeadFraction_12p5km_Rheinlaender/LeadFraction_12p5km_LatLonGrid_subset.nc'
        ds_latlon = nc.Dataset(path_grid)
        # for remaped divergence this must be transposed
        self.lat = ds_latlon['lat'][:]
        self.lon = ds_latlon['lon'][:]
        #im = plt.imshow(np.absolute(self.lon))
        #plt.colorbar(im)
        #print(self.lon)
        #plt.show()

    def vals(self):
        # Method used to generate grid description, should not be used anymore
        print(self.lon.flatten().shape, self.lat.flatten().shape)
        np.savetxt('yvals.txt', self.lat.flatten(), delimiter=' ')
        np.savetxt('xvals.txt', self.lon.flatten(), delimiter=' ')


def lead_avg(date1, date2):
    dates = ds.time_delta(date1, date2)
    leads_list = np.array([Lead(date).new_leads() for date in dates])

    return np.nanmean(leads_list, axis=0)


def lead_avg_diff(date, avg):
    lead = Lead(date)

    return np.subtract(avg, lead.lead_data)


if __name__ == '__main__':

    LAY = LeadAllY('20150101')
    CG = CoordinateGridAllY()
    lon, lat = CG.lon, CG.lat

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    ax.coastlines(resolution='50m')
    ax.set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    ax.set_title('divergence, test')

    # print(ds.time_delta2('20021101', '20210430'))
    # LeadAllY('20201230')
    #Era5Regrid('cyclone_occurence')

    pass
'''
date = '20180105'
    # date_p = date[:4] + '_' + date[4:]
    dt_date = ds.string_time_to_datetime(date)
    dt_datem1, dt_datep1 = dt_date - datetime.timedelta(days=1), dt_date + datetime.timedelta(days=1)
    datem1, datep1 = ds.datetime_to_string(dt_datem1), ds.datetime_to_string(dt_datep1)
    path = './data/ice drift/Eumetsat/2010-2022/'
    path += f'ice_drift_nh_polstere-625_multi-oi_{datem1}1200-{datep1}1200.nc'

    CG = CoordinateGridAllY()
    lon, lat = CG.lon, CG.lat
    fig, axs = plt.subplots(2, 4, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))
    axs = axs.flatten()

    axs[0].coastlines(resolution='50m')
    axs[0].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[0].set_title('u, remaped')
    LAY = LeadAllY(date)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    print(lon[skip].shape, lat[skip].shape, LAY.u[skip].shape)
    axs[0].pcolormesh(lon[skip], lat[skip], LAY.u[skip], transform=ccrs.PlateCarree())

    axs[4].coastlines(resolution='50m')
    axs[4].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[4].set_title('u')
    LAY = LeadAllY(date, path)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    axs[4].pcolormesh(LAY.xx[skip], LAY.yy[skip], LAY.u[skip], transform=ccrs.NorthPolarStereo(-45))

    axs[1].coastlines(resolution='50m')
    axs[1].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[1].set_title('v, remaped')
    LAY = LeadAllY(date)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    axs[1].pcolormesh(lon[skip], lat[skip], LAY.v[skip], transform=ccrs.PlateCarree())

    axs[5].coastlines(resolution='50m')
    axs[5].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[5].set_title('v')
    LAY = LeadAllY(date, path)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    axs[5].pcolormesh(LAY.xx[skip], LAY.yy[skip], LAY.v[skip], transform=ccrs.NorthPolarStereo(-45))

    axs[2].coastlines(resolution='50m')
    axs[2].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[2].set_title('quiv, remaped')
    LAY = LeadAllY(date)
    n_skip = 7
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    axs[2].quiver(lon[skip], lat[skip], LAY.u[skip], LAY.v[skip], transform=ccrs.PlateCarree())

    axs[6].coastlines(resolution='50m')
    axs[6].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[6].set_title('quiv')
    LAY = LeadAllY(date, path)
    n_skip = 2
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    axs[6].quiver(LAY.xx[skip], LAY.yy[skip], LAY.u[skip], LAY.v[skip], transform=ccrs.NorthPolarStereo(-45))
    # plt.savefig('test_quiver')

    axs[3].coastlines(resolution='50m')
    axs[3].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[3].set_title('div, remaped')
    LAY = LeadAllY(date)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    im = axs[3].pcolormesh(lon[skip], lat[skip], id.divergence(np.array([LAY.u, LAY.v]), [12000, -12000]),
                           transform=ccrs.PlateCarree(), cmap='bwr', vmin=-2.e-6, vmax=2.e-6)
    fig.colorbar(im, ax=axs[3])

    axs[7].coastlines(resolution='50m')
    axs[7].set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
    axs[7].set_title('div')
    LAY = LeadAllY(date, path)
    n_skip = None
    skip = (slice(None, None, n_skip), slice(None, None, n_skip))
    im = axs[7].pcolormesh(LAY.xx[skip], LAY.yy[skip], id.divergence(np.array([LAY.u, LAY.v]), [62500, -62500]),
                           transform=ccrs.NorthPolarStereo(-45), cmap='bwr', vmin=-2.e-6, vmax=2.e-6)
    fig.colorbar(im, ax=axs[7])
    plt.tight_layout()
    plt.savefig(f'test_div_remap_{date}')
'''





''' generate image of lattice structure
CG = CoordinateGridAllY()
lon, lat = CG.lon, CG.lat
fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.NorthPolarStereo(-45)}, figsize=(15, 10))

ax1.coastlines(resolution='50m')
ax1.set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
ax1.set_title('Leads lattice structure')
struc = np.arange(lon.size).reshape(lon.shape)
print(struc)
im = ax1.pcolormesh(lon, lat, struc, cmap='jet', transform=ccrs.PlateCarree())
fig.colorbar(im, ax=ax1, orientation='horizontal')
plt.savefig('lattice_structure_leads')



ax2.coastlines(resolution='50m')
ax2.set_extent(ci.arctic_extent, crs=ccrs.PlateCarree())
ax2.set_title('Drift lattice structure')
date = '20180108'
# date_p = date[:4] + '_' + date[4:]
dt_date = ds.string_time_to_datetime(date)
dt_datem1, dt_datep1 = dt_date - datetime.timedelta(days=1), dt_date + datetime.timedelta(days=1)
datem1, datep1 = ds.datetime_to_string(dt_datem1), ds.datetime_to_string(dt_datep1)
path = './data/ice drift/Eumetsat/2010-2022/'
path += f'ice_drift_nh_polstere-625_multi-oi_{datem1}1200-{datep1}1200.nc'
LAY = LeadAllY(date, path)
im = ax2.pcolormesh(LAY.xx, LAY.yy, np.arange(LAY.xx.size).reshape(LAY.xx.shape), transform=ccrs.NorthPolarStereo(-45))
fig.colorbar(im, ax=ax2, orientation='horizontal')

plt.tight_layout()
plt.savefig('lattice_structure_leads')'''




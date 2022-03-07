import datetime
import data_science as ds
import cftime
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np


class Lead:
    def __init__(self, date):
        # import lead fraction data
        self.date = date
        path = f'./data/{self.date}.nc'
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
        path_grid = './data/LatLonGrid.nc'
        ds_latlon = nc.Dataset(path_grid)
        self.lat = ds_latlon['Lat Grid'][:]
        self.lon = ds_latlon['Lon Grid'][:]

    def vals(self):
        # Method used to generate grid description, should not be used anymore
        np.savetxt('yvals.txt', self.lat.flatten(), delimiter=' ')
        np.savetxt('xvals.txt', self.lon.flatten(), delimiter=' ')


class Era5:
    def __init__(self, variable):
        # import air pressure data
        self.var = variable
        variable_dict = {'msl': 'data/ERA5_MSLP_2020_JanApr.nc', 'wind': 'data/ERA5_meta_1920.nc',
                         't2m': 'data/ERA5_T2m_2020_JanApr_new.nc', 'siconc': 'data/ERA5_meta_1920.nc',
                         'cyclone_occurence': 'data/cyc_time.nc', 'wind_quiver': 'data/ERA5_Wind_2020_JanApr.nc'}

        path = variable_dict[self.var]
        data_set = nc.Dataset(path)

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
        mean_var = np.zeros(self.variable[0].shape)
        for t in range(t1, t2 + 1):
            mean_var = np.add(mean_var, self.variable[t])
        return ds.variable_manip(self.var, .25 * mean_var)

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


class Era5Regrid:
    def __init__(self, lead, variable):
        # import air pressure data
        variable_dict = {'msl': 'data/ERA5_2020_MSL_regrid_bil.nc', 'wind': 'data/ERA5_meta_1920_regrid.nc',
                         't2m': 'data/ERA5_2020_T2m_regrid_bil.nc', 'siconc': 'data/ERA5_meta_1920_regrid.nc',
                         'cyclone_occurence': 'data/cyc_time_regrid.nc',
                         'wind_quiver': 'data/ERA5_Wind_2020_JanApr_regrid.nc'}

        self.var = variable
        path = variable_dict[self.var]
        self.lead = lead

        data_set = nc.Dataset(path)
        self.shape = lead.old_shape
        self.time = data_set['time']
        self.lon = np.reshape(data_set.variables['lon'], self.shape)
        self.lat = np.reshape(data_set.variables['lat'], self.shape)

        if self.var == 'wind_quiver':
            self.u10 = data_set.variables['u10']
            self.v10 = data_set.variables['v10']
        else:
            self.variable = data_set.variables[self.var]

        #self.lon = ds.clear_matrix(self.lon, lead.del_row, lead.del_col)
        #self.lat = ds.clear_matrix(self.lat, lead.del_row, lead.del_col)

    def get_variable(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        new_shape = self.lon.shape
        mean_variable = np.zeros(new_shape)
        for t in range(t1, t2 + 1):
            add_msl = np.reshape(self.variable[t], self.shape)
            #add_msl = ds.clear_matrix(add_msl, self.lead.del_row, self.lead.del_col)
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


def lead_avg(date1, date2):
    dates = ds.time_delta(date1, date2)
    leads_list = np.array([Lead(date).new_leads() for date in dates])

    return np.nanmean(leads_list, axis=0)


def lead_avg_diff(date, avg):
    lead = Lead(date)

    return np.subtract(avg, lead.lead_data)



if __name__ == '__main__':
    #Era5('msl')
    #Era5('siconc')
    #Era5('t2m')
    #Era5('wind_quiver').get_quiver('20200101')
    #Era5('cyclone_occurence')

    avg = lead_avg('20200213', '20200224')
    print(avg.shape)
    plt.imshow(avg)
    plt.show()

    diff = lead_avg_diff('20200220', avg)
    im = plt.imshow(diff)
    plt.colorbar(im)
    plt.show()

    pass






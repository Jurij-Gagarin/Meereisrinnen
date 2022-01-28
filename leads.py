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
        self.clear_matrix()
        self.sort_matrix()

    def clear_matrix(self, trigger=-0.1):
        # Returns indices of rows and columns without any data
        trigger = np.float32(trigger)
        m, n = self.lead_frac.shape
        row_clear, col_clear = np.repeat(trigger, n), np.repeat(trigger, m)

        for i, row in enumerate(self.lead_frac):
            if np.array_equal(row, row_clear):
                self.del_row.append(i)
        for i in range(n):
            if np.array_equal(self.lead_frac[:, i], col_clear):
                self.del_col.append(i)

        self.lead_frac = np.delete(self.lead_frac, self.del_row, 0)
        self.lead_frac = np.delete(self.lead_frac, self.del_col, 1)

    def visualize_matrix(self, file_name=None, show=False):
        # very simple visualization of the lead fraction matrix
        # This was used for testing. Might be removed in the future.
        fig, ax = plt.subplots(figsize=(10, 10))
        if not file_name:
            file_name = f'./plots/{self.date}.png'
        im = ax.imshow(self.lead_frac, cmap='cool')
        im.cmap.set_over('#dddddd')
        im.cmap.set_under('#00394d')

        im.set_clim(0, 1)
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title(f'Sea ice leads {self.date[6:]}.{self.date[4:6]}.{self.date[:4]}')
        if show:
            plt.show()
        plt.savefig(file_name)
        plt.close(fig)

    def sort_matrix(self):
        # Creates lead frac matrix that contains only the data-points
        self.land, self.water = np.copy(self.lead_frac), np.copy(self.lead_frac)
        self.cloud, self.lead_data = np.copy(self.lead_frac), np.copy(self.lead_frac)

        self.land[self.land != np.float32(1.2)] = np.nan
        self.water[self.water != np.float32(-0.1)] = np.nan
        self.cloud[self.cloud != np.float32(-0.2)] = np.nan
        self.lead_data[self.lead_data > 1] = np.nan
        self.lead_data[self.lead_data < 0] = np.nan


class CoordinateGrid:
    def __init__(self, lead):
        # import corresponding coordinates
        path_grid = './data/LatLonGrid.nc'
        ds_latlon = nc.Dataset(path_grid)
        self.lat = ds_latlon['Lat Grid'][:]
        self.lon = ds_latlon['Lon Grid'][:]
        self.lon = ds.clear_matrix(self.lon, lead.del_row, lead.del_col)
        self.lat = ds.clear_matrix(self.lat, lead.del_row, lead.del_col)

    def vals(self):
        np.savetxt('yvals.txt', self.lat.flatten(), delimiter=' ')
        np.savetxt('xvals.txt', self.lon.flatten(), delimiter=' ')


class AirPressure:
    def __init__(self, path=None):
        # import air pressure data
        if not path:
            path = 'data/ERA5_MSLP_2020_JanApr.nc'

        ds = nc.Dataset(path)
        self.msl = ds.variables['msl']
        self.time = ds['time']

        self.lon = np.tile(ds['longitude'][:], (161, 1))
        self.lat = np.transpose(np.tile(ds['latitude'][:], (1440, 1)))

    def get_msl(self, date):
        # Get time index
        # datetime(year, month, day, hour, minute, second, microsecond)
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        # Calculate mean msl of the given date
        mean_msl = np.zeros(self.msl[0].shape)
        for t in range(t1, t2 + 1):
            mean_msl += self.msl[t]
        return .0025 * mean_msl


class Era5Regrid:
    def __init__(self, lead, path=None):
        # import air pressure data
        if not path:
            path = 'data/ERA5_2020_regrid_bil.nc'
        self.lead = lead

        data_set = nc.Dataset(path)
        self.shape = lead.old_shape
        self.time = data_set['time']
        self.lon = np.reshape(data_set.variables['lon'], self.shape)
        self.lat = np.reshape(data_set.variables['lat'], self.shape)
        self.msl = data_set.variables['msl']

        self.lon = ds.clear_matrix(self.lon, lead.del_row, lead.del_col)
        self.lat = ds.clear_matrix(self.lat, lead.del_row, lead.del_col)

    def get_msl(self, date):
        d1 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 0, 0, 0, 0)
        d2 = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 18, 0, 0, 0)
        t1, t2 = cftime.date2index([d1, d2], self.time)

        new_shape = self.lon.shape
        mean_msl = np.zeros(new_shape)
        for t in range(t1, t2 + 1):
            add_msl = np.reshape(self.msl[t], self.shape)
            add_msl = ds.clear_matrix(add_msl, self.lead.del_row, self.lead.del_col)
            mean_msl += add_msl
        return .0025 * mean_msl


if __name__ == '__main__':
    lead = Lead('20200217')
    test = Era5Regrid(lead)
    print(test.get_msl('20200217'))

    grid = CoordinateGrid(lead)







import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt


class Lead:
    def __init__(self, date):
        # import lead fraction data
        self.date = date
        path = f'./data/{self.date}.nc'
        ds_lead = nc.Dataset(path)
        self.lead_frac = ds_lead['Lead Fraction'][:]

    def visualize_matrix(self, file_name=None, show=False):
        # very simple visualization of the lead fraction matrix
        if not file_name:
            file_name = f'./plots/{self.date}.png'
        plt.imshow(self.lead_frac, cmap='RdYlBu')
        plt.colorbar()
        plt.axis('off')
        if show:
            plt.show()
        plt.savefig(file_name)


class CoordinateGrid:
    def __init__(self):
        # import corresponding coordinates
        path_grid = './data/LatLonGrid.nc'
        ds_latlon = nc.Dataset(path_grid)
        self.lat = ds_latlon['Lat Grid'][:]
        self.lon = ds_latlon['Lon Grid'][:]


if __name__ == '__main__':
    lead1 = Lead('20191101')
    lead1.visualize_matrix()


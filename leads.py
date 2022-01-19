import netCDF4 as nc
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
        fig, ax = plt.subplots(figsize=(10,10))
        if not file_name:
            file_name = f'./plots/{self.date}.png'
        #'RdYlBu_r'
        im = ax.imshow(self.lead_frac, cmap='cool')
        im.cmap.set_over('#dddddd')
        im.cmap.set_under('#002633')

        im.set_clim(0, 1.0)
        fig.colorbar(im, ax=ax)
        ax.axis('off')
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
    days = list(range(17, 22))
    dates = [f'202002{day}' for day in days]
    for date in dates:
        lead = Lead(date)
        lead.visualize_matrix()

import leads
import netCDF4 as nc


class OnlyLeadAllY:
    def __init__(self):
        pass


class DriftAllY:
    def __init__(self):
        pass


class CycAllY:
    def __init__(self):
        pass


class CoordinateGridAllY:
    def __init__(self):
        # import corresponding coordinates
        path_grid = './data/DailyArcticLeadFraction_12p5km_Rheinlaender/LeadFraction_12p5km_LatLonGrid_subset.nc'
        ds_latlon = nc.Dataset(path_grid)
        # for remaped divergence this must be transposed
        self.lat = ds_latlon['lat'][:]
        self.lon = ds_latlon['lon'][:]
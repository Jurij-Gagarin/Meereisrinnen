import leads
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
import scipy
import scipy.ndimage


def string_time_to_datetime(dates):
    if not isinstance(dates, list):
        return date(int(dates[:4]), int(dates[4:6]), int(dates[6:]))
    else:
        return [date(int(d[:4]), int(d[4:6]), int(d[6:])) for d in dates]


def time_delta(date1, date2):
    # Returns a list that contains all dates between date1 and date2 in a '20200101.nc' way.
    days = []
    start_date = date(int(date1[:4]), int(date1[4:6]), int(date1[6:]))
    end_date = date(int(date2[:4]), int(date2[4:6]), int(date2[6:]))

    delta = end_date - start_date   # returns timedelta

    for i in range(delta.days + 1):
        day = str(start_date + timedelta(days=i))
        days.append(''.join(c for c in day if c not in '-'))

    return days


def compare_nan(arr1, arr2):
    # Returns array 1/2 * (arr1 + arr2). If one element is nan its treated like 0.
    arr1[np.isnan(arr1)] = 0
    arr2[np.isnan(arr2)] = 0

    arr = .5 * (arr1 + arr2)
    arr[arr == 0] = np.nan
    arr[~np.isnan(arr)] = 1
    return arr


def sum_nan_arrays(arr1, arr2):
    # Returns arr1 + arr2 element wise. If one element is NaN it's treated like 0. If both elements are Nan, the return
    # is Nan as well.
    ma = np.isnan(arr1)
    mb = np.isnan(arr2)
    return np.where(ma & mb, np.nan, np.where(ma, 0, arr1) + np.where(mb, 0, arr2))


def two_lead_diff(lead1, lead2):
    # Used to calculate day to day difference in the lead fraction
    lead2.lead_data = lead2.lead_data - lead1.lead_data
    lead2.cloud = compare_nan(lead1.cloud, lead2.cloud)
    lead2.water = compare_nan(lead1.water, lead2.water)
    lead2.land = compare_nan(lead1.land, lead2.land)


def clear_matrix(matrix, rows, cols):
    matrix = np.delete(matrix, rows, 0)
    matrix = np.delete(matrix, cols, 1)
    return matrix


def variable_manip(var, matrix):
    # This method does small manipulations (unit change) to data from Era5 that is stored in a lead fraction like matrix
    if var == 'msl':
        return .01 * matrix
    else:
        return matrix


def select_area(grid, lead, matrix, points):
    # The goal of this method is to select data points in a certain area. It picks only matrix elements, where lead
    # fraction matrix has real enties (not NaN).
    lat = grid.lat
    lon = grid.lon
    mask_lat_a = lat >= points[3]
    mask_lat_b = lat <= points[2]
    mask_lon_a = lon <= points[0]
    mask_lon_b = lon >= points[1]
    mask_water = np.isnan(lead.water)
    mask_land = np.isnan(lead.land)
    mask_cloud = np.isnan(lead.cloud)
    mask = mask_lon_a & mask_lat_b & mask_lat_a & mask_lon_b & mask_land & mask_water & mask_cloud
    matrix[mask == False] = None

    return lon, lat, matrix


def variable_average(date1, date2, extent, variable, filter_data=False):
    # This Method calculates the average of the cyclone_occurence matrix within the range of date1,2.
    # Optionally the data can be filtered via Gaussian. You may want to set different sigma values
    dates = time_delta(date1, date2)
    grid = leads.CoordinateGrid()
    cum_var = np.full(grid.lon.shape, np.nan)
    count_values = np.zeros(grid.lon.shape)

    for date in dates:
        lead = leads.Lead(date)
        var = leads.Era5Regrid(lead, variable)
        var = var.get_variable(date)
        var = select_area(grid, lead, var, extent)[2]
        cum_var = sum_nan_arrays(cum_var, var)
        row, col = np.where(~np.isnan(var))
        count_values[row, col] += 1
    cum_var = cum_var / count_values
    if filter_data:
        cum_var = scipy.ndimage.filters.gaussian_filter(cum_var, [1.0, 1.0], mode='constant', order=0)
    return cum_var


def lead_average(date1, date2, extent):
    # This Method calculates the average of the lead data matrix within the range of date1,2.
    dates = time_delta(date1, date2)
    grid = leads.CoordinateGrid()
    cum_leads = np.full(grid.lon.shape, np.nan)
    count_values = np.zeros(grid.lon.shape)

    for date in dates:
        lead = leads.Lead(date)
        lead = select_area(grid, lead, lead.lead_data, extent)[2]
        cum_leads = sum_nan_arrays(cum_leads, lead)
        row, col = np.where(~np.isnan(lead))
        count_values[row, col] += 1

    cum_leads = cum_leads / count_values
    return cum_leads


def lead_monthly_average(year, month, extent):
    # This method calculates the monthly average of lead fraction.
    time = str(year) + '-' + str(month).zfill(2)
    dates = np.arange(time, time[:-2] + str(int(time[-2:])+1).zfill(2), dtype='datetime64[D]')
    dates = [str(date).replace('-', '')for date in dates]   # gives a list of '20200101' like dates, for chosen month
    grid = leads.CoordinateGrid()
    monthly_leads = np.full(grid.lon.shape, np.nan)
    count_values = np.zeros(grid.lon.shape)
    N = len(dates)

    for date in dates:
        lead = leads.Lead(date)
        lead = select_area(grid, lead, lead.lead_data, extent)[2]
        row, col = np.where(~np.isnan(lead))
        count_values[row, col] += 1
        monthly_leads = sum_nan_arrays(monthly_leads, lead)

    return dates


if __name__ == '__main__':
    #lead_hist('20200218')
    case1 = ['20200216', '20200217', '20200218', '20200219', '20200220', '20200221', '20200222']
    extent1 = [-90, 100, 65, 90]
    case2 = ['20200114', '20200115', '20200116', '20200117', '20200118', '20200119', '20200120']
    extent2 = None
    case3 = ['20200128', '20200129', '20200130', '20200131', '20200201', '20200202', '20200203']
    extent3 = None
    case4 = ['20200308', '20200309', '20200310', '20200311', '20200312', '20200313', '20200314', '20200315', '20200316']
    extent4 = None

    #lead_average('20200101', '20200228')

    print(lead_monthly_average(2020, 2, extent1))


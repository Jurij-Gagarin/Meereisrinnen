import leads
import numpy as np
import matplotlib.pyplot as plt


def compare_nan(arr1, arr2):
    # Returns array 1/2 * (arr1 + arr2). If one element is nan its treated like 0.
    arr1[np.isnan(arr1)] = 0
    arr2[np.isnan(arr2)] = 0

    arr = .5 * (arr1 + arr2)
    arr[arr == 0] = np.nan
    arr[~np.isnan(arr)] = 1
    return arr


def two_lead_diff(lead1, lead2):
    lead2.lead_data = lead2.lead_data - lead1.lead_data
    lead2.cloud = compare_nan(lead1.cloud, lead2.cloud)
    lead2.water = compare_nan(lead1.water, lead2.water)
    lead2.land = compare_nan(lead1.land, lead2.land)


def clear_matrix(matrix, rows, cols):
    matrix = np.delete(matrix, rows, 0)
    matrix = np.delete(matrix, cols, 1)
    return matrix


def variable_manip(var, matrix):
    if var == 'msl':
        return .01 * matrix
    else:
        return matrix


def select_area(grid, lead, matrix, points=(90, -30, 85, 70)):
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
    #lon[mask == False] = 100
    #lat[mask == False] = 100
    matrix[mask == False] = None
    #plt.imshow(matrix)
    #plt.show()
    print(mask)
    print(type(matrix))
    print(matrix)
    return lon, lat, matrix


if __name__ == '__main__':
    select_area(leads.CoordinateGrid(), leads.Lead('20200217'), np.ones(leads.CoordinateGrid().lat.shape))

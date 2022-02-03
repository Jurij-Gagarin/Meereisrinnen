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


def select_area(grid, lead, matrix, points=(90, 0, 90, 70)):
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


def cyclone_trace():
    pass


def lead_hist(date):
    lead = leads.Lead(date)
    grid = leads.CoordinateGrid()
    lead_data = lead.lead_data
    cyclone = leads.Era5Regrid(lead, 'cyclone_occurence').get_variable(date)

    lead_data = select_area(grid, lead, lead_data)[2]
    cyclone = select_area(grid, lead, cyclone)[2]
    #plt.imshow(cyclone)

    for i in [0, .5, 1]:
        mask = cyclone == i
        print(lead_data[mask], len(lead_data[mask]))
        plt.hist(lead_data[mask], density=True, bins=100, alpha=.25, label=f'{i}, {np.mean(lead_data[mask])}, N={len(lead_data[mask])}')
    cyclone = np.ceil(cyclone)
    mask = cyclone == 1
    plt.hist(lead_data[mask], density=True, bins=20, alpha=.25, label=f'ceil, {np.mean(lead_data[mask])}, N={len(lead_data[mask])}')
    plt.legend()


    plt.show()



if __name__ == '__main__':
    #lead_hist('20200218')
    case1 = ['20200216', '20200217', '20200218', '20200219', '20200220', '20200221', '20200222']
    extent1 = [-70, 100, 65, 90]
    case2 = ['20200114', '20200115', '20200116', '20200117', '20200118', '20200119', '20200120']
    extent2 = None
    case3 = ['20200128', '20200129', '20200130', '20200131', '20200201', '20200202', '20200203']
    extent3 = None
    case4 = ['20200308', '20200309', '20200310', '20200311', '20200312', '20200313', '20200314', '20200315', '20200316']
    extent4 = None

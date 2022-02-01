import leads
import numpy as np


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
    if var == 'cyclone_occurence':
        return np.ceil(matrix)
    elif var == 'msl':
        return .01 * matrix
    elif var == 'u10':
        return matrix
    elif var == 't2m':
        return matrix


def cyclone_analysis():
    dates = ['202001' + str(d).zfill(2) for d in list(range(1, 6))]
    print(dates)

if __name__ == '__main__':
    cyclone_analysis()

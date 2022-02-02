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
    if var == 'cyclone_occurence':
        return matrix
    elif var == 'msl':
        return .01 * matrix
    elif var == 'u10':
        return matrix
    elif var == 't2m':
        return matrix


def cyclone_analysis():
    co = 'cyclone_occurence'
    dates = ['202002' + str(d).zfill(2) for d in list(range(19, 22))]
    lead_dummy = leads.Lead('20200220')
    cyclone_cells = np.array([])
    non_cyclone_cells = np.array([])
    for date in dates:
        lead_fraction = leads.Lead(date).lead_data
        mask = leads.Era5Regrid(lead_dummy, co).get_variable(date) == 1
        cyclone_cells = np.hstack([cyclone_cells, lead_fraction[mask]])
        non_cyclone_cells = np.hstack([non_cyclone_cells, lead_fraction[~mask]])

    a = non_cyclone_cells[~np.isnan(non_cyclone_cells)]
    b = cyclone_cells[~np.isnan(cyclone_cells)]
    plt.hist(a, alpha=.5, bins=100, density=True)
    plt.hist(b, alpha=.5, bins=100, density=True)
    plt.show()


if __name__ == '__main__':
    cyclone_analysis()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy
import numpy as np
import leads

extent = [-50, 120, 60, 90]
#ax = plt.axes(projection=ccrs.NorthPolarStereo(0))
#ax.set_extent(extent, crs=ccrs.PlateCarree())
#ax.gridlines()
#ax.set_global()
#ax.coastlines(resolution='50m')

grid = leads.CoordinateGrid()
date = '20200217'
lead = leads.Lead(date)

print(np.min(grid.lat), np.max(grid.lat))
print(np.min(grid.lon), np.max(grid.lon))
lat = grid.lat.flatten()
lon = grid.lon.flatten()
#plt.contourf(grid.lon, grid.lat, numpy.zeros(np.shape(grid.lat)), cmap='RdYlBu')
plt.scatter(lat, lon)
plt.show()

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

plt.contourf(grid.lon, grid.lat, lead.lead_frac, cmap='RdYlBu')

plt.show()

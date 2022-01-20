import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import leads

extent = [-50, 120, 60, 90]
ax = plt.axes(projection=ccrs.NorthPolarStereo(0))
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.gridlines()
ax.set_global()
ax.coastlines(resolution='50m')

grid = leads.CoordinateGrid()
date = '20200217'
lead = leads.Lead(date)
lead.clear_matrix()
grid.clear_grid(lead.del_row, lead.del_col)
print(grid.lon.shape, grid.lat.shape, lead.lead_frac.shape)
print(type(lead.lead_frac))
# np.random.rand(441, 481)
ax.contourf(grid.lon, grid.lat, lead.lead_frac, cmap='RdYlBu', transform=ccrs.PlateCarree())

plt.show()

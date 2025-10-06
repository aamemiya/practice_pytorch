import sys
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import basemap  
from mpl_toolkits.basemap import Basemap,cm 
import netCDF4

ncfile=netCDF4.Dataset("../121240-11TT00-0000-20250715000000.nc")
#print(ncfile.variables)
lon=ncfile.variables['lon'][:]
lat=ncfile.variables['lat'][:]
t2m=ncfile.variables['TT'][:][0]
# Lambert Conformal Conic map.
fig,ax=plt.subplots()
lons,lats = np.meshgrid(lon,lat) # compute map proj coordinates.
#print(lats.shape)
#print(t2m.shape)
#quit()

t2m_colors = [
    "#9933ff", 
    "#6633ff",  
    "#3333ff",  
    "#3366ff",  
    "#3399ff",  
    "#33ccff",  
    "#33ffcc",  
    "#33ff99",  
    "#33ff66",  
    "#33ff33",  
    "#66ff33",  
    "#ccff33",  
    "#ffff33",  
    "#ffcc33",  
    "#ff9933",  
    "#ff6633",  
    "#ff3333",  
]
t2m_colormap = matplotlib.colors.ListedColormap(t2m_colors[1:-1])
t2m_colormap.set_over(t2m_colors[-1])
t2m_colormap.set_under(t2m_colors[0])

t2m_levels = np.arange(-10,35,3)
t2m_norm=matplotlib.colors.BoundaryNorm(t2m_levels, len(t2m_levels))

# draw filled contours.

cs = ax.contourf(lons,lats,t2m,levels=t2m_levels,norm=t2m_norm, cmap=t2m_colormap,extend="both")

# add colorbar.
cbar = fig.colorbar(cs,location='bottom')
cbar.set_label('C')

#ax.clabel(cs,cs.levels,fontsize="small")

plt.title('Test')
figname="sample.png"
fig.savefig(figname)
plt.clf()

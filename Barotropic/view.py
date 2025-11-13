import sys
import glob
import numpy as np
import netCDF4
import matplotlib
import matplotlib.pyplot as plt


indir=sys.argv[1]
files=glob.glob(indir + "/state_phys*.nc")

vor_colors = [
    "#000066", 
    "#0000cc", 
    "#0000ff", 
    "#0066ff",  
    "#3399ff",  
    "#66ccff",  

    "#ffffff", 
    "#ffffff", 

    "#ffff99", 
    "#ffcc66", 
    "#ff9933", 
    "#ff3300",  
    "#ff0000",  
    "#800000",  
]
vor_colormap = matplotlib.colors.ListedColormap(vor_colors[1:-1])
vor_colormap.set_over(vor_colors[-1])
vor_colormap.set_under(vor_colors[0])

vor_levels = np.arange(-12,14,2)
vor_norm=matplotlib.colors.BoundaryNorm(vor_levels, len(vor_levels))




it=0
for f in files :

### nature run
  nc = netCDF4.Dataset(f,'r')

#  print(nc)
#  print(nc.groups['state_phys'])
#for var in nc.variables[:]:
#    print(var)
#  quit()
  vor = np.array(nc.groups['state_phys'].variables['rot'][:])
  nx = np.array(nc.groups['state_phys'].dimensions['x'].size)
  ny = np.array(nc.groups['state_phys'].dimensions['y'].size)
  nc.close

#vor=np.random.rand(256,256)

#print(vor)
#print(np.max(vor),np.min(vor))


#plt.contourf()

  x1=np.linspace(1,nx,num=ny)
  y1=np.linspace(1,ny,num=ny)
  x,y=np.meshgrid(x1,y1)

  cs=plt.pcolormesh(x,y,vor, norm=vor_norm,cmap=vor_colormap)
#  cs=plt.contourf(x,y,vor,levels=vor_levels,norm=vor_norm, cmap=vor_colormap,extend="both")
  cbar = plt.colorbar(cs,location='right')
  #cbar.set_label('m/s')
  fname="test_"+str(it).zfill(3)+".png"
  print(fname)
  plt.savefig(fname)
  plt.clf()
  it += 1


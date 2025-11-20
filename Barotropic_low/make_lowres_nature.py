import os
import sys
import glob
import numpy as np
import netCDF4


dir_high = "nature/control"
dir_low ="nature/x2"
 
paths=glob.glob(dir_high + "/state_phys_*.nc")

for path in paths :
    f=os.path.basename(path)
    nc = netCDF4.Dataset(dir_high + "/" + f, 'r')
    nc_low = netCDF4.Dataset(dir_low + "/" + f,'r+')

#  print(nc)
#  print(nc.groups['state_phys'])
#for var in nc.variables[:]:
#    print(var)
#  quit()
    ux = np.array(nc.groups['state_phys'].variables['ux'][:])
    uy = np.array(nc.groups['state_phys'].variables['uy'][:])
    vor = np.array(nc.groups['state_phys'].variables['rot'][:])
    nx = np.array(nc.groups['state_phys'].dimensions['x'].size)
    ny = np.array(nc.groups['state_phys'].dimensions['y'].size)

    nx_low = np.array(nc_low.groups['state_phys'].dimensions['x'].size)
    ny_low = np.array(nc_low.groups['state_phys'].dimensions['y'].size)

    nmx=int(nx/nx_low)
    nmy=int(ny/ny_low)

    vor_low=np.zeros((ny_low,nx_low))
    ux_low=np.zeros((ny_low,nx_low))
    uy_low=np.zeros((ny_low,nx_low))


    for ix in range(nmx):
        for iy in range(nmy):
            ux_low[:,:] += ux[iy::nmy,ix::nmx]
            uy_low[:,:] += uy[iy::nmy,ix::nmx]
            vor_low[:,:] += vor[iy::nmy,ix::nmx]
    ux_low /= nmx*nmy
    uy_low /= nmx*nmy
    vor_low /= nmx*nmy

    nc_low.groups['state_phys'].variables['ux'][:] = ux_low
    nc_low.groups['state_phys'].variables['uy'][:] = uy_low
    nc_low.groups['state_phys'].variables['rot'][:] = vor_low

#print(np.max(ux),np.min(ux))
#print(np.max(ux_low),np.min(ux_low))
    print(np.max(vor),np.min(vor))
    print("low aft",np.max(vor_low),np.min(vor_low))
#quit()
    nc.close
    nc_low.close


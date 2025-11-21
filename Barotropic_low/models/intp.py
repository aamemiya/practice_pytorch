import os, sys, glob
import netCDF4 
import numpy as np
import matplotlib.pyplot as plt

# model size
nx  = 256
nx_low  = 128

ncdir="../nature/orig/"
ncdir_low="../nature/x2/"

files=glob.glob(ncdir + "/state_phys*.nc")
files_low=glob.glob(ncdir_low + "/state_phys*.nc")

usmp=[]
vsmp=[]
ulsmp=[]
vlsmp=[]


for f,fl in zip(files, files_low) : 
    
# load nature 
    nc = netCDF4.Dataset(f,'r',format='NETCDF4')
    ncl = netCDF4.Dataset(fl,'r',format='NETCDF4')
    usmp.append(np.array(nc["state_phys"].variables['ux'][:]))
    vsmp.append(np.array(nc["state_phys"].variables['uy'][:]))
    ulsmp.append(np.array(ncl["state_phys"].variables['ux'][:]))
    vlsmp.append(np.array(ncl["state_phys"].variables['uy'][:]))
    nc.close 
    ncl.close 

usmp=np.array(usmp)
vsmp=np.array(vsmp)
ulsmp=np.array(ulsmp)
vlsmp=np.array(vlsmp)

ursmp=np.full_like(usmp,0)
vrsmp=np.full_like(vsmp,0)

for i in range(2):
  for j in range(2):
    ursmp[:,i::2,j::2]=ulsmp[:,:,:]

nsmp=usmp.shape[0]

uabs=np.sqrt(usmp**2+vsmp**2)
ulabs=np.sqrt(ulsmp**2+vlsmp**2)

umean=np.mean(uabs,axis=0)
ulmean=np.mean(ulabs,axis=0)

print(np.max(umean),np.max(ulmean))



#cs=plt.pcolormesh(wmat[:,:-1])
#cbar = plt.colorbar(cs,location='right')
#plt.savefig("wmat.png")




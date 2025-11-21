import os, sys, glob
import netCDF4 
import numpy as np
import matplotlib.pyplot as plt

# model size
nx  = 128
nx_low  = 64

ncdir="/data/Barotropic/output/nature/control/"
ncdir_low="/data/Barotropic/output/forecast_low/"

files=glob.glob(os.path.join(ncdir + "/state_phys*.nc"))
tstamp_smp=files[0].split("/")[-1].replace("state_phys_t","").split(".")[0:2]
digit_all=str(len(tstamp_smp[0])+len(tstamp_smp[1])+1)
digit_dec=len(tstamp_smp[1])

#for dir_fcst_low in dirs_fcst_low:
    
dir_fcst_low=os.path.join(ncdir_low +"000400.000")
files_low=glob.glob(os.path.join(dir_fcst_low + "/state_phys*.nc"))

var_orig="000400.200"
print()

usmp=[]
vsmp=[]
ulsmp=[]
vlsmp=[]

width = 9
precision = 3
x = 400.2

#formatted = f"{x:0{width}.{precision}f}"
#print(formatted)
#quit()

#digit_all=10
#digit_dec=3
#ftstamp=400.2
#aa=format(f"{ftstamp:0{digit_all}.{digit_dec}f}")
#print(aa)
#quit()

for fl in files_low : 
    
# load nature 
    tstamp=fl.split("/")[-1].replace("state_phys_t","").replace(".nc","")
    ftstamp=float(tstamp)
    tstamp_n=format(f"{ftstamp:0{digit_all}.{digit_dec}f}")
   # print(digit_all,digit_dec)
   # print(tstamp)
   # print(tstamp_n)

    fn = os.path.join(ncdir,fl.split("/")[-1]).replace(tstamp,tstamp_n)
    nc  = netCDF4.Dataset(fn,'r',format='NETCDF4')
    ncl = netCDF4.Dataset(fl,'r',format='NETCDF4')
    ulsmp.append(np.array(ncl["state_phys"].variables['ux'][:]))
    vlsmp.append(np.array(ncl["state_phys"].variables['uy'][:]))
    usmp.append(np.array(nc["state_phys"].variables['ux'][:]))
    vsmp.append(np.array(nc["state_phys"].variables['uy'][:]))
    nc.close 
    ncl.close 

usmp=np.array(usmp)
vsmp=np.array(vsmp)
ulsmp=np.array(ulsmp)
vlsmp=np.array(vlsmp)

ulsmp_ds=usmp.copy()
vlsmp_ds=vsmp.copy()
ulsmp_ds[:,0::2,0::2]=ulsmp
ulsmp_ds[:,1::2,0::2]=ulsmp
ulsmp_ds[:,0::2,1::2]=ulsmp
ulsmp_ds[:,1::2,1::2]=ulsmp
vlsmp_ds[:,0::2,0::2]=vlsmp
vlsmp_ds[:,1::2,0::2]=vlsmp
vlsmp_ds[:,0::2,1::2]=vlsmp
vlsmp_ds[:,1::2,1::2]=vlsmp


#### RMSE as a function of FT
for j in range(usmp.shape[0]):
    var_mean= np.sqrt ( np.sum(( (ulsmp_ds[j]-usmp[j])**2 + (vlsmp_ds[j]-vsmp[j])**2 )) / float(usmp.shape[-1]*usmp.shape[-2]))
    print(j, var_mean)
  # print(j,np.max(usmp[j]-ulsmp_ds[j]),np.min(usmp[j]-ulsmp_ds[j]))


#### confirm the downscaled field doesn't have positional dependency 
for it in [0, 6, 12]: 
  knl=np.array((2,2))
  smp_knl=[]
  for j in range(nx_low):
      for i in range(nx_low): 
          smp_knl.append(  ulsmp_ds[it,2*j:2*(j+1),2*i:2*(i+1)] - usmp[it,2*j:2*(j+1),2*i:2*(i+1)]  ) 

  smp_knl=np.array(smp_knl)
  smp_mean=np.mean(smp_knl,axis=0)
  smp_std=np.std(smp_knl,axis=0)
  print("it = ",it)
  print(smp_mean)
  print(smp_std)
quit()





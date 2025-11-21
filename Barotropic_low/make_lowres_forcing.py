import os 
import numpy as np, math
from netCDF4 import Dataset
from fluidsim.solvers.ns2d.solver import Simul

# --- open a NetCDF file to load the forcing ---
path_nc_in = os.path.join(os.getcwd(),"forcings","forcing_x2.nc")
nc = Dataset(path_nc_in, "r")

vrot_in=nc["rot_forcing"]
forcing_rot_r_in=nc["rot_fft_forcing_r"]
forcing_rot_i_in=nc["rot_fft_forcing_i"]
vtime_in=nc["time"]
nx=nc.dimensions['x'].size
ny=nc.dimensions['y'].size
nt=nc.dimensions['time'].size
#print(nt)
#quit()
# --- open a NetCDF file to log the forcing ---
path_nc_out = os.path.join(os.getcwd(),"forcings","forcing_x4.nc")
nc = Dataset(path_nc_out, "w")
nc.createDimension("time", None)
nc.createDimension("y", ny//2)
nc.createDimension("x", nx/2)
nc.createDimension("l", ny//2)
nc.createDimension("k", nx//4+1)
vtime = nc.createVariable("time", "f8", ("time",))
vrot  = nc.createVariable("rot_forcing", "f4", ("time","y","x"), zlib=True)
forcing_rot_r_out  = nc.createVariable("rot_fft_forcing_r", "f4", ("time","l","k"), zlib=True)
forcing_rot_i_out  = nc.createVariable("rot_fft_forcing_i", "f4", ("time","l","k"), zlib=True)

def reduce(array_in):
    top = array_in[ : ny//4, : nx//4 + 1]
    bottom = array_in[ ny-ny//4:, : nx//4 + 1]
    return np.vstack([top,bottom])

for it in range(nt):
    if np.mod(it,100) == 0 :
        print(it)
    vtime[it]=vtime_in[it]
    vrot[it,:,:]=vrot_in[it,::2,::2]
    forcing_rot_r_out[it,:,:]=reduce(forcing_rot_r_in[it,:,:])
    forcing_rot_i_out[it,:,:]=reduce(forcing_rot_i_in[it,:,:])
    

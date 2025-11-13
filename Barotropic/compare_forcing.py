from fluidsim.solvers.ns2d.solver import Simul
import sys
import glob
import numpy as np
import netCDF4
import matplotlib
import matplotlib.pyplot as plt

figname="low_forcing"
ncname="forcing_test_low.nc"

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


nc = netCDF4.Dataset(ncname,'r')
forcing_rot_r=nc["rot_fft_forcing_r"]
forcing_rot_i=nc["rot_fft_forcing_i"]
nx=nc.dimensions["x"].size
ny=nc.dimensions["y"].size

params = Simul.create_default_params()
params.oper.nx = nx 
params.oper.ny = ny
params.oper.Lx = params.oper.Ly = 2.0 * 3.141592653589793
sim = Simul(params)


for it in range(0,4000,200):
   forcing_fft_rot = forcing_rot_r[it] + 1j * forcing_rot_i[it]
   forcing_rot = sim.oper.ifft(forcing_fft_rot)
#   print(it,np.max(forcing_rot))

   x1=np.linspace(1,nx,num=ny)
   y1=np.linspace(1,ny,num=ny)
   x,y=np.meshgrid(x1,y1)

   cs=plt.pcolormesh(x,y,forcing_rot, norm=vor_norm,cmap=vor_colormap)
   cbar = plt.colorbar(cs,location='right')
   fname=figname+"_"+str(it).zfill(4)+".png"
   print(fname)
   plt.savefig(fname)
   plt.clf()


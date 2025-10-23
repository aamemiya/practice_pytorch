import os
import netCDF4 
import numpy as np
import model 
import param

# model size
nx  = param.param_model['dimension'] 
f   = param.param_model['forcing']
dt  = param.param_model['dt']
# integration
exp_length = param.param_exp['exp_length']
expdir = param.param_exp['expdir']
dt_nature = param.param_exp['dt_nature']

amp = 0.0 

intv_nature=int(dt_nature/dt)

### nature run
nc = netCDF4.Dataset(expdir + '/spinup/init_nature.nc','r',format='NETCDF4')
x0 = np.array(nc.variables['v'][:], dtype=type(np.float64))
nc.close 

l96=model.Lorenz96(nx, f, dt, init_x = x0,amp_const=amp)
nature = []
time_nature = []

time_now=0
nature.append(l96.x.copy())
time_nature.append(time_now)

irec=0
for i in range(exp_length*intv_nature):
  l96.runge_kutta() 
  time_now += dt
  if (round(time_now/dt_nature,4).is_integer()):  
    irec+=1
    print(str(irec) + ' / ' +  str(exp_length))
    nature.append(l96.x.copy())
    time_nature.append(round(time_now,6))
nature = np.array(nature, dtype=np.float64)
time_nature = np.array(time_nature, dtype=np.float64)

nc = netCDF4.Dataset(expdir + '/nature.nc','w',format='NETCDF3_CLASSIC')
nc.createDimension('x',nx)
nc.createDimension('t',None)
x_in = nc.createVariable('x',np.dtype('float64').char,('x'))
t_in = nc.createVariable('t',np.dtype('float64').char,('t'))
v_in = nc.createVariable('v',np.dtype('float64').char,('t','x'))
x_in[:] = np.array(range(1,nx+1))
t_in[:] = time_nature
v_in[:,:] = nature
nc.close 



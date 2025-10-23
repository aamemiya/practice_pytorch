import os
import netCDF4 
import numpy as np
import numpy.linalg as LA             
import param

import matplotlib.pyplot as plt

# integration
ncdir = param.param_exp['expdir']

# load nature 
nc = netCDF4.Dataset('../' + ncdir + '/nature.nc','r',format='NETCDF4')
v = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

#nsmp=data_input.shape[0]
nsmp=3001
nval=1001

data_input=v[:nsmp-1,:]
data_output=v[1:nsmp,:]-v[:nsmp-1,:]

data_input_val=v[nsmp-1:nsmp+nval-1,:]
data_output_val=v[nsmp:nsmp+nval,:]-v[nsmp-1:nsmp+nval-1,:]


def calc_val(wmat): 
   predict_val = data_input_val @ wmat[:,:-1].transpose() + np.ones((nval,1)) @ np.array([wmat[:,-1]])
   rmse = np.sqrt(np.mean((predict_val - data_output_val)**2))
   return rmse
#print(data_input.shape)
#print(np.max(data_input),np.min(data_input))
#print(data_output.shape)
#print(np.max(data_output),np.min(data_output))
#quit()
data_input_ext=np.concatenate((data_input,np.ones((data_input.shape[0],1))),axis=1)

wmat=np.random.randn(data_output.shape[1],data_input_ext.shape[1])

#print(calc_val(wmat).shape)
#quit()
eps=0.000001 
val_bef=9.99e10
for j in range(1000) :
  grad= (data_output.transpose() - wmat @ data_input_ext.transpose() ) @ data_input_ext
  wmat = wmat + eps * grad
  val=calc_val(wmat)
  print(val)
  if abs(val-val_bef) < 0.001 : 
    break
  val_bef=val
for j in range(100,200):
  data_predict=[data_input[j]] @ wmat[:,:-1].transpose()  + [wmat[:,-1]]
 # print(data_predict[0][1:3],data_output[j][1:3])
  print(np.sqrt(np.mean((data_output[j]-data_predict[0])**2)),np.sqrt(np.mean(data_output[j]**2)))


cs=plt.pcolormesh(wmat[:,:-1])
cbar = plt.colorbar(cs,location='right')
plt.savefig("wmat.png")




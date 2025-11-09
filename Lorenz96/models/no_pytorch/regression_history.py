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

### 
nhist=3 ### default: 1

#nsmp=data_input.shape[0]
nsmp=3001

data_input=[]
for ih in range(nhist,0,-1):
  data_input.append(v[nhist-ih:nsmp-ih,:])

data_input=np.array(data_input)
nx=data_input.shape[-1]

data_input=np.transpose(data_input,axes=(1,0,2)).reshape((nsmp-nhist,nhist*nx))
data_output=v[nhist:nsmp,:]-v[nhist-1:nsmp-1,:]

#print(data_input.shape)
#print(np.max(data_input),np.min(data_input))
#print(data_output.shape)
#print(np.max(data_output),np.min(data_output))
#quit()
data_input_ext=np.concatenate((data_input,np.ones((data_input.shape[0],1))),axis=1)

bmatinv=LA.inv(data_input_ext.transpose() @ data_input_ext)

wmat = data_output.transpose() @ ( data_input_ext @ bmatinv )

print(wmat.shape)

for j in range(100,200):
  data_predict=[data_input[j]] @ wmat[:,:-1].transpose()  + [wmat[:,-1]]
  print(data_predict[0][1:3],data_output[j][1:3])
 # print(np.sqrt(np.mean((data_output[j]-data_predict[0])**2)),np.sqrt(np.mean(data_output[j]**2)))
#print(data_input_ext.shape)


cs=plt.pcolormesh(wmat[:,:-1])
cbar = plt.colorbar(cs,location='right')
plt.savefig("wmat.png")




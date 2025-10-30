import os
import netCDF4 
import numpy as np
import numpy.linalg as LA             
import param

import matplotlib.pyplot as plt

import torch
from torch import nn

# model size
nx  = param.param_model['dimension'] 
# integration
ncdir = param.param_exp['expdir']

# load nature 
nc = netCDF4.Dataset('../' + ncdir + '/nature.nc','r',format='NETCDF4')
v = np.array(nc.variables['v'][:], dtype=type(np.float64)).astype(np.float32)
time = np.array(nc.variables['t'][:], dtype=type(np.float64)).astype(np.float32)
nc.close 

### 
local_dist=3

###

#nsmp=data_input.shape[0]
nsmp=3001

data_input=v[:nsmp-1,:]
data_output=v[1:nsmp,:]-v[:nsmp-1,:]

print(data_input.shape)
print(np.max(data_input),np.min(data_input))
print(data_output.shape)
print(np.max(data_output),np.min(data_output))

data_input_local=[]
for shift in range(-local_dist,local_dist+1): 
  data_input_local.append(np.roll(data_input,shift,axis=1))
data_input_local=np.array(data_input_local).reshape(local_dist*2+1,data_input.size).transpose()
#print(data_input_local.shape)
data_input_local=np.concatenate((data_input_local,np.ones((data_input_local.shape[0],1))),axis=1)


data_output_local=np.array(data_output).reshape(1,data_output.size).transpose()

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
            nn.Linear(2*local_dist+2,2*local_dist+2, bias=False),
            nn.ReLU(),
            nn.Linear(2*local_dist+2,1, bias=True),
    )
  def forward(self, x):
    y = self.layers(x)
    return y

model=nnmodel()
loss_L2=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)

wmat_torch=next(model.parameters())
print("wmat_torch init",wmat_torch.tolist())

torch_data_input=torch.from_numpy(data_input_local.astype(np.float32))
torch_data_output=torch.from_numpy(data_output_local.astype(np.float32))

loss=loss_L2(model(torch_data_input),torch_data_output)
print("Loss step=0",loss)

model.train()
for j in range(2000):
  loss=loss_L2(model(torch_data_input),torch_data_output)

  # Backpropagation
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  if np.mod(j,100) == 0 : 
     print("Loss step= ",j,loss.item())

wmat_torch=next(model.parameters())
print("wmat_torch trained",wmat_torch.tolist())


bmatinv=LA.inv(data_input_local.transpose() @ data_input_local)

wmat = data_output_local.transpose() @ ( data_input_local @ bmatinv )

print("wmat trained",wmat.tolist())

quit()

### Varidation

for j in range(100,200):
  data_smp_local=[]
  for shift in range(-local_dist,local_dist+1): 
    data_smp_local.append(np.roll(data_input[j],shift))
  data_smp_local=np.array(data_smp_local).transpose()
  #data_smp_local=np.array(data_smp_local).swapaxes(0,-1)
  #data_smp_local=np.concatenate((data_smp_local,np.ones((data_smp_local.shape[0],1))),axis=1)
  #print(data_smp_local.shape)
  #print(wmat.shape)
  data_predict=data_smp_local @ wmat[:,:-1].transpose()  + wmat[:,-1]
  data_predict=data_predict.transpose()
#  print(data_smp_local.shape)
#  print(wmat.shape)
#  print(data_output.shape)
#  print(data_predict.shape)
#  quit()
 # print(data_predict[0][1:3],data_output[j][1:3])
  print(np.sqrt(np.mean((data_output[j]-data_predict[0])**2)),np.sqrt(np.mean(data_output[j]**2)))
#print(data_input_ext.shape)


cs=plt.pcolormesh(wmat[:,:-1])
cbar = plt.colorbar(cs,location='right')
plt.savefig("wmat.png")




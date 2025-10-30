import os
import netCDF4 
import numpy as np
import numpy.linalg as LA             
import param

import matplotlib.pyplot as plt

import torch
from torch import nn


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

class nnmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
            nn.Linear(data_input_ext.shape[-1],40, bias=False),
            nn.ReLU(),
            nn.Linear(40,data_output.shape[-1], bias=False),
    )
  def forward(self, x):
    y = self.layers(x)
    return y

model=nnmodel()
loss_L2=nn.MSELoss()
optimizer= torch.optim.SGD(model.parameters(),lr=1.0e-2)

torch_data_input=torch.from_numpy(data_input_ext.astype(np.float32))
torch_data_output=torch.from_numpy(data_output.astype(np.float32))

loss=loss_L2(model(torch_data_input),torch_data_output)
print("Loss step=0",loss)

model.train()
for j in range(10000):
  loss=loss_L2(model(torch_data_input),torch_data_output)

  # Backpropagation
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()

  if np.mod(j,100) == 0 : 
     print("Loss step= ",j,loss.item())

model.eval()
for j in range(100,200):
  data_predict=model(torch_data_input[j].unsqueeze(0)).detach().numpy() 
  #print(data_predict[0][1:3],data_output[j][1:3])
  print(np.sqrt(np.mean((data_output[j]-data_predict[0])**2)),np.sqrt(np.mean(data_output[j]**2)))
quit()




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



